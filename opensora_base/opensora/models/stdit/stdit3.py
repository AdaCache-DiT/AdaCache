import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from transformers import PretrainedConfig, PreTrainedModel

from opensora_base.opensora.acceleration.checkpoint import auto_grad_checkpoint
from opensora_base.opensora.acceleration.communications import gather_forward_split_backward, split_forward_gather_backward
from opensora_base.opensora.acceleration.parallel_states import get_sequence_parallel_group
from opensora_base.opensora.models.layers.blocks import (
    Attention,
    CaptionEmbedder,
    MultiHeadCrossAttention,
    PatchEmbed3D,
    PositionEmbedding2D,
    SeqParallelAttention,
    SeqParallelMultiHeadCrossAttention,
    SizeEmbedder,
    T2IFinalLayer,
    TimestepEmbedder,
    approx_gelu,
    get_layernorm,
    t2i_modulate,
)
from opensora_base.opensora.registry import MODELS
from opensora_base.opensora.utils.ckpt_utils import load_checkpoint


class STDiT3Block(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        rope=None,
        qk_norm=False,
        temporal=False,
        enable_flash_attn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        blk_id = 0,
        num_sampling_steps = 100,
        do_cache = False,
        cache_res = 't-attn',
        cache_loc = [13],
        codebook = {0.03: 12, 0.05: 10, 0.07: 8, 0.09: 6, 0.11: 4, 1.00: 3},
        apply_moreg=False,
        moreg_strides=[1],
        moreg_steps=(10, 90),
        moreg_hyp=(0.385, 8, 1,2),
        mograd_mul=10,
    ):
        super().__init__()
        self.temporal = temporal
        self.hidden_size = hidden_size
        self.enable_flash_attn = enable_flash_attn
        self.enable_sequence_parallelism = enable_sequence_parallelism

        if self.enable_sequence_parallelism and not temporal:
            attn_cls = SeqParallelAttention
            mha_cls = SeqParallelMultiHeadCrossAttention
        else:
            attn_cls = Attention
            mha_cls = MultiHeadCrossAttention

        self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn = attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
            rope=rope,
            enable_flash_attn=enable_flash_attn,
        )
        self.cross_attn = mha_cls(hidden_size, num_heads)
        self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

        self.blk_id = blk_id
        self.num_sampling_steps = num_sampling_steps
        self.fwd_id = 0

        self.st_attn_cache, self.ca_mlp_cache = 0, 0
        self.attn_cache_rate, self.mlp_cache_rate = 1, 1
        self.attn_next_step, self.mlp_next_step = 2, 2
        self.attn_recomputed_steps, self.mlp_recomputed_steps = [], []
        
        self.prev_compute_step = 0
        self.prev_moreg = 1.
        
        self.do_cache = do_cache
        self.cache_res = cache_res
        self.cache_loc = cache_loc
        self.codebook = codebook

        self.apply_moreg = apply_moreg
        self.moreg_strides = moreg_strides
        self.moreg_steps = moreg_steps
        self.moreg_hyp = moreg_hyp
        self.mograd_mul = mograd_mul


    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x


    def compute_next_step(
            self, cache, res, codebook, prev_rate, ada_dict,fwd_id, 
            norm_ord, verbose, which_module, spatial_dim, enable_sequence_parallelism=False
        ):
        if enable_sequence_parallelism:
            sp_group = get_sequence_parallel_group()
            cache = gather_forward_split_backward(cache, sp_group, dim=1)
            res = gather_forward_split_backward(res, sp_group, dim=1)

        # distance metric [synced across layers if needed]
        cache_diff = (cache - res).norm(dim=(0,1,2), p=norm_ord) / cache.norm(dim=(0,1,2), p=norm_ord)
        # normalize across steps
        cache_diff = cache_diff / prev_rate
        #step_cache_diff = cache_diff
        # sync across layers
        cache_diff = (cache_diff + ada_dict['count'] * ada_dict['avg_diff']) / (ada_dict['count'] + 1)
        ada_dict['avg_diff'], ada_dict['count'] = cache_diff, ada_dict['count'] + 1

        # motion regularization
        if self.apply_moreg:
            if self.moreg_steps[0] <= fwd_id <= self.moreg_steps[1]:
                moreg = 0
                for i in self.moreg_strides: # motion across multiple frame-rates
                    moreg_i = (res[:, i*spatial_dim:, :] - res[:, :-i*spatial_dim, :]).norm(p=norm_ord) 
                    moreg_i /= (res[:, i*spatial_dim:, :].norm(p=norm_ord) + res[:, :-i*spatial_dim, :].norm(p=norm_ord)) # normalize
                    moreg += moreg_i
                moreg = moreg / len(self.moreg_strides)

                # normalized around 1.
                # reduce cache-rate if >1., increase otherwise
                moreg = ((1/self.moreg_hyp[0] * moreg) ** self.moreg_hyp[1]) / self.moreg_hyp[2] 
            
            else:
                moreg = 1.

            # motion gradient
            mograd = self.mograd_mul * (moreg - self.prev_moreg) / prev_rate
            self.prev_moreg = moreg
            moreg = moreg + abs(mograd)

        else:
            moreg, mograd = 1., 0,
        
        cache_diff = cache_diff * moreg

        metric_thres, cache_rates = list(codebook.keys()), list(codebook.values())
        if cache_diff < metric_thres[0]: new_rate = cache_rates[0]
        elif cache_diff < metric_thres[1]: new_rate = cache_rates[1]
        elif cache_diff < metric_thres[2]: new_rate = cache_rates[2]
        elif cache_diff < metric_thres[3]: new_rate = cache_rates[3]
        elif cache_diff < metric_thres[4]: new_rate = cache_rates[4]
        else: new_rate = cache_rates[-1]

        if verbose:
            print(f'{which_module} - step {str(fwd_id).zfill(3)} - cachediff {cache_diff:.3f} - moreg {moreg:.3f} - mograd {mograd:.3f}' )
        
        return new_rate


    def forward(
        self,
        x,
        y,
        t,
        ada_dict, # adacache dict
        mask=None,  # text mask
        x_mask=None,  # temporal mask
        t0=None,  # t with timestamp=0
        T=None,  # number of frames
        S=None,  # number of pixel patches
    ):
        # prepare modulate parameters
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        if x_mask is not None:
            shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = (
                self.scale_shift_table[None] + t0.reshape(B, 6, -1)
            ).chunk(6, dim=1)

        self.fwd_id += 1

        skip_cache_steps = [1, self.num_sampling_steps-1, self.num_sampling_steps]
        if self.do_cache:
            attn_cache_rate = ada_dict['attn_cache_rate']
            self.attn_next_step = self.prev_compute_step + attn_cache_rate
            self.mlp_next_step = self.prev_compute_step + attn_cache_rate

            attn_active, mlp_active = False, False
            if (self.fwd_id == self.attn_next_step) or (self.fwd_id in skip_cache_steps):
                attn_active = True
            if (self.fwd_id == self.mlp_next_step) or (self.fwd_id in skip_cache_steps):
                mlp_active = True
        else:
            attn_active, mlp_active = True, True

        if attn_active:
            # modulate (attention)
            x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
            if x_mask is not None:
                x_m_zero = t2i_modulate(self.norm1(x), shift_msa_zero, scale_msa_zero)
                x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

            # attention
            if self.temporal:
                x_m = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
                x_m = self.attn(x_m)
                x_m = rearrange(x_m, "(B S) T C -> B (T S) C", T=T, S=S)
            else:
                x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
                x_m = self.attn(x_m)
                x_m = rearrange(x_m, "(B T) S C -> B (T S) C", T=T, S=S)

            # modulate (attention)
            x_m_s = gate_msa * x_m
            if x_mask is not None:
                x_m_s_zero = gate_msa_zero * x_m
                x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

            # residual
            attn_res = self.drop_path(x_m_s)
            x = x + attn_res

            if self.do_cache: # avoid writing if caching disabled, avoid memory fill          
                # adaptive caching, deciding the next compute step
                compute_cache_schedule = (self.cache_res == 't-attn' and self.temporal) or (self.cache_res == 's-attn' and not self.temporal)

                if (type(self.st_attn_cache) is torch.Tensor) and (self.blk_id in self.cache_loc) and compute_cache_schedule:

                    cache_in = self.st_attn_cache
                    res_in = attn_res

                    verbose = True if self.blk_id==self.cache_loc[-1] else False
                    attn_cache_rate = self.compute_next_step(
                        cache_in, res_in, self.codebook, attn_cache_rate, ada_dict, 
                        self.fwd_id, 1, verbose, self.cache_res, S, self.enable_sequence_parallelism
                        )

                    ada_dict['new_attn_cache_rate'] = attn_cache_rate  

                self.attn_recomputed_steps.append(self.fwd_id)
                self.st_attn_cache = attn_res
                self.prev_compute_step = self.fwd_id

        else: 
            x = x + self.st_attn_cache


        if mlp_active:
            # cross attention
            cattn_res = self.cross_attn(x, y, mask)
            x = x + cattn_res

            # modulate (MLP)
            x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
            if x_mask is not None:
                x_m_zero = t2i_modulate(self.norm2(x), shift_mlp_zero, scale_mlp_zero)
                x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

            # MLP
            x_m = self.mlp(x_m)

            # modulate (MLP)
            x_m_s = gate_mlp * x_m
            if x_mask is not None:
                x_m_s_zero = gate_mlp_zero * x_m
                x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

            # residual
            mlp_res = self.drop_path(x_m_s)
            x = x + mlp_res

            cattn_mlp_res = cattn_res + mlp_res

            if self.do_cache: # avoid writing if caching disabled, avoid memory fill 
                # adaptive caching, deciding the next compute step
                compute_cache_schedule = (self.cache_res == 'ca-mlp')

                if (type(self.ca_mlp_cache) is torch.Tensor) and (self.blk_id in self.cache_loc) and compute_cache_schedule:

                    cache_in = self.ca_mlp_cache
                    res_in = cattn_mlp_res

                    verbose = True if self.blk_id==self.cache_loc[-1] else False
                    attn_cache_rate = self.compute_next_step(
                        cache_in, res_in, self.codebook, attn_cache_rate, ada_dict, 
                        self.fwd_id, 1, verbose, self.cache_res, S, self.enable_sequence_parallelism
                        )

                    ada_dict['new_attn_cache_rate'] = attn_cache_rate  

                self.mlp_recomputed_steps.append(self.fwd_id)
                self.ca_mlp_cache = cattn_mlp_res
                #self.prev_compute_step = self.fwd_id

        else:
            x = x + self.ca_mlp_cache

        
        if self.fwd_id == self.num_sampling_steps:
            self.fwd_id = 0
            del self.st_attn_cache, self.ca_mlp_cache

            if self.do_cache and self.blk_id == 0 and self.temporal:
                print(f'({len(self.attn_recomputed_steps)}) recomputed steps {self.attn_recomputed_steps}')

            self.st_attn_cache, self.ca_mlp_cache = 0, 0
            self.attn_cache_rate, self.mlp_cache_rate = 1, 1
            self.attn_next_step, self.mlp_next_step = 2, 2
            self.attn_recomputed_steps, self.mlp_recomputed_steps = [], []

            self.prev_compute_step = 0

        return x, ada_dict


class STDiT3Config(PretrainedConfig):
    model_type = "STDiT3"

    def __init__(
        self,
        input_size=(None, None, None),
        input_sq_size=512,
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path=0.0,
        caption_channels=4096,
        model_max_length=300,
        qk_norm=True,
        enable_flash_attn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        only_train_temporal=False,
        freeze_y_embedder=False,
        skip_y_embedder=False,
        num_sampling_steps=100,
        do_cache=False,
        cache_res='t-attn',
        cache_loc=[13],
        codebook={0.03: 12, 0.05: 10, 0.07: 8, 0.09: 6, 0.11: 4, 1.00: 3},
        apply_moreg=False,
        moreg_strides=[1],
        moreg_steps=(10, 90),
        moreg_hyp=(0.385, 8, 1,2),
        mograd_mul=10,
        **kwargs,
    ):
        self.input_size = input_size
        self.input_sq_size = input_sq_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.pred_sigma = pred_sigma
        self.drop_path = drop_path
        self.caption_channels = caption_channels
        self.model_max_length = model_max_length
        self.qk_norm = qk_norm
        self.enable_flash_attn = enable_flash_attn
        self.enable_layernorm_kernel = enable_layernorm_kernel
        self.enable_sequence_parallelism = enable_sequence_parallelism
        self.only_train_temporal = only_train_temporal
        self.freeze_y_embedder = freeze_y_embedder
        self.skip_y_embedder = skip_y_embedder
        # for adacache
        self.num_sampling_steps = num_sampling_steps
        self.do_cache = do_cache
        self.cache_res = cache_res
        self.cache_loc = cache_loc
        self.codebook = codebook
        self.apply_moreg=apply_moreg
        self.moreg_strides=moreg_strides
        self.moreg_steps=moreg_steps
        self.moreg_hyp=moreg_hyp
        self.mograd_mul=mograd_mul
        super().__init__(**kwargs)


class STDiT3(PreTrainedModel):
    config_class = STDiT3Config

    def __init__(self, config):
        super().__init__(config)
        self.pred_sigma = config.pred_sigma
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels * 2 if config.pred_sigma else config.in_channels

        # model size related
        self.depth = config.depth
        self.mlp_ratio = config.mlp_ratio
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads

        # computation related
        self.drop_path = config.drop_path
        self.enable_flash_attn = config.enable_flash_attn
        self.enable_layernorm_kernel = config.enable_layernorm_kernel
        self.enable_sequence_parallelism = config.enable_sequence_parallelism

        # input size related
        self.patch_size = config.patch_size
        self.input_sq_size = config.input_sq_size
        self.pos_embed = PositionEmbedding2D(config.hidden_size)
        self.rope = RotaryEmbedding(dim=self.hidden_size // self.num_heads)

        # embedding
        self.x_embedder = PatchEmbed3D(config.patch_size, config.in_channels, config.hidden_size)
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.fps_embedder = SizeEmbedder(self.hidden_size)
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size, bias=True),
        )
        self.y_embedder = CaptionEmbedder(
            in_channels=config.caption_channels,
            hidden_size=config.hidden_size,
            uncond_prob=config.class_dropout_prob,
            act_layer=approx_gelu,
            token_num=config.model_max_length,
        )

        # spatial blocks
        drop_path = [x.item() for x in torch.linspace(0, self.drop_path, config.depth)]
        self.spatial_blocks = nn.ModuleList(
            [
                STDiT3Block(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=config.qk_norm,
                    enable_flash_attn=config.enable_flash_attn,
                    enable_layernorm_kernel=config.enable_layernorm_kernel,
                    enable_sequence_parallelism=config.enable_sequence_parallelism,
                    # for adacache
                    blk_id=i,
                    num_sampling_steps=config.num_sampling_steps,
                    do_cache=config.do_cache,
                    cache_res=config.cache_res,
                    cache_loc=config.cache_loc,
                    codebook=config.codebook,
                    apply_moreg=config.apply_moreg,
                    moreg_strides=config.moreg_strides,
                    moreg_steps=config.moreg_steps,
                    moreg_hyp=config.moreg_hyp,
                    mograd_mul=config.mograd_mul,
                )
                for i in range(config.depth)
            ]
        )

        # temporal blocks
        drop_path = [x.item() for x in torch.linspace(0, self.drop_path, config.depth)]
        self.temporal_blocks = nn.ModuleList(
            [
                STDiT3Block(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=config.qk_norm,
                    enable_flash_attn=config.enable_flash_attn,
                    enable_layernorm_kernel=config.enable_layernorm_kernel,
                    enable_sequence_parallelism=config.enable_sequence_parallelism,
                    # temporal
                    temporal=True,
                    rope=self.rope.rotate_queries_or_keys,
                    # for adacache
                    blk_id=i,
                    num_sampling_steps=config.num_sampling_steps,
                    do_cache=config.do_cache,
                    cache_res=config.cache_res,
                    cache_loc=config.cache_loc,
                    codebook=config.codebook,
                    apply_moreg=config.apply_moreg,
                    moreg_strides=config.moreg_strides,
                    moreg_steps=config.moreg_steps,
                    moreg_hyp=config.moreg_hyp,
                    mograd_mul=config.mograd_mul,
                )
                for i in range(config.depth)
            ]
        )

        # final layer
        self.final_layer = T2IFinalLayer(config.hidden_size, np.prod(self.patch_size), self.out_channels)

        self.initialize_weights()
        if config.only_train_temporal:
            for param in self.parameters():
                param.requires_grad = False
            for block in self.temporal_blocks:
                for param in block.parameters():
                    param.requires_grad = True

        if config.freeze_y_embedder:
            for param in self.y_embedder.parameters():
                param.requires_grad = False

        self.fwd_id = 0
        self.attn_cache_rate = 1
        self.num_sampling_steps = config.num_sampling_steps


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize fps_embedder
        nn.init.normal_(self.fps_embedder.mlp[0].weight, std=0.02)
        nn.init.constant_(self.fps_embedder.mlp[0].bias, 0)
        nn.init.constant_(self.fps_embedder.mlp[2].weight, 0)
        nn.init.constant_(self.fps_embedder.mlp[2].bias, 0)

        # Initialize timporal blocks
        for block in self.temporal_blocks:
            nn.init.constant_(block.attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.mlp.fc2.weight, 0)

    def get_dynamic_size(self, x):
        _, _, T, H, W = x.size()
        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        T = T // self.patch_size[0]
        H = H // self.patch_size[1]
        W = W // self.patch_size[2]
        return (T, H, W)

    def encode_text(self, y, mask=None):
        y = self.y_embedder(y, self.training)  # [B, 1, N_token, C]
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, self.hidden_size)
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, self.hidden_size)
        return y, y_lens

    def forward(self, x, timestep, y, mask=None, x_mask=None, fps=None, height=None, width=None, **kwargs):
        dtype = self.x_embedder.proj.weight.dtype
        B = x.size(0)
        x = x.to(dtype)
        timestep = timestep.to(dtype)
        y = y.to(dtype)

        # === get pos embed ===
        _, _, Tx, Hx, Wx = x.size()
        T, H, W = self.get_dynamic_size(x)

        # adjust for sequence parallelism
        # we need to ensure H * W is divisible by sequence parallel size
        # for simplicity, we can adjust the height to make it divisible
        if self.enable_sequence_parallelism:
            sp_size = dist.get_world_size(get_sequence_parallel_group())
            if H % sp_size != 0:
                h_pad_size = sp_size - H % sp_size
            else:
                h_pad_size = 0

            if h_pad_size > 0:
                hx_pad_size = h_pad_size * self.patch_size[1]

                # pad x along the H dimension
                H += h_pad_size
                x = F.pad(x, (0, 0, 0, hx_pad_size))

        S = H * W
        base_size = round(S**0.5)
        resolution_sq = (height[0].item() * width[0].item()) ** 0.5
        scale = resolution_sq / self.input_sq_size
        pos_emb = self.pos_embed(x, H, W, scale=scale, base_size=base_size)

        # === get timestep embed ===
        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
        fps = self.fps_embedder(fps.unsqueeze(1), B)
        t = t + fps
        t_mlp = self.t_block(t)
        t0 = t0_mlp = None
        if x_mask is not None:
            t0_timestep = torch.zeros_like(timestep)
            t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
            t0 = t0 + fps
            t0_mlp = self.t_block(t0)

        # === get y embed ===
        if self.config.skip_y_embedder:
            y_lens = mask
            if isinstance(y_lens, torch.Tensor):
                y_lens = y_lens.long().tolist()
        else:
            y, y_lens = self.encode_text(y, mask)

        # === get x embed ===
        x = self.x_embedder(x)  # [B, N, C]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        x = x + pos_emb

        # shard over the sequence dim if sp is enabled
        if self.enable_sequence_parallelism:
            x = split_forward_gather_backward(x, get_sequence_parallel_group(), dim=2, grad_scale="down")
            S = S // dist.get_world_size(get_sequence_parallel_group())

        x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)

        # sync caching shedule across layers
        ada_dict = {
            'avg_diff': 0,
            'count':0, 
            'attn_cache_rate':self.attn_cache_rate, 
            'new_attn_cache_rate':self.attn_cache_rate,
            }

        # === blocks ===
        for spatial_block, temporal_block in zip(self.spatial_blocks, self.temporal_blocks):
            x, ada_dict = auto_grad_checkpoint(spatial_block, x, y, t_mlp, ada_dict, y_lens, x_mask, t0_mlp, T, S)
            x, ada_dict = auto_grad_checkpoint(temporal_block, x, y, t_mlp, ada_dict, y_lens, x_mask, t0_mlp, T, S)
        self.attn_cache_rate = ada_dict['new_attn_cache_rate']

        if self.enable_sequence_parallelism:
            x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
            x = gather_forward_split_backward(x, get_sequence_parallel_group(), dim=2, grad_scale="up")
            S = S * dist.get_world_size(get_sequence_parallel_group())
            x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)

        # === final layer ===
        x = self.final_layer(x, t, x_mask, t0, T, S)
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)

        # cast to float32 for better accuracy
        x = x.to(torch.float32)

        self.fwd_id += 1
        if self.fwd_id == self.num_sampling_steps:
            self.fwd_id = 0
            self.attn_cache_rate = 1

        return x

    def unpatchify(self, x, N_t, N_h, N_w, R_t, R_h, R_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        # N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        # unpad
        x = x[:, :, :R_t, :R_h, :R_w]
        return x


@MODELS.register_module("STDiT3-XL/2")
def STDiT3_XL_2(from_pretrained=None, **kwargs):
    force_huggingface = kwargs.pop("force_huggingface", False)
    if force_huggingface or from_pretrained is not None and not os.path.exists(from_pretrained):
        model = STDiT3.from_pretrained(from_pretrained, **kwargs)
    else:
        config = STDiT3Config(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)
        model = STDiT3(config)
        if from_pretrained is not None:
            load_checkpoint(model, from_pretrained)
    return model


@MODELS.register_module("STDiT3-3B/2")
def STDiT3_3B_2(from_pretrained=None, **kwargs):
    force_huggingface = kwargs.pop("force_huggingface", False)
    if force_huggingface or from_pretrained is not None and not os.path.exists(from_pretrained):
        model = STDiT3.from_pretrained(from_pretrained, **kwargs)
    else:
        config = STDiT3Config(depth=28, hidden_size=1872, patch_size=(1, 2, 2), num_heads=26, **kwargs)
        model = STDiT3(config)
        if from_pretrained is not None:
            load_checkpoint(model, from_pretrained)
    return model
