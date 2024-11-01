resolution = "720p"
aspect_ratio = "9:16"
num_frames = 51
fps = 24
frame_interval = 1
save_fps = 24

num_sampling_steps = 100

do_cache = True
# cache_res ['t-attn' 's-attn' 'ca-mlp']
cache_res = 't-attn'
cache_loc = [13]
# 30-step, fast codebook {0.08: 6, 0.16: 5, 0.24: 4, 0.32: 3, 0.40: 2, 1.00: 1} 
# 30-step, slow codebook {0.08: 3, 0.16: 2, 0.24: 1, 0.32: 1, 0.40: 1, 1.00: 1}
codebook = {0.03: 12, 0.05: 10, 0.07: 8, 0.09: 6, 0.11: 4, 1.00: 3} # 100-step

apply_moreg = True
moreg_strides = [1]
# 30-step (4,26)
moreg_steps = (10, 90) # 100-step
moreg_hyp = (0.385, 8, 1,2)
mograd_mul = 10

seed = 1024
batch_size = 1
multi_resolution = "STDiT2"
dtype = "bf16"
condition_frame_length = 5
align = 5
aes = 6.5
flow = None

model = dict(
    type="STDiT3-XL/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v3",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
    num_sampling_steps=num_sampling_steps,
    do_cache=do_cache,
    cache_res=cache_res,
    cache_loc=cache_loc,
    codebook=codebook,
    apply_moreg=apply_moreg,
    moreg_strides=moreg_strides,
    moreg_steps=moreg_steps,
    moreg_hyp=moreg_hyp,
    mograd_mul=mograd_mul,
)
vae = dict(
    type="OpenSoraVAE_V1_2",
    from_pretrained="hpcai-tech/OpenSora-VAE-v1.2",
    micro_frame_size=17,
    micro_batch_size=4,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=300,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=num_sampling_steps,
    cfg_scale=7.0,
)
