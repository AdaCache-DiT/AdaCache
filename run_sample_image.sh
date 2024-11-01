config=$1

prompt_path='./prompts/gallery.txt'
save_dir='./samples_image/gallery'

#prompt_path='./prompts/sora.txt'
#save_dir='./samples_image/sora'

python inference.py ${config} \
  --flash-attn True \
  --layernorm-kernel True \
  --num-frames 1 \
  --resolution '1080p' \
  --aspect-ratio '9:16' \
  --num-sampling-steps 100 \
  --prompt-path ${prompt_path} \
  --save-dir ${save_dir} \
  --logdate-dir True \
  --fps 24 \
  --save-fps 24 \
  --seed 1024 \
