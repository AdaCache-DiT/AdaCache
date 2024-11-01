config=$1

prompt_path='./prompts/gallery.txt'
reference_path='./samples_image/gallery'
save_dir='./samples_video/gallery/'

#prompt_path='./prompts/sora.txt'
#reference_path='./samples_image/sora'
#save_dir='./sample_video/sora/'

python inference.py ${config} \
  --flash-attn True \
  --layernorm-kernel True \
  --num-frames '2s' \
  --resolution '720p' \
  --aspect-ratio '9:16' \
  --num-sampling-steps 100 \
  --prompt-path ${prompt_path} \
  --reference-path ${reference_path} \
  --save-dir ${save_dir} \
  --logdate-dir True \
  --mask-strategy "0" \
  --aes 7 \
  --flow 5 \
  --fps 24 \
  --save-fps 24 \
  --seed 1024 \
