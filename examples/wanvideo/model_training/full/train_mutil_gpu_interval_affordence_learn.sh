source .venv/bin/activate

accelerate launch --num_processes 8 --mixed_precision bf16 train_wan_with_affordence.py \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 1 \
  --model_paths "[\"/mnt/data/DiffSynth-Studio/Wan-AI/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors\",\"/mnt/data/DiffSynth-Studio/Wan-AI/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth\",\"/mnt/data/DiffSynth-Studio/Wan-AI/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth\"]" \
  --learning_rate 1e-4 \
  --num_epochs 10 \
  --trainable_models "vace,dit" \
  --remove_prefix_in_ckpt "pipe.vace.,pipe.dit." \
  --output_path "./train_output/train_mutil_gpu_interval_affordence" \
  --extra_inputs "vace_reference_image,vace_video_mask" \
  --dataset_num_workers 4 \
  --sample_strategy "interval" \
  --checkpoints_output_dir "./train_output/train_mutil_gpu_interval_affordence_checkpoints" \
  2>&1 | tee train_mutil_gpu_interval_affordence.txt