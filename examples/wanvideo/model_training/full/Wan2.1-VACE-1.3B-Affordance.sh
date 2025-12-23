#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

MODEL_PATHS='[
  "/mnt/gaoge/Wan2.1/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors",
  "/mnt/gaoge/Wan2.1/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
  "/mnt/gaoge/DiffSynth-Studio/Wan2.1_VAE.pth"
]'

echo "🚀 启动 Affordance 8卡训练..."

# 启动命令
# 注意：extra_inputs 里加入了 vace_video_mask
/mnt/gaoge/DiffSynth-Studio/.venv/bin/python3 -m accelerate.commands.accelerate_cli launch --num_processes 8 --mixed_precision bf16 examples/wanvideo/model_training/train_affordance.py \
  --dataset_base_path "./" \
  --data_file_keys "video,vace_video,vace_reference_image,vace_video_mask" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_paths "$MODEL_PATHS" \
  --learning_rate 2e-4 \
  --num_epochs 10 \
  --remove_prefix_in_ckpt "pipe." \
  --output_path "./models/train/Wan2.1-VACE-1.3B_Affordance_Full" \
  --trainable_models "vace,dit" \
  --extra_inputs "vace_video,vace_reference_image,vace_video_mask" \
  --use_gradient_checkpointing_offload \
  --dataset_num_workers 16 \
  --sample_strategy "uniform"