#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

# 确保这些路径在你的服务器上是真实存在的
MODEL_PATHS='[
  "/mnt/gaoge/Wan2.1/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors",
  "/mnt/gaoge/Wan2.1/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
  "/mnt/gaoge/DiffSynth-Studio/Wan2.1_VAE.pth"
]'

# --num_processes 8: 启用 8 张显卡
# --trainable_models "vace,dit": 激活底座训练
# --dataset_num_workers 16: 增加 CPU 数据加载进程
/mnt/gaoge/DiffSynth-Studio/.venv/bin/python3 -m accelerate.commands.accelerate_cli launch --num_processes 8 --mixed_precision bf16 examples/wanvideo/model_training/train_droid.py \
  --droid_metadata_path "droid_metadata_with_annotations_success.pkl" \
  --dataset_base_path "./" \
  --data_file_keys "video,vace_video,vace_reference_image" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_paths "$MODEL_PATHS" \
  --learning_rate 5e-5 \
  --num_epochs 10 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-VACE-1.3B_Droid_8GPU_Full" \
  --trainable_models "dit" \
  --extra_inputs "vace_video,vace_reference_image" \
  --use_gradient_checkpointing_offload \
  --dataset_num_workers 16 \
  --sample_strategy "uniform"