#!/bin/bash

# 2. 设置路径
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 3. 定义本地模型路径 (JSON 格式)
MODEL_PATHS='[
  "/mnt/gaoge/Wan2.1/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors",
  "/mnt/gaoge/Wan2.1/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
  "/mnt/gaoge/DiffSynth-Studio/Wan2.1_VAE.pth"
]'

# 4. 启动命令
# --num_processes 8: 如果是单卡，改为 1
# tee train_log.txt: 自动保存日志
python -m accelerate.commands.accelerate_cli launch --num_processes 1 --mixed_precision bf16 examples/wanvideo/model_training/train_droid.py \
  --droid_metadata_path "droid_metadata_with_annotations_success.pkl" \
  --dataset_base_path "./" \
  --data_file_keys "video,vace_video,vace_reference_image" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_paths "$MODEL_PATHS" \
  --learning_rate 1e-4 \
  --num_epochs 15 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/Wan2.1-VACE-1.3B_Droid_Full" \
  --trainable_models "vace,dit" \
  --extra_inputs "vace_video,vace_reference_image" \
  --use_gradient_checkpointing_offload \
  --sample_strategy "random" \
  --save_steps 15 \
  2>&1 | tee train_droid_log.txt