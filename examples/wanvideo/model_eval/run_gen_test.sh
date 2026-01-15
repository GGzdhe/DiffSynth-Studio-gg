#!/bin/bash

CKPT_PATH="/mnt/afs_gaoge/DiffSynth-Studio/models/train/Wan2.1-VACE-1.3B_full_interval_dit/epoch-2.safetensors"

# 2. 输出目录
OUTPUT_DIR="/mnt/afs_gaoge/DiffSynth-Studio/eval_output/test/_001"

export CUDA_VISIBLE_DEVICES=0

echo "🚀 Starting Generation Test..."
echo "Using Checkpoint: $CKPT_PATH"

/mnt/gaoge/DiffSynth-Studio/.venv/bin/python3 /mnt/afs_gaoge/DiffSynth-Studio/examples/wanvideo/model_eval/generate_for_eval.py \
    --ckpt_path "$CKPT_PATH" \
    --output_base_dir "$OUTPUT_DIR" \
    --num_samples 2 \
    --gpu_id 0 \
    --world_size 1

echo "✅ Test Finished. Check output in $OUTPUT_DIR"