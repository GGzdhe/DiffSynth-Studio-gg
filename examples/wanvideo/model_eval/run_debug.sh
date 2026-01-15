#!/bin/bash

TASK_NAME="debug_task"
OUTPUT_ROOT="/mnt/afs_gaoge/DiffSynth-Studio/eval_output/${TASK_NAME}"

# 模型路径
CKPT_PATH="/mnt/afs_gaoge/DiffSynth-Studio/models/train/Wan2.1-VACE-1.3B_full_interval_dit/epoch-2.safetensors"
UNIFIED_REWARD_PATH="/mnt/afs_gaoge/UnifiedReward/checkpoints"

# 参数
NUM_SAMPLES=2
SEED=24 
GPU_ID=0

# 环境
PYTHON_VACE="/mnt/afs_gaoge/DiffSynth-Studio/.venv/bin/python3"
CMD_EVAL="uv run python" # UnifiedReward 环境命令
UNIFIED_REWARD_PATH="/mnt/afs_gaoge/UnifiedReward/checkpoints"

echo "Starting Single-GPU Debug Pipeline..."
echo "Output: ${OUTPUT_ROOT}"
mkdir -p "$OUTPUT_ROOT"

# 设置 VACE 环境路径
export PYTHONPATH=$PYTHONPATH:/mnt/afs_gaoge/DiffSynth-Studio

echo -e "\n>>> Generating Videos..."
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_VACE examples/wanvideo/model_eval/generate_for_eval.py \
    --ckpt_path "$CKPT_PATH" \
    --output_base_dir "$OUTPUT_ROOT" \
    --num_samples "$NUM_SAMPLES" \
    --seed "$SEED" \
    --gpu_id 0 \
    --rank 0 \
    --world_size 1

if [ $? -ne 0 ]; then
    echo "❌ Stage 1 Failed!"
    exit 1
fi

# 评估 (UnifiedReward 环境)
echo -e "\n>>> Evaluating..."
cd /mnt/afs_gaoge/UnifiedReward

$CMD_EVAL eval_vace.py \
    --eval_input_dir "$OUTPUT_ROOT" \
    --model_path "$UNIFIED_REWARD_PATH" \
    --gpu_id 0 \
    --world_size 1

if [ $? -ne 0 ]; then
    echo "❌ Stage 2 Failed!"
    exit 1
fi
