#!/bin/bash

TASK_NAME="eval_full_test_01"
OUTPUT_ROOT="/mnt/afs_gaoge/DiffSynth-Studio/eval_output/${TASK_NAME}"

# 模型路径
CKPT_PATH="/mnt/afs_gaoge/DiffSynth-Studio/models/train/Wan2.1-VACE-1.3B_Droid_8GPU_Full_Corrected/step-200.safetensors"
UNIFIED_REWARD_PATH="/mnt/afs_gaoge/UnifiedReward/checkpoints"

# 参数
NUM_SAMPLES=100  # 总共生成多少个视频
SEED=42          # 全局种子
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

IFS=',' read -r -a GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_ARRAY[@]}
PYTHON_VACE="/mnt/afs_gaoge/DiffSynth-Studio/.venv/bin/python3"
CMD_EVAL="uv run python"

echo " Launching ${NUM_GPUS}-GPU Pipeline"
echo " Output: ${OUTPUT_ROOT}"


mkdir -p "$OUTPUT_ROOT"
export PYTHONPATH=$PYTHONPATH:/mnt/afs_gaoge/DiffSynth-Studio

# VACE 并行生成
echo ">>> Generating Videos..."

pids=()
for ((i=0; i<NUM_GPUS; i++)); do
    GPU_ID=${GPU_ARRAY[i]}
    echo "    -> Launching Gen Worker-$i on Physical GPU $GPU_ID"
    
    # 核心逻辑：
    # 1. CUDA_VISIBLE_DEVICES=$GPU_ID: 让进程只看得到这一张卡，变成逻辑上的 cuda:0
    # 2. --gpu_id 0: 告诉脚本用第一张可见卡
    # 3. --rank $i: 告诉脚本我是第几个进程（用于文件名 meta_part_1.json）
    
    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_VACE examples/wanvideo/model_eval/generate_for_eval.py \
        --ckpt_path "$CKPT_PATH" \
        --output_base_dir "$OUTPUT_ROOT" \
        --num_samples "$NUM_SAMPLES" \
        --seed "$SEED" \
        --gpu_id 0 \
        --rank "$i" \
        --world_size "$NUM_GPUS" \
        > "${OUTPUT_ROOT}/gen_log_gpu${i}.txt" 2>&1 &
        
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait $pid
done
echo "✅ Stage 1 Done."

# UnifiedReward 并行打分
echo ">>> Scoring Videos..."
cd /mnt/afs_gaoge/UnifiedReward

pids=()
for ((i=0; i<NUM_GPUS; i++)); do
    GPU_ID=${GPU_ARRAY[i]}
    echo "    -> Launching Eval Worker-$i on Physical GPU $GPU_ID"
    
    # eval_vace.py 内部会根据 world_size 和 gpu_id 切分读取 meta_part_*.json
    # 这里的 --gpu_id 传给 python 脚本用于逻辑判断，但也用于指定 device (因为我们做了物理隔离，所以内部 device 还是 0)
    
    CUDA_VISIBLE_DEVICES=$GPU_ID $CMD_EVAL eval_vace.py \
        --eval_input_dir "$OUTPUT_ROOT" \
        --model_path "$UNIFIED_REWARD_PATH" \
        --gpu_id "$i" \
        --world_size "$NUM_GPUS" \
        > "${OUTPUT_ROOT}/eval_log_gpu${i}.txt" 2>&1 &
    
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait $pid
done
echo "✅ Stage 2 Done."

