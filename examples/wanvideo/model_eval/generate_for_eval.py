import os
import sys
import json
import torch
import argparse
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))

# 添加上一级目录 (examples/wanvideo)，以便导入 droid_load_dataset
sys.path.append(os.path.abspath(os.path.join(current_dir, "../")))
# 添加根目录 (DiffSynth-Studio)，以便导入 diffsynth
sys.path.append(os.path.abspath(os.path.join(current_dir, "../../../")))

from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.models import load_state_dict

try:
    from droid_load_dataset import DroidvideoDataset, LoadVideoFromOSS, ImageCropAndResize
except ImportError:
    print("❌ Error: Could not import DroidvideoDataset.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="VACE 批量生成评估视频")
    parser.add_argument("--ckpt_path", type=str, required=True, help="训练好的 Checkpoint 路径")
    parser.add_argument("--output_base_dir", type=str, default="/mnt/gaoge/DiffSynth-Studio/eval_output", help="输出根目录")
    parser.add_argument("--num_samples", type=int, default=50, help="总共需要生成的样本数量")
    parser.add_argument("--gpu_id", type=int, default=0, help="当前进程使用的 GPU ID")
    parser.add_argument("--world_size", type=int, default=1, help="总 GPU 数量 (用于数据分片)")
    parser.add_argument("--metadata_path", type=str, default="droid_metadata_with_annotations_success.pkl")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--rank", type=int, default=0, help="进程编号(用于文件命名,防止多卡覆盖)")
    return parser.parse_args()

def load_mixed_checkpoint(pipe, ckpt_path):
    print(f"Loading checkpoint from: {ckpt_path}")
    state_dict = load_state_dict(ckpt_path)
    vace_dict = {}
    dit_dict = {}
    for key, value in state_dict.items():
        clean_key = key.replace("pipe.", "")
        if clean_key.startswith("dit."):
            dit_dict[clean_key[4:]] = value
        elif clean_key.startswith("vace."):
            vace_dict[clean_key[5:]] = value
        else:
            dit_dict[clean_key] = value
            vace_dict[clean_key] = value
            
    if len(dit_dict) > 0: 
        msg = pipe.dit.load_state_dict(dit_dict, strict=False) 
        # print(f"✅ DiT weights loaded: {msg}")
    if len(vace_dict) > 0: 
        msg = pipe.vace.load_state_dict(vace_dict, strict=False)
        # print(f"✅ VACE weights loaded: {msg}")
    return pipe

def main():
    args = parse_args()
    device = f"cuda:{args.gpu_id}"
    
    # 初始化模型
    print(f"[GPU {args.gpu_id}] Loading Model...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="/mnt/gaoge/Wan2.1/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="/mnt/gaoge/Wan2.1/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="/mnt/gaoge/DiffSynth-Studio/Wan2.1_VAE.pth"),
        ],
    )
    if os.path.exists(args.ckpt_path):
        pipe = load_mixed_checkpoint(pipe, args.ckpt_path)
    else:
        print(f"❌ Error: Checkpoint not found at {args.ckpt_path}.")
        return

    # 初始化数据集
    # 使用 start 策略固定视角，便于评估
    dataset = DroidvideoDataset(
        repeat=1,
        video_operator=LoadVideoFromOSS(
            num_frames=49, 
            time_division_factor=4, 
            time_division_remainder=1, 
            frame_processor=ImageCropAndResize(480, 832, None, 16, 16),
            sample_strategy="start" 
        ),
        video_list=["left_mp4_path"]
    )

    # 确定当前 GPU 需要处理的 indices
    # 每隔 100 个取一个样本，保证样本多样性，然后截取前 num_samples 个
    total_candidates = list(range(0, len(dataset), 100))
    target_indices = total_candidates[:args.num_samples]
    
    # 数据分片 (Sharding)
    my_indices = [idx for i, idx in enumerate(target_indices) if i % args.world_size == args.rank]
    
    print(f"[GPU {args.rank}] Processing {len(my_indices)} samples...")

    # 生成循环
    results_metadata = []
    os.makedirs(args.output_base_dir, exist_ok=True)
    video_dir = os.path.join(args.output_base_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    for idx in my_indices:
        try:
            sample = dataset[idx]
            prompt = sample["prompt"]
            ref_data = sample["vace_reference_image"]
            ref_img_list = ref_data if isinstance(ref_data, list) else [ref_data]

            safe_prompt = prompt[:20].replace(" ", "_").replace("/", "")
            filename = f"rank{args.rank}_idx{idx}_{safe_prompt}.mp4"
            save_path = os.path.join(video_dir, filename)
            
            # 推理
            video = pipe(
                prompt=prompt,
                negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                vace_reference_image= ref_img_list, # Pipe 需要 List
                seed=args.seed,
                tiled=True,
                num_frames=49,
                height=480,
                width=832,
                num_inference_steps=50,
                cfg_scale=5.0
            )
            save_video(video, save_path, fps=15, quality=5)
            
            # 记录元数据 (供 UnifiedReward 使用)
            results_metadata.append({
                "video_path": save_path,
                "prompt": prompt,
                "dataset_idx": idx
            })
            print(f"[GPU {args.gpu_id}] Saved: {filename}")
            
        except Exception as e:
            print(f"[GPU {args.gpu_id}] Error on index {idx}: {e}")

    # 保存分片 Metadata
    json_path = os.path.join(args.output_base_dir, f"meta_part_{args.rank}.json")
    with open(json_path, "w") as f:
        json.dump(results_metadata, f, indent=4)
    print(f"[GPU {args.rank}] Metadata saved to {json_path}")

if __name__ == "__main__":
    main()