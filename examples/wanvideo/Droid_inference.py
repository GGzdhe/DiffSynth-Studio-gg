import torch
import os # 处理文件和路径
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir) 
sys.path.append(os.path.abspath(os.path.join(current_dir, "../../")))

from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.models import load_state_dict


try:
    from droid_load_dataset import DroidDataset, LoadVideoFromOSS, ImageCropAndResize
except ImportError:
    raise ImportError("Error: Could not import DroidDataset. Please make sure!")

# 设置输出目录
OUTPUT_DIR = "/mnt/gaoge/DiffSynth-Studio/output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_mixed_checkpoint(pipe, ckpt_path):
    """
    加载训练的 Checkpoint 
    """
    print(f"Loading checkpoint from: {ckpt_path}")
    # 使用 diffsynth 自带的 load_state_dict，自动处理 .bin 和 .safetensors
    state_dict = load_state_dict(ckpt_path)
    
    vace_dict = {}
    dit_dict = {}
    
    print("Parsing weight keys...")
    for key, value in state_dict.items():
        # 移除 'pipe.' 前缀 (如果存在)
        clean_key = key.replace("pipe.", "")
        
        if clean_key.startswith("dit."):
            # 移除 'dit.' 前缀，剩下就是参数名
            real_key = clean_key[4:] 
            dit_dict[real_key] = value
        elif clean_key.startswith("vace."):
            real_key = clean_key[5:]
            vace_dict[real_key] = value
        else:
            # 兜底：如果是旧格式，尝试直接匹配
            dit_dict[clean_key] = value
            vace_dict[clean_key] = value
            
    # 加载 DiT
    if len(dit_dict) > 0:
        print(f"   --> Found {len(dit_dict)} DiT weights, injecting...")
        pipe.dit.load_state_dict(dit_dict, strict=False)
        print("   ✅ DiT weights loaded.")
    
    # 加载 VACE
    if len(vace_dict) > 0:
        print(f"   --> Found {len(vace_dict)} VACE weights, injecting...")
        pipe.vace.load_state_dict(vace_dict, strict=False)
        print("   ✅ VACE weights loaded.")
    
    return pipe

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="/mnt/gaoge/Wan2.1/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="/mnt/gaoge/Wan2.1/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="/mnt/gaoge/DiffSynth-Studio/Wan2.1_VAE.pth"),
    ],
)

pipe.enable_vram_management() # 启用显存管理以节省显存

CKPT_PATH = "/mnt/gaoge/DiffSynth-Studio/models/train/Wan2.1-VACE-1.3B_Droid_8GPU_Full/epoch-1.safetensors"

if os.path.exists(CKPT_PATH):
    print(f"Loading fine-tuned weights from {CKPT_PATH}...")
    # pipe = load_mixed_checkpoint(pipe, CKPT_PATH)
else:
    print(f"Fine-tuned weights not found at {CKPT_PATH}, using base model weights.")

print("Loading DroidDataset to get reference image and control video...")
dataset = DroidDataset(
    metadata_path = "/mnt/gaoge/DiffSynth-Studio/droid_metadata_with_annotations_success.pkl",
    video_operator = LoadVideoFromOSS(
        num_frames=49, 
        sample_stride=1, 
        sample_strategy="random", 
        frame_processor=ImageCropAndResize(height=480, width=832)
    )
)

# 选择一个测试样本 (可以随便选)
sample_index = 0
data_sample = dataset[sample_index]

prompt = data_sample["prompt"]
# 返回的 vace_refimage 是一个 List [Image]，这正是 pipe 需要的
vace_ref_image = data_sample["vace_reference_image"] 

print(f"   - 样本 ID: {sample_index}")
print(f"   - Prompt: {prompt}")
print(f"   - Ref Image Size: {vace_ref_image[0].size}")

print("Generating video...")

video = pipe(
    prompt=prompt,
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    vace_reference_image=vace_ref_image,
    seed=42, 
    tiled=True,
    num_frames=49,
    height=480,
    width=832,
    num_inference_steps=50,
    cfg_scale=5.0
)
# tiled=True 表示生成的视频可以无缝平铺
output_filename = os.path.join(OUTPUT_DIR, "video_droid_result_1.mp4")
save_video(video, output_filename, fps=15, quality=5)

print(f"Video saved to: {output_filename}")