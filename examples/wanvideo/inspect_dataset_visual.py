import os
import sys
import random
import numpy as np
from PIL import Image
import imageio

# 1. 路径 Hack (确保能导入 DroidDataset)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

print(f"🔄 Importing DroidDataset from root: {project_root}")
from examples.wanvideo.droid_load_dataset import DroidDataset

def save_visual_sample(dataset, output_dir="visual_check", num_samples=3):
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 Output directory created: {output_dir}")
    
    indices = random.sample(range(len(dataset)), num_samples)
    
    for i, idx in enumerate(indices):
        print(f"\nProcessing Sample {i+1}/{num_samples} (Index: {idx})...")
        try:
            data = dataset[idx]
            
            # 1. 获取基本信息
            video_frames = data["video"] # List[PIL.Image]
            ref_image = data["vace_reference_image"][0] # PIL.Image
            prompt = data["prompt"]
            
            # 2. 保存 Reference Image (参考图)
            ref_path = os.path.join(output_dir, f"sample_{idx}_ref.jpg")
            ref_image.save(ref_path)
            print(f"  📸 Saved Reference Image: {ref_path}")
            
            # 3. 保存 Video 为 GIF (检查时序连贯性)
            gif_path = os.path.join(output_dir, f"sample_{idx}_video.gif")
            # 将 PIL Images 转换为 numpy 数组列表
            frames_np = [np.array(f) for f in video_frames]
            imageio.mimsave(gif_path, frames_np, fps=8, loop=0)
            print(f"  🎬 Saved Video GIF: {gif_path} ({len(frames_np)} frames)")
            
            # 4. 保存 Prompt 文本
            txt_path = os.path.join(output_dir, f"sample_{idx}_prompt.txt")
            with open(txt_path, "w") as f:
                f.write(prompt)
            print(f"  📝 Saved Prompt: {txt_path}")
            
            # 5. 打印一些元数据供检查
            print(f"  [Check] Ref Image Size: {ref_image.size}")
            print(f"  [Check] Video Frame 0 Size: {video_frames[0].size}")
            
        except Exception as e:
            print(f"❌ Error processing index {idx}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Initializing Droid Dataset...")
    # 使用和训练脚本一样的参数
    dataset = DroidDataset(
        metadata_path="droid_metadata_with_annotations_success.pkl",
        width=832,
        height=480,
        num_frames=49,
        sample_stride=1 
    )
    
    print(f"✅ Dataset loaded. Total samples: {len(dataset)}")
    print("running visual inspection...")
    save_visual_sample(dataset, num_samples=5) # 随机抽取5个样本进行检查
    print("\n🎉 Inspection Done! Please check the 'visual_check' folder.")