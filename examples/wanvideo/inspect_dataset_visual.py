import os
import sys
import numpy as np
from PIL import Image
import imageio

# 1. 路径 Hack (确保能导入 DroidDataset)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

print(f"🔄 Importing DroidDataset from root: {project_root}")
# 确保导入了必要的算子类
try:
    from examples.wanvideo.droid_load_dataset import DroidDataset, LoadVideoFromOSS, ImageCropAndResize
except ImportError:
    try:
        # 兼容 Affordance Dataset 的情况
        from examples.wanvideo.droid_load_dataset_affordance import AffordanceDataset as DroidDataset, LoadVideoFromOSS, ImageCropAndResize
    except ImportError:
        print("❌ Error: Could not import dataset classes. Please check your file paths.")
        sys.exit(1)

def save_specific_samples(dataset, output_dir, target_indices):
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 Output directory created: {output_dir}")
    
    for idx in target_indices:
        # 1. 越界检查
        if idx >= len(dataset):
            print(f"⚠️ Index {idx} out of range (Dataset size: {len(dataset)}), skipping.")
            continue

        print(f"  Processing Index: {idx}...")
        try:
            data = dataset[idx]
            
            # 2. 获取基本信息
            # 兼容不同的 key (video 或 vace_video)
            video_frames = data.get("video", data.get("vace_video")) 
            
            # 兼容 vace_reference_image 可能是 List 或 单张图的情况
            ref_data = data.get("vace_reference_image")
            if isinstance(ref_data, list):
                ref_image = ref_data[0]
            else:
                ref_image = ref_data
                
            prompt = data.get("prompt", "")
            
            # 3. 保存 Reference Image (参考图)
            if ref_image:
                ref_path = os.path.join(output_dir, f"idx_{idx}_ref.jpg")
                ref_image.save(ref_path)
            
            # 4. 保存 Video 为 GIF
            if video_frames:
                gif_path = os.path.join(output_dir, f"idx_{idx}_video.gif")
                frames_np = [np.array(f) for f in video_frames]
                # fps=8 适合预览动作
                imageio.mimsave(gif_path, frames_np, fps=8, loop=0)
                print(f"    ✅ Saved: {gif_path} ({len(frames_np)} frames)")
            
            # 5. 保存 Prompt 文本
            txt_path = os.path.join(output_dir, f"idx_{idx}_prompt.txt")
            with open(txt_path, "w") as f:
                f.write(prompt)
            
        except Exception as e:
            print(f"    ❌ Error processing index {idx}: {e}")
            # import traceback
            # traceback.print_exc()

if __name__ == "__main__":
    # 配置参数
    METADATA_PATH = "droid_metadata_with_annotations_success.pkl"
    # 你指定的 target indices (部分过大的index可能会被跳过)
    TARGET_INDICES = [10, 1000, 2000, 3000, 6000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000, 30000, 35000, 40000, 45000, 50000]
    
    # 实验组合
    STRATEGIES = ["interval", "uniform"]
    FRAME_COUNTS = [49, 81, 121]
    
    # 通用图片处理器 (保持分辨率一致)
    img_processor = ImageCropAndResize(height=480, width=832)

    print("🚀 Starting Batch Inspection...")
    print(f"🎯 Target Indices: {TARGET_INDICES}")

    # 双层循环遍历所有组合
    for strategy in STRATEGIES:
        for num_frames in FRAME_COUNTS:
            print(f"\n==========================================")
            print(f"⚙️  Config: Strategy='{strategy}' | Frames={num_frames}")
            print(f"==========================================")
            
            # 1. 初始化 Video Loader (算子)
            # sample_stride 对 interval 策略很重要，这里设为 4 (跳帧采样)
            video_loader = LoadVideoFromOSS(
                num_frames=num_frames,
                sample_stride=4, 
                sample_strategy=strategy,
                frame_processor=img_processor
            )
            
            # 2. 初始化 Dataset (注入算子)
            try:
                # 尝试标准初始化 (DroidDataset)
                try:
                    dataset = DroidDataset(
                        metadata_path=METADATA_PATH, 
                        video_operator=video_loader
                    )
                except TypeError:
                    # 如果上面的失败，可能是 AffordanceDataset，它需要 width/height 但不需要 metadata_path
                    print("⚠️ Standard init failed, trying AffordanceDataset signature...")
                    dataset = DroidDataset(
                        video_operator=video_loader,
                        width=832,
                        height=480
                    )
                
                print(f"✅ Dataset initialized. Total samples: {len(dataset)}")
            except Exception as e:
                print(f"❌ Failed to init dataset: {e}")
                continue

            # 3. 定义输出目录
            current_output_dir = os.path.join("visual_check", f"{strategy}_{num_frames}frames")
            
            # 4. 执行采样保存
            save_specific_samples(dataset, current_output_dir, TARGET_INDICES)

    print("\n🎉 All inspections completed! Check the 'visual_check' folder.")