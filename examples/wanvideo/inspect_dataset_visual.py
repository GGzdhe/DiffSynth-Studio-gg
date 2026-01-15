import os
import sys
import numpy as np
from PIL import Image
import imageio

# =========================================================================
# 🔐 注入鉴权信息 (防止 Unable to locate credentials 报错)
# =========================================================================
os.environ["AWS_ACCESS_KEY_ID"] = "01989CF435517971960CFB24AAE04E3B"
os.environ["AWS_SECRET_ACCESS_KEY"] = "01989CF4355179609DADC82EC3B04ADD"
os.environ["S3_ENDPOINT_URL"] = "http://aoss-internal.cn-sh-01b.sensecoreapi-oss.cn"

# 1. 路径 Hack
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

print(f"🔄 Importing DroidDataset from root: {project_root}")

try:
    from examples.wanvideo.droid_load_dataset import DroidDataset, LoadVideoFromOSS, ImageCropAndResize
except ImportError:
    try:
        from examples.wanvideo.droid_load_dataset_affordance import AffordanceDataset as DroidDataset, LoadVideoFromOSS, ImageCropAndResize
    except ImportError:
        print(" Error: Could not import dataset classes. Please check your file paths.")
        sys.exit(1)

def save_specific_samples(dataset, output_dir, target_indices):
    os.makedirs(output_dir, exist_ok=True)
    print(f" Output directory created: {output_dir}")
    
    for idx in target_indices:
        if idx >= len(dataset):
            print(f" Index {idx} out of range (Dataset size: {len(dataset)}), skipping.")
            continue

        print(f"  Processing Index: {idx}...")
        try:
            data = dataset[idx]
            
            # 兼容不同的 key
            video_frames = data.get("video", data.get("vace_video")) 
            ref_data = data.get("vace_reference_image")
            
            if isinstance(ref_data, list):
                ref_image = ref_data[0]
            else:
                ref_image = ref_data
                
            prompt = data.get("prompt", "")
            
            # 保存参考图
            if ref_image:
                ref_path = os.path.join(output_dir, f"idx_{idx}_ref.jpg")
                ref_image.save(ref_path)
            
            # 保存 GIF
            if video_frames:
                gif_path = os.path.join(output_dir, f"idx_{idx}_video.gif")
                frames_np = [np.array(f) for f in video_frames]
                imageio.mimsave(gif_path, frames_np, fps=8, loop=0)
                print(f"    ✅ Saved: {gif_path} ({len(frames_np)} frames)")
            
            # 保存 Prompt
            txt_path = os.path.join(output_dir, f"idx_{idx}_prompt.txt")
            with open(txt_path, "w") as f:
                f.write(prompt)
            
        except Exception as e:
            print(f"    ❌ Error processing index {idx}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    METADATA_PATH = "droid_metadata_with_annotations_success.pkl"
    # 你指定的 indices
    TARGET_INDICES = [10, 1000, 2000, 3000, 6000, 8000, 9000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
    
    STRATEGIES = ["interval", "uniform"]
    FRAME_COUNTS = [49, 81, 121]
    
    img_processor = ImageCropAndResize(height=480, width=832)

    print(" Starting Batch Inspection...")
    print(f" Target Indices: {TARGET_INDICES}")

    for strategy in STRATEGIES:
        for num_frames in FRAME_COUNTS:
            print(f"\n==========================================")
            print(f" Config: Strategy='{strategy}' | Frames={num_frames}")
            print(f"==========================================")
            
            video_loader = LoadVideoFromOSS(
                num_frames=num_frames,
                frame_interval=4,      # <--- 修正了这里
                sample_strategy=strategy,
                frame_processor=img_processor
            )
            
            # 2. 初始化 Dataset
            try:
                try:
                    dataset = DroidDataset(
                        metadata_path=METADATA_PATH, 
                        video_operator=video_loader
                    )
                except TypeError:
                    print("⚠️ Standard init failed, trying AffordanceDataset signature...")
                    dataset = DroidDataset(
                        video_operator=video_loader,
                        width=832,
                        height=480
                    )
                
                print(f"✅ Dataset initialized. Total samples: {len(dataset)}")
            except Exception as e:
                print(f" Failed to init dataset: {e}")
                continue

            # 3. 定义输出目录
            current_output_dir = os.path.join("visual_check", f"{strategy}_{num_frames}frames")
            
            # 4. 执行采样保存
            save_specific_samples(dataset, current_output_dir, TARGET_INDICES)

    print("\n All inspections completed! Check the 'visual_check' folder.")