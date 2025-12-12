import os
import sys
import numpy as np
from PIL import Image

# ==================== 核心修复 ====================
# 1. 获取当前脚本所在目录 (examples/wanvideo)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 获取项目根目录 (DiffSynth-Studio)
#    current_dir 的上一级是 examples，再上一级是 DiffSynth-Studio
project_root = os.path.dirname(os.path.dirname(current_dir))

# 3. 将根目录加入系统路径，这样 Python 才能找到 'diffsynth' 包
sys.path.append(project_root)
# =================================================

# 4. 再次尝试导入 (去掉 try-except 以便看到真实的报错信息)
print(f"🔄 正在尝试导入 DroidDataset... (Project Root: {project_root})")
from droid_load_dataset import DroidDataset

def test_dataset():
    print("✅ 导入成功！开始初始化数据集...")
    
    # 这里的参数要和你的 Shell 脚本一致
    dataset = DroidDataset(
        metadata_path="droid_metadata_with_annotations_success.pkl",
        width=832,
        height=480,
        num_frames=49,
        sample_stride=1
    )
    
    print(f"✅ Dataset 初始化成功! 总样本数: {len(dataset)}")
    
    # 测试随机读取 1 个样本
    print("\n🔄 开始读取测试 (Testing 1 random sample)...")
    idx = np.random.randint(0, len(dataset))
    print(f"  Reading index {idx}...")
    
    try:
        data = dataset[idx]
        
        # 1. 检查 Key
        print(f"  Keys found: {list(data.keys())}")
        
        # 2. 检查视频格式
        video = data["video"]
        print(f"  Video Type: {type(video)}")
        if isinstance(video, list) and len(video) > 0:
             print(f"  Frame Type: {type(video[0])}")
             print(f"  Frame Size: {video[0].size}")
        
        # 3. 检查帧数
        print(f"  Frame Count: {len(video)}")
        
        if len(video) == 49 and video[0].size == (832, 480):
            print("\n🎉🎉🎉 数据层验证通过！格式完美符合训练要求！")
        else:
            print("\n⚠️ 警告：数据尺寸或帧数与预期不符，请检查参数。")
            
    except Exception as e:
        print(f"❌ 读取样本失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()