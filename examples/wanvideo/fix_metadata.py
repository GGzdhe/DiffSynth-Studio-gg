import pickle
import sys
import os

# 1. 导入正确的类定义
# 确保能找到 droid_load_dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current_dir))) # 添加 DiffSynth-Studio 根目录
from droid_load_dataset import Droid_DAindex as RealClass

# 2. 定义一个“假”类，骗过 pickle，让它能读出数据
# 因为 pickle 抱怨找不到 __main__.Droid_DAindex，我们就给它造一个
class Droid_DAindex:
    pass

# 3. 把这个假类挂载到 __main__ 上
# 这样 pickle.load 找 __main__.Droid_DAindex 时就能找到了
sys.modules['__main__'].Droid_DAindex = Droid_DAindex

def fix_pickle():
    file_path = "droid_metadata_with_annotations_success.pkl"
    print(f"🔧 正在修复 {file_path} ...")
    
    # === 步骤 A: 读取旧数据 ===
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        print(f"  ✅ 读取成功！共 {len(data)} 条记录。")
    except Exception as e:
        print(f"  ❌ 读取失败: {e}")
        return

    # data 的值现在是 __main__.Droid_DAindex 类型
    # 我们要强制把它们的类型改成 droid_load_dataset.Droid_DAindex
    
    count = 0
    for key, value in data.items():
        # 强制修改对象的类指向
        value.__class__ = RealClass
        count += 1
        
    print(f"  🔄 已修正 {count} 条数据的类归属。")

    # === 步骤 C: 保存新数据 ===
    # 这次保存时，value 的类已经是 RealClass (来自 droid_load_dataset)
    # 所以 pickle 会正确地记录它是 "droid_load_dataset.Droid_DAindex"
    try:
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        print("🎉 修复完成！文件已覆盖保存。")
        print("➡️ 现在你可以重新运行 debug_dataset.py 了！")
    except Exception as e:
        print(f"  ❌ 保存失败: {e}")

if __name__ == "__main__":
    fix_pickle()