import sys
import os
import pickle
import pandas as pd

# 1. 设置路径 (保持你之前的成功设置)
current_dir = os.getcwd()
module_path = os.path.join(current_dir, "examples/wanvideo")
sys.path.append(module_path)

metadata_path = "droid_metadata_with_annotations_success.pkl" 
print(f"📖 正在读取元数据: {metadata_path} ...")

try:
    with open(metadata_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✅ 读取成功!")
    print(f"📊 数据类型: {type(data)}")
    
    # ---------------- 智能解析第一条数据 ----------------
    sample = None
    
    if isinstance(data, list):
        print(f"ℹ️ 这是一个列表，长度: {len(data)}")
        if len(data) > 0:
            sample = data[0]
            
    elif isinstance(data, dict):
        print(f"ℹ️ 这是一个字典，Key的数量: {len(data)}")
        # 获取第一个 Key
        first_key = next(iter(data))
        print(f"🔑 First Key: {first_key}")
        sample = data[first_key]
        
    elif isinstance(data, pd.DataFrame):
        print(f"ℹ️ 这是一个 Pandas DataFrame，行数: {len(data)}")
        if len(data) > 0:
            # 获取第一行
            sample = data.iloc[0].to_dict()
    else:
        print("⚠️ 未知数据类型，尝试直接打印:")
        print(data)

    # ---------------- 打印样本详情 ----------------
    if sample:
        print("\n🔍 --- 样本详情 (请重点看 video/path 字段) ---")
        # 如果是对象，转成字典打印
        if hasattr(sample, '__dict__'):
            print(f"【Object Type】: {type(sample)}")
            print(sample.__dict__)
        elif isinstance(sample, dict):
            print(sample)
        else:
            print(sample)

except Exception as e:
    print(f"❌ 读取失败: {repr(e)}")