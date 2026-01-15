import os
import json
import uuid
import pickle
import contextlib
import boto3
from dataclasses import dataclass
from tqdm import tqdm

# ==================== 配置区域 ====================
OSS_BUCKET = "hidream-dataset-embodied-ai"
OSS_PREFIX = "droid/1.0.1/"
OUTPUT_FILENAME = "droid_metadata_with_annotations_success.pkl"

# ==================== 核心定义 ====================
@dataclass(slots=True)
class Droid_DAindex:
    uuid: str
    hdf5_path: str
    success: bool
    trajectory_length: int
    wrist_mp4_path: str
    left_mp4_path: str
    right_mp4_path: str
    language_instruction1: str
    language_instruction2: str
    language_instruction3: str

@contextlib.contextmanager
def temp_file_contextmanager(suffix="", directory="/dev/shm"):
    filename = os.path.join(directory, f"{uuid.uuid4()}{suffix}")
    try:
        yield filename
    finally:
        if os.path.exists(filename):
            os.remove(filename)

def make_droid_metadata(output_path):
    print(f"🚀 Starting metadata reconstruction...")
    print(f"📂 Output Path: {output_path}")
    
    s3 = boto3.client('s3')
    
    # 1. 下载索引列表
    print("⬇️ Downloading index list...")
    with temp_file_contextmanager(".pkl") as temp_file:
        s3.download_file(OSS_BUCKET, OSS_PREFIX + "droid_meatadata_index.pkl", temp_file)
        droid_meatadata_index = pickle.load(open(temp_file, "rb"))

    # 2. 下载注释文件
    print("⬇️ Downloading annotations...")
    annotations = {}
    with temp_file_contextmanager(".json") as temp_file:
        s3.download_file(OSS_BUCKET, OSS_PREFIX + "aggregated-annotations-030724.json", temp_file)
        annotations = json.load(open(temp_file))
        
    data = dict()
    total_count = 0
    not_found = []
    
    # 3. 遍历并构建元数据
    print(f"🔍 Scanning {len(droid_meatadata_index)} entries...")
    for metadata_file_prefix in tqdm(droid_meatadata_index):
        try:
            with temp_file_contextmanager(".json") as temp_file:
                s3.download_file(OSS_BUCKET, metadata_file_prefix, temp_file)
                with open(temp_file) as tf:
                    metadata = json.load(tf)

            # 过滤条件
            if metadata["uuid"] not in annotations or not metadata["success"]:
                continue
            
            # 验证 OSS 文件存在性 (这一步可能会花点时间，但能保证数据有效)
            # 如果为了速度想跳过，可以注释掉下面两行 s3.head_object
            # s3.head_object(Bucket=OSS_BUCKET, Key="droid/1.0.1/" + metadata["lab"] +"/"+ metadata["wrist_mp4_path"])
            # s3.head_object(Bucket=OSS_BUCKET, Key="droid/1.0.1/" + metadata["lab"] +"/"+ metadata["hdf5_path"])
            
            # 构建条目
            data[metadata["uuid"]] = Droid_DAindex(
                uuid                    = metadata["uuid"],
                hdf5_path               = "droid/1.0.1/" + metadata["lab"] +"/"+ metadata["hdf5_path"],
                success                 = metadata["success"],
                trajectory_length       = metadata["trajectory_length"],
                wrist_mp4_path          = "droid/1.0.1/" + metadata["lab"] +"/"+ metadata["wrist_mp4_path"],
                left_mp4_path           = "droid/1.0.1/" + metadata["lab"] +"/"+ metadata["left_mp4_path"],
                right_mp4_path          = "droid/1.0.1/" + metadata["lab"] +"/"+ metadata["right_mp4_path"],
                language_instruction1   = annotations.get(metadata["uuid"]).get("language_instruction1") or "",
                language_instruction2   = annotations.get(metadata["uuid"]).get("language_instruction2") or "",
                language_instruction3   = annotations.get(metadata["uuid"]).get("language_instruction3") or "",
            )
            total_count += 1
            
        except Exception as e:
            # print(f"⚠️ Error processing {metadata_file_prefix}: {e}")
            pass
            
    print(f"✅ Total valid entries generated: {total_count}")
    
    # 保存结果
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    print(f"💾 Metadata saved to: {output_path}")

if __name__ == "__main__":
    make_droid_metadata(OUTPUT_FILENAME)