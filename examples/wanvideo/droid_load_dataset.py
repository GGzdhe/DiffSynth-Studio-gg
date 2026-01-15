import os
os.environ["HTTPS_PROXY"] = ""
os.environ["HTTP_PROXY"] = ""
import gc
import json
import uuid
import pickle
import random
import contextlib
from dataclasses import dataclass

import boto3
import decord
import numpy as np
import h5py
from tqdm import tqdm
from PIL import Image

import cv2
import torch
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

# 尝试导入基类
try:
    from diffsynth.trainers.unified_dataset import (
        DataProcessingPipeline,
        DataProcessingOperator,
    )
except ImportError:
    # 如果找不到，定义空类作为替身，保证代码不崩
    class DataProcessingOperator: pass
    class DataProcessingPipeline: pass

# ==================== 基础配置 ====================
OSS_BUCKET = "hidream-dataset-embodied-ai"
OSS_PREFIX = "droid/1.0.1/"

@contextlib.contextmanager
def videoReader_contextmanager(*args, **kwargs):
    vr = decord.VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()

@contextlib.contextmanager
def temp_file_contextmanager(suffix = "", directory="/dev/shm"):
    filename = os.path.join(directory, f"{uuid.uuid4()}{suffix}")
    try:
        yield filename
    finally:
        if os.path.exists(filename):
            os.remove(filename)

# ==================== Metadata ====================
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
    
def load_droid_metadata(metadata_name:str = "droid_metadata_with_annotations_success.pkl") -> dict[str, Droid_DAindex]:
    s3 = boto3.client('s3')
    
    # 动态修复 Pickle 类定义问题
    import sys
    if not hasattr(sys.modules['__main__'], 'Droid_DAindex'):
        sys.modules['__main__'].Droid_DAindex = Droid_DAindex

    # 优先读本地
    if os.path.exists(metadata_name):
        with open(metadata_name, "rb") as f:
            return pickle.load(f)
    
    # 读不到则下载   
    with temp_file_contextmanager() as temp_file:
        s3.download_file(OSS_BUCKET, OSS_PREFIX+metadata_name, temp_file)
        with open(temp_file, "rb") as f:
            return pickle.load(f)

def make_droid_metadata(output_path:str = "droid_metadata_with_annotations_success.pkl") -> None:
    s3 = boto3.client('s3')
    with temp_file_contextmanager(".pkl") as temp_file:
        s3.download_file(OSS_BUCKET, OSS_PREFIX + "droid_meatadata_index.pkl", temp_file)
        droid_meatadata_index = pickle.load(open(temp_file, "rb"))

    annotations : dict[str, dict]
    with temp_file_contextmanager(".json") as temp_file:
        s3.download_file(OSS_BUCKET, OSS_PREFIX + "aggregated-annotations-030724.json", temp_file)
        annotations = json.load(open(temp_file))
        
    data = dict()
    metadata_file_prefix : str
    total_count = 0
    not_found = []
    for metadata_file_prefix in tqdm(droid_meatadata_index):
        with temp_file_contextmanager(".json") as temp_file:
            s3.download_file(OSS_BUCKET, metadata_file_prefix, temp_file)
            with open(temp_file) as temp_file:
                metadata = json.load(temp_file)
        # print(metadata)
        if metadata["uuid"] not in annotations or not metadata["success"]:
            continue
        
        try:
            # 判断数据是否存在
            s3.head_object(Bucket=OSS_BUCKET, Key="droid/1.0.1/" + metadata["lab"] +"/"+ metadata["wrist_mp4_path"])
            s3.head_object(Bucket=OSS_BUCKET, Key="droid/1.0.1/" + metadata["lab"] +"/"+ metadata["hdf5_path"])
            
            # 记录metadata
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
            print(f"Error processing {metadata['uuid']}: {e}")
            not_found.append(metadata["uuid"])
            
    print(f"total count: {total_count}")
    print(f"not found: {not_found}")
    if len(not_found) > 0:
        with open(output_path + "_not_found.json", "w") as f:
            json.dump(not_found, f)
    with open(output_path, "wb") as f:
        pickle.dump(data, f)


# 视频加载算子 (LoadVideoFromOSS)

class LoadVideoFromOSS(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, 
                 frame_processor=lambda x: x, sample_strategy="random", frame_interval=3):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_processor = frame_processor
        self.sample_strategy = sample_strategy
        self.frame_interval = frame_interval
        self.s3 = boto3.client('s3')
    
    def get_actual_frames(self, total_frames):
        actual_frames = self.num_frames
        if total_frames < actual_frames:
            actual_frames = total_frames
            while actual_frames > 1 and actual_frames % self.time_division_factor != self.time_division_remainder:
                actual_frames -= 1
        return actual_frames
    
    def __call__(self, data: str):
        frames = []
        
        with temp_file_contextmanager(".mp4") as temp_file:
            try:
                self.s3.download_file(OSS_BUCKET, data, temp_file)
            except Exception:
                if data.startswith("/"):
                    self.s3.download_file(OSS_BUCKET, data[1:], temp_file)
                else:
                    raise

            with videoReader_contextmanager(temp_file) as reader:
                total_frames = len(reader)
                indices = []
                
                actual_frames = self.get_actual_frames(total_frames)
                
                if total_frames <= self.num_frames: 
                    indices = list(range(actual_frames))
                else:
                    if self.sample_strategy == "start":
                        indices = list(range(actual_frames))
                    elif self.sample_strategy == "random":
                        max_start = total_frames - actual_frames
                        start_index = random.randint(0, max_start)
                        indices = list(range(start_index, start_index + actual_frames))
                    elif self.sample_strategy == "uniform":
                        indices = np.linspace(0, total_frames - 1, actual_frames, dtype=int).tolist()
                    elif self.sample_strategy == "interval":
                        required_span = (actual_frames - 1) * self.frame_interval + 1
                        if total_frames >= required_span:
                            max_start = total_frames - required_span
                            start_index = random.randint(0, max_start)
                            indices = [start_index + i * self.frame_interval for i in range(actual_frames)]
                        else:
                            indices = np.linspace(0, total_frames - 1, actual_frames, dtype=int).tolist()
                    
                video_data = reader.get_batch(indices).asnumpy()
                frames = []
                for i in range(actual_frames):
                    frame = Image.fromarray(video_data[i], mode='RGB')
                    frame = self.frame_processor(frame)
                    frames.append(frame)
                
        return frames

#  图像处理算子 (ImageCropAndResize)
class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height, width, max_pixels, height_division_factor, width_division_factor):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = TF.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=TF.InterpolationMode.BILINEAR
        )
        image = TF.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image

# =========================================================================
# Dataset (DroidvideoDataset)
# =========================================================================
class DroidvideoDataset(Dataset):
    """
    提取Droid视频数据
    """
    def __init__(self, 
                 metadata_path:str = "droid_metadata_with_annotations_success.pkl",
                 repeat = 1,
                 video_operator = lambda x:x,
                 video_list = ["left_mp4_path", "right_mp4_path"]) -> None:
        self.s3 = boto3.client('s3')
        self.metadata = load_droid_metadata(metadata_path)
        self.metadata_keys = list(self.metadata.keys())
        self.length = len(self.metadata)
        self.repeat = repeat
        self.video_operator = video_operator
        self.video_list = video_list
        self.load_from_cache = False 
    
    def get_metadata(self, index) -> Droid_DAindex:
        index = index % self.length
        return self.metadata[self.metadata_keys[index]]
        
    def __len__(self) -> int:
        return self.length * self.repeat
    
    def __getitem__(self, index) -> dict[str, any]:
        metadata = self.get_metadata(index)
        video_choose = random.choice(self.video_list)

        keys_map = {
            "video" : video_choose, 
            "prompt" : "language_instruction1",
            "vace_reference_image" : "vace_reference_video"
        }
        
        data :dict[str, any] = dict()
        for key in keys_map:
            if hasattr(metadata, keys_map[key]):
                if key == "video":
                    data[key] = self.video_operator(getattr(metadata, keys_map[key]))
                elif key == "prompt":
                    data[key] = getattr(metadata, keys_map[key])
                else:
                    raise ValueError(f"key {key} not found")
            else:
                if key == "vace_reference_image":
                    data[key] = data["video"][0] if data["video"] is not None else None
                else:
                    raise ValueError(f"key {key} not found")
        
        # 手动添加 vace_video 别名，解决 KeyError: 'vace_video'
        if "video" in data:
            data["vace_video"] = data["video"]

        return data

# 加入 Affordence Mask dataloader
class AffordenceMaskDataset(Dataset):
    def __init__(self, 
                 metadata_path:str = "droid_metadata_with_annotations_success.pkl",
                 affordence_mask_path:str = "data/affordence/droid_annotations_merged_new_filtered.h5",
                 affordence_index:str = "data/affordence/droid_annotations_merged_new_filtered.pkl",
                 repeat = 1,
                 video_operator = lambda x:x,
                 mask_reshape = None) -> None:
        
        self.s3 = boto3.client('s3')
        
        with open(affordence_index, "rb") as f:
            self.affordence_index = pickle.load(f)
        self.affordence_mask_path = affordence_mask_path
        self.metadata = load_droid_metadata(metadata_path)
        self.length = len(self.affordence_index)
        self.mask_reshape = mask_reshape
        
        self.repeat = repeat
        self.video_operator = video_operator
        self.load_from_cache = False 
        
    def __len__(self) -> int:
        return self.length * self.repeat
    
    def get_metadata(self, index) -> tuple[Droid_DAindex, str]:
        index = index % self.length
        uuid, left_or_right =  self.affordence_index[index].split("/")
        return self.metadata[uuid], left_or_right

    def load_affordence_mask(self, uuid, left_or_right, frame_nums) -> list:
        with h5py.File(self.affordence_mask_path, "r") as f:
            raw_masks = f[uuid][left_or_right]['masks'][:]
        
        if raw_masks.shape[0] > 0:
            merged_mask = np.any(raw_masks, axis=0).astype(np.uint8)
        else:
            merged_mask = np.zeros((720, 1280), dtype=np.uint8)

        if self.mask_reshape is not None:
            merged_mask = cv2.resize(
                merged_mask, 
                self.mask_reshape[::-1], 
                interpolation=cv2.INTER_NEAREST
            )
    
        main_mask_pil = Image.fromarray(merged_mask * 255, mode='L').convert("RGB")
        mask_list = [main_mask_pil]
        if frame_nums > 1:
            w, h = main_mask_pil.size
            empty_mask_pil = Image.new("RGB", (w, h), (0, 0, 0))
            for _ in range(frame_nums - 1):
                mask_list.append(empty_mask_pil)
        return mask_list
    
    def __getitem__(self, index) -> dict[str, any]:
        metadata, left_or_right = self.get_metadata(index)
        
        if left_or_right == "left_frame":
            video_choose = "left_mp4_path"
        else:
            video_choose = "right_mp4_path"

        keys_map = {
            "video" : video_choose, 
            "prompt" : "language_instruction1",
            "vace_reference_image" : "vace_reference_video",
            "vace_video_mask" : "affordence_mask"
        }
        
        data :dict[str, any] = dict()
        for key in keys_map:
            if hasattr(metadata, keys_map[key]):
                if key == "video":
                    data[key] = self.video_operator(getattr(metadata, keys_map[key]))
                elif key == "prompt":
                    data[key] = getattr(metadata, keys_map[key])
                else:
                    raise ValueError(f"key {key} not found")
            else:
                if key == "vace_reference_image":
                    data[key] = data["video"][0] if data["video"] is not None else None
                elif keys_map[key] == "affordence_mask":
                    data[key] = self.load_affordence_mask(metadata.uuid, left_or_right, len(data["video"]))
                else:
                    raise ValueError(f"key {key} not found")
        
        # 修复：确保也为 MaskDataset 添加 vace_video
        if "video" in data:
            data["vace_video"] = data["video"]

        return data

# 单元测试模块
if __name__ == "__main__":
    print("🚀 Running Unit Test for DroidvideoDataset...")
    dataset = DroidvideoDataset(
        repeat=1,
        video_operator=LoadVideoFromOSS(
            num_frames=49, 
            time_division_factor=4, 
            time_division_remainder=1, 
            frame_processor=ImageCropAndResize(480, 832, 1280*720, 16, 16),
            sample_strategy="random"
        ),
        video_list=["left_mp4_path", "right_mp4_path"]
    )
    print(f"✅ DroidvideoDataset Loaded. Length: {len(dataset)}")
    
    try:
        sample = dataset[0]
        print("   Sample keys:", sample.keys())
        print("   Video shape (frames):", len(sample["video"]))
        print("   vace_video shape (frames):", len(sample["vace_video"])) # 验证 key 是否存在
        print("   Prompt:", sample["prompt"])
        print("✅ Sample 0 loaded successfully.")
    except Exception as e:
        print(f"❌ Sample loading failed: {e}")
        