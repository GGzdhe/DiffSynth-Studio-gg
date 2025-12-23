import os
import sys
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
import cv2
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

# ==================== 基础配置 ====================
VIDEO_BUCKET = "hidream-dataset-embodied-ai"
ANNOTATION_BUCKET = "hidream-user-gaoge"
OSS_ENDPOINT = os.getenv("S3_ENDPOINT_URL", "http://aoss-internal.cn-sh-01b.sensecoreapi-oss.cn")

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

def download_annotation_file(file_name, bucket_name=ANNOTATION_BUCKET):
    if os.path.exists(file_name):
        return file_name
    print(f"📥 Downloading {file_name} from {bucket_name} ...")
    s3 = boto3.client('s3', endpoint_url=OSS_ENDPOINT)
    try:
        s3.download_file(bucket_name, file_name, file_name)
    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise
    return file_name

def load_droid_metadata(metadata_path):
    if not hasattr(sys.modules['__main__'], 'Droid_DAindex'):
        sys.modules['__main__'].Droid_DAindex = Droid_DAindex
    with open(metadata_path, "rb") as f:
        return pickle.load(f)

# ==================== Operators ====================
class ImageCropAndResize:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        new_h = round(height * scale)
        new_w = round(width * scale)
        image = TF.resize(image, (new_h, new_w), interpolation=TF.InterpolationMode.BILINEAR)
        image = TF.center_crop(image, (self.height, self.width))
        return image

class LoadVideoFromOSS:
    def __init__(self, num_frames=49, sample_stride=1, sample_strategy="random", frame_processor=None):
        self.num_frames = num_frames
        self.sample_stride = sample_stride
        self.sample_strategy = sample_strategy
        self.frame_processor = frame_processor
        self.bucket_name = VIDEO_BUCKET
        self.time_division_factor = 4
        self.time_division_remainder = 1

    def get_valid_indices(self, total_frames):
        req_frames = self.num_frames
        if total_frames < req_frames:
            req_frames = total_frames
            while req_frames > 1 and req_frames % self.time_division_factor != self.time_division_remainder:
                req_frames -= 1
        
        actual_span = (req_frames - 1) * self.sample_stride + 1
        indices = []
        if total_frames < actual_span:
            indices = list(range(req_frames))
        else:
            if self.sample_strategy == "uniform":
                indices = np.linspace(0, total_frames - 1, req_frames, dtype=int).tolist()
            elif self.sample_strategy == "start":
                indices = [i * self.sample_stride for i in range(req_frames)]
            else: 
                max_start = total_frames - actual_span
                start_idx = random.randint(0, max_start)
                indices = [start_idx + i * self.sample_stride for i in range(req_frames)]
        return indices

    def __call__(self, oss_path: str):
        s3 = boto3.client('s3', endpoint_url=OSS_ENDPOINT)
        with temp_file_contextmanager(".mp4") as temp_file:
            try:
                s3.download_file(self.bucket_name, oss_path, temp_file)
            except Exception:
                if oss_path.startswith("/"):
                    s3.download_file(self.bucket_name, oss_path[1:], temp_file)
                else:
                    raise
            
            with videoReader_contextmanager(temp_file) as vr:
                indices = self.get_valid_indices(len(vr))
                video_data = vr.get_batch(indices).asnumpy()
                frames = []
                for frame_arr in video_data:
                    img = Image.fromarray(frame_arr)
                    if self.frame_processor:
                        img = self.frame_processor(img)
                    frames.append(img)
        return frames

# ==================== Affordance Dataset (Corrected) ====================

class AffordanceDataset(Dataset):
    def __init__(self, video_operator, width=832, height=480) -> None:
        self.pkl_name = "droid_annotations_merged_new_filtered.pkl"
        self.h5_name = "droid_annotations_merged_new_filtered.h5"
        download_annotation_file(self.pkl_name)
        download_annotation_file(self.h5_name)
        
        self.metadata = load_droid_metadata(self.pkl_name)
        self.metadata_keys = list(self.metadata.keys())
        self.video_operator = video_operator
        self.width = width
        self.height = height
        self.length = len(self.metadata)
        self.load_from_cache = False 
        print(f"AffordanceDataset initialized (Teacher's Logic). Samples: {self.length}")

    def __len__(self) -> int:
        return self.length

    def load_affordance_mask(self, uuid, left_or_right, total_frames_needed):
        """
        [修正版] 完全遵循老师的逻辑：
        1. 聚合所有时刻的 Mask (np.any)
        2. 仅在第1帧返回 Mask，其余帧全黑
        3. 返回 PIL Image List
        """
        # 1. 确定 Key (H5里的key通常是 left/right)
        camera_key = "left" if "left" in left_or_right else "right"
        
        try:
            with h5py.File(self.h5_name, 'r') as f:
                if uuid not in f or camera_key not in f[uuid]:
                    # 异常情况：全黑
                    merged_mask = np.zeros((720, 1280), dtype=np.uint8)
                else:
                    # 读取所有 Mask
                    raw_masks = f[uuid][camera_key]['mask'][:] # Shape: (T, H, W)
                    
                    if raw_masks.shape[0] > 0:
                        # [核心逻辑] 沿时间轴压缩，只有有 mask 的地方都算 (Logical OR)
                        merged_mask = np.any(raw_masks, axis=0).astype(np.uint8)
                    else:
                        merged_mask = np.zeros((720, 1280), dtype=np.uint8)

        except Exception as e:
            print(f"Error reading H5 for {uuid}: {e}")
            merged_mask = np.zeros((720, 1280), dtype=np.uint8)

        # 2. Resize 到训练分辨率
        # 注意：OpenCV resize 接受 (width, height)
        merged_mask = cv2.resize(
            merged_mask, 
            (self.width, self.height), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # 3. 转为 PIL Image (RGB模式，虽然是黑白的，但为了对齐 pipeline)
        # 老师代码里乘了 255，把 0/1 变成 0/255 可视化像素值
        main_mask_pil = Image.fromarray(merged_mask * 255, mode='L').convert("RGB")

        # 4. 构建返回列表
        # 第1帧是 merged_mask
        mask_list = [main_mask_pil]
        
        # 后续帧全是纯黑图
        if total_frames_needed > 1:
            empty_mask_pil = Image.new("RGB", (self.width, self.height), (0, 0, 0))
            for _ in range(total_frames_needed - 1):
                mask_list.append(empty_mask_pil)
                
        return mask_list

    def __getitem__(self, index: int) -> dict:
        for _ in range(10): 
            try:
                if _ > 0: index = random.randint(0, self.length - 1)
                
                uuid = self.metadata_keys[index]
                metadata = self.metadata[uuid]
                
                use_left = random.random() < 0.5
                video_path = metadata.left_mp4_path if use_left else metadata.right_mp4_path
                view_key = "left" if use_left else "right"
                
                # 1. 加载视频
                frames = self.video_operator(video_path)
                if not frames or len(frames) == 0:
                    raise ValueError("Empty video frames")

                # 2. 加载 Mask (传入视频帧数)
                # 注意：这里返回的是 List[PIL.Image]
                masks = self.load_affordance_mask(metadata.uuid, view_key, len(frames))

                prompt = metadata.language_instruction1
                if not prompt: prompt = "A robot performing a manipulation task."
                
                return {
                    "video": frames,               
                    "vace_video": frames,          
                    "vace_reference_image": [frames[0]], 
                    "vace_video_mask": masks,  # 现在这里是 [Mask, Black, Black...]
                    "prompt": prompt,
                }
                
            except Exception as e:
                print(f"Error loading index {index}: {e}, retrying...")
        
        raise RuntimeError("Failed to load valid data after multiple attempts.")