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
from tqdm import tqdm
from PIL import Image

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

from diffsynth.trainers.unified_dataset import (
    DataProcessingPipeline,
    DataProcessingOperator,
)

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
        
class DroidVedioFrameDataset(Dataset):
    """
    提取Droid视频固定帧数据，一般是第一帧，做标注
    """
    def __init__(self, metadata_path:str = "droid_metadata_with_annotations_success.pkl", frame_index:int = 0) -> None:
        self.s3 = boto3.client('s3')
        self.metadata = load_droid_metadata(metadata_path)
        self.metadata_keys = list(self.metadata.keys())
        self.frame_index = frame_index
        self.length = len(self.metadata)
        
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, index:int) -> dict[str, any]:
        metadata = self.metadata[self.metadata_keys[index]]
        try:
            with temp_file_contextmanager(".mp4") as temp_file:
                self.s3.download_file(OSS_BUCKET, metadata.left_mp4_path, temp_file)
                with videoReader_contextmanager(temp_file) as vr:
                    left_frame = vr[self.frame_index].asnumpy()
            
            with temp_file_contextmanager(".mp4") as temp_file:
                self.s3.download_file(OSS_BUCKET, metadata.left_mp4_path, temp_file)
                with videoReader_contextmanager(temp_file) as vr:
                    right_frame = vr[self.frame_index].asnumpy()
        except Exception as e:
            print(e)
            print(index)
            print(metadata)
            exit(1)
        
        return {
            "uuid": metadata.uuid,
            "frame": self.frame_index,
            "left_frame": left_frame,
            "right_frame": right_frame,
            "instruction1": metadata.language_instruction1,
        }
        

##############################################################
#   下面的代码是用来训练Wan模型时编写的Dataloader以及各种tools   #  
##############################################################

class LoadVideoFromOSS(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x, sample_strategy="random"):
        """
        :param sample_strategy: 采样策略
            - "start": 从头截取连续片段
            - "random": 随机截取连续片段
            - "uniform": 在整个视频时长内均匀采样
        """
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_processor = frame_processor
        self.sample_strategy = sample_strategy
        self.s3 = boto3.client('s3')
    
    def get_actual_frames(self, total_frames):
        """
        计算实际应该取多少帧（处理视频比目标短的情况，并对齐VAE）
        """
        actual_frames = self.num_frames
        if total_frames < actual_frames:
            actual_frames = total_frames
            # 确保帧数满足 4n+1 (VAE 约束)
            while actual_frames > 1 and actual_frames % self.time_division_factor != self.time_division_remainder:
                actual_frames -= 1
        return actual_frames
    
    def __call__(self, data: str):
        frames = []
        
        with temp_file_contextmanager(".mp4") as temp_file:
            self.s3.download_file(OSS_BUCKET, data, temp_file)
            with videoReader_contextmanager(temp_file) as reader:
                total_frames = len(reader)
                indices = []
                
                actual_frames = self.get_actual_frames(total_frames)
                
                if total_frames <= self.num_frames: # 视频比目标帧数少
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
                    
                video_data = reader.get_batch(indices).asnumpy()
                frames = []
                for i in range(actual_frames):
                    frame = Image.fromarray(video_data[i], mode='RGB')
                    frame = self.frame_processor(frame)
                    frames.append(frame)
                
        return frames

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
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
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
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
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


class DroidVedioDataset(Dataset):
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
        self.load_from_cache = False # 兼容DiffSynth代码
    
    def get_metadata(self, index) -> Droid_DAindex:
        index = index % self.length
        return self.metadata[self.metadata_keys[index]]
        
    def __len__(self) -> int:
        return self.length * self.repeat
    
    def __getitem__(self, index) -> dict[str, any]:
        metadata = self.get_metadata(index)
        video_choose = random.choice(self.video_list)

        keys_map = {
            "video" : video_choose, # video 要在最前面前声明
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

        return data

if __name__ == "__main__":
    # metadata = load_droid_metadata()
    # print(len(metadata))
    # make_droid_metadata()
    
    dataset = DroidVedioDataset(
        video_operator = LoadVideoFromOSS(49, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16))
    )
    print(len(dataset))
    print(dataset[0])