import os  # 导入操作系统相关功能模块
os.environ["HTTPS_PROXY"] = ""  # 清空 HTTPS 代理环境变量，避免走代理
os.environ["HTTP_PROXY"] = ""  # 清空 HTTP 代理环境变量，避免走代理
import gc  # 导入垃圾回收模块，用于手动回收内存
import json  # 导入 JSON 读写模块
import uuid  # 导入 UUID 生成模块，用于生成临时文件名
import pickle  # 导入 pickle，用于对象序列化和反序列化
import random  # 导入随机数模块（目前这个文件里没有使用）
import contextlib  # 导入上下文管理工具模块
from dataclasses import dataclass  # 导入 dataclass 装饰器，用于简化数据类定义

import h5py
import cv2

import boto3  # 导入 boto3，用于访问 AWS S3（或兼容 S3 的存储）
import decord  # 导入 decord，用于视频读取
import numpy as np  # 导入 numpy，常用为 np
from tqdm import tqdm  # 导入 tqdm，用于显示进度条
from PIL import Image  # 导入 PIL 图像模块（目前这个文件里没有使用）

import torchvision.transforms.functional as TF  
import torch  # 导入 PyTorch 主模块
import torchvision  # 导入 torchvision
from torch.utils.data import DataLoader, Dataset  # 导入 DataLoader 和 Dataset 基类

from diffsynth.trainers.unified_dataset import (  # 从项目内部导入统一数据处理相关类
    DataProcessingPipeline,  # 数据处理流水线类（这里导入但未使用）
    DataProcessingOperator,  # 数据处理算子类（这里导入但未使用）
)

VIDEO_BUCKET = "hidream-dataset-embodied-ai"  # S3 / OSS 上的存储桶名称
ANNOTATION_BUCKET = "hidream-user-gaoge"
OSS_PREFIX = "droid/1.0.1/"  # 数据集在桶中的前缀路径

@contextlib.contextmanager  # 使用 contextmanager 将函数封装为上下文管理器
def videoReader_contextmanager(*args, **kwargs):
    vr = decord.VideoReader(*args, **kwargs)  # 创建 decord 的 VideoReader 对象
    try:
        yield vr  # 将视频读取器对象作为上下文返回
    finally:
        del vr  # 显式删除引用
        gc.collect()  # 手动触发垃圾回收，及时释放显存/内存

@contextlib.contextmanager  # 再定义一个通用临时文件的上下文管理器
def temp_file_contextmanager(suffix = "", directory="/dev/shm"):
    filename = os.path.join(directory, f"{uuid.uuid4()}{suffix}")  # 在指定目录生成唯一文件名
    try:
        yield filename  # 将临时文件路径返回给调用方使用
    finally:
        if os.path.exists(filename):  # 退出上下文时，如果文件存在
            os.remove(filename)  # 删除临时文件，避免残留

@dataclass(slots=True)  # 定义数据类，并使用 slots 节省内存
class Droid_DAindex:
    uuid: str  # 轨迹的唯一标识
    hdf5_path: str  # 轨迹对应的 hdf5 文件在 OSS 上的路径
    success: bool  # 该轨迹是否是成功的（过滤条件）
    trajectory_length: int  # 轨迹长度（步数）
    wrist_mp4_path: str  # 手腕视角视频路径
    left_mp4_path: str  # 左相机视角视频路径
    right_mp4_path: str  # 右相机视角视频路径
    language_instruction1: str  # 文本指令 1
    language_instruction2: str  # 文本指令 2
    language_instruction3: str  # 文本指令 3
    
def load_annotation_file(file_name,bucket_name=ANNOTATION_BUCKET):
    if os.path.exists(file_name):
        print(f"Annotation file already exist:{file_name},skip downloading.")
        return file_name
    
    print(f"Downloading {file_name} from {bucket_name} ...")
    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket_name,file_name,file_name)
        print(f"Successful download :{file_name}")
    except Exception as e:
        print(f"Failed download: {e}")
        raise
    return file_name

def load_droid_metadata(metadata_name:str = "droid_metadata_with_annotations_success.pkl") -> dict[str, Droid_DAindex]:
    # 动态修复 Pickle 找不到类的问题
    # 无论 Pickle 认为 Droid_DAindex 在哪里，我们都把它塞到 __main__ 里去
    import sys
    if not hasattr(sys.modules['__main__'], 'Droid_DAindex'):
        sys.modules['__main__'].Droid_DAindex = Droid_DAindex

    
    if os.path.exists(metadata_name):
        with open(metadata_name, "rb") as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"Metadata file {metadata_name} not found locally.")

class ImageCropAndResize:
    """
    负责图像的空间变换
    逻辑: DiffSynth 原生逻辑 (torchvision + round + bilinear)
    """
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        # 1. 计算缩放比例 (保证短边填满目标)
        scale = max(self.width / width, self.height / height)
        
        # 2. Resize (使用 round 四舍五入，对齐官方逻辑)
        new_h = round(height * scale)
        new_w = round(width * scale)
        image = TF.resize(image, (new_h, new_w), interpolation=TF.InterpolationMode.BILINEAR)
        
        # 3. Center Crop
        image = TF.center_crop(image, (self.height, self.width))
        return image


class LoadVideoFromOSS:
    """
    负责从 OSS 下载视频、时序采样、解码
    """
    def __init__(self, num_frames=49, sample_stride=1, sample_strategy="random", frame_processor=None):
        self.num_frames = num_frames
        self.sample_stride = sample_stride
        self.sample_strategy = sample_strategy  # "random", "uniform", "start"
        self.frame_processor = frame_processor  # 通常传入 ImageCropAndResize 实例
        
        # VAE 约束参数
        self.time_division_factor = 4
        self.time_division_remainder = 1
        
        self.s3 = boto3.client('s3')
        self.bucket_name = VIDEO_BUCKET

    def get_valid_indices(self, total_frames):
        """计算需要读取的帧索引"""
        req_frames = self.num_frames
        
        # 1. 短视频处理：降级帧数以适应 VAE (4k+1)
        if total_frames < req_frames:
            req_frames = total_frames
            while req_frames > 1 and req_frames % self.time_division_factor != self.time_division_remainder:
                req_frames -= 1
        
        # 2. 根据策略生成索引
        actual_span = (req_frames - 1) * self.sample_stride + 1
        indices = []

        if total_frames < actual_span:
            # 极短视频：全取
            indices = list(range(req_frames))
        else:
            if self.sample_strategy == "uniform":
                # 均匀采样 (覆盖全视频)
                indices = np.linspace(0, total_frames - 1, req_frames, dtype=int).tolist()
            elif self.sample_strategy == "start":
                # 从头开始
                indices = [i * self.sample_stride for i in range(req_frames)]
            else: 
                # "random" (默认)：随机连续切片
                max_start = total_frames - actual_span
                start_idx = random.randint(0, max_start)
                indices = [start_idx + i * self.sample_stride for i in range(req_frames)]
                
        return indices

    def __call__(self, oss_path: str):
        with temp_file_contextmanager(".mp4") as temp_file:
            # 下载
            try:
                self.s3.download_file(self.bucket_name, oss_path, temp_file)
            except Exception as e:
                # 兼容路径前缀差异
                if oss_path.startswith("/"):
                    oss_path = oss_path[1:]
                    self.s3.download_file(self.bucket_name, oss_path, temp_file)
                else:
                    raise e
            
            # 读取与处理
            with videoReader_contextmanager(temp_file) as vr:
                indices = self.get_valid_indices(len(vr))
                video_data = vr.get_batch(indices).asnumpy()
                
                frames = []
                for frame_arr in video_data:
                    img = Image.fromarray(frame_arr)
                    # 调用传入的 frame_processor (CropAndResize)
                    if self.frame_processor:
                        img = self.frame_processor(img)
                    frames.append(img)
                    
        return frames
    
class AffordanceDataset(Dataset):
    def __init__(
            self,
            video_operator,
            width=832,
            height=480,
            repeat = 1,
    )->None:
        self.pkl_name = "droid_annotations_merged_new_filtered.pkl"
        load_annotation_file(self.pkl_name)

        self.h5_name = "droid_annotations_merged_new_filtered.h5"
        load_annotation_file(self.h5_name)

        self.metadata = load_droid_metadata(self.pkl_name)
        self.metadata_keys = list(self.metadata.keys())

        self.video_operator = video_operator
        self.width = width
        self.height = height
        self.length = len(self.metadata)
        self.load_from_cache = False
        self.repeat = repeat

        print(f"AffordanceDataset initialized. Samples: {self.length}, Mask H5: {self.h5_name}")

    def __len__(self) -> int:
        return self.length * self.repeat
    
    def load_affordance_mask(self,uuid,left_or_right,total_frames_needed):
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

        # Resize 到训练分辨率
        merged_mask = cv2.resize(
            merged_mask, 
            (self.width, self.height), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # 转为 PIL Image (RGB模式，对齐 pipeline)
        main_mask_pil = Image.fromarray(merged_mask * 255, mode='L').convert("RGB")

        # 构建返回列表
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
                
                data = {} # 构造基础数据
                
                # 加载视频
                frames = self.video_operator(video_path)
                if not frames or len(frames) == 0:
                    raise ValueError("Empty video frames")
                data["video"] = frames
                data["vace_video"] = frames # 兼容 Shell 参数 keys

                prompt = metadata.language_instruction1
                if not prompt: prompt = "A robot performing a manipulation task."
                data["prompt"] = prompt

                data["vace_reference_image"] = frames[0]

                # 加载 Mask (传入视频帧数)
                data["vace_video_mask"] = self.load_affordance_mask(metadata.uuid, view_key, len(frames))

                return data
                
            except Exception as e:
                print(f"Error loading index {index}: {e}, retrying...")
        
        raise RuntimeError("Failed to load valid data after multiple attempts.")
