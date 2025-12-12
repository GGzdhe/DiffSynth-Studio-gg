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

import boto3  # 导入 boto3，用于访问 AWS S3（或兼容 S3 的存储）
import decord  # 导入 decord，用于视频读取
import numpy as np  # 导入 numpy，常用为 np
from tqdm import tqdm  # 导入 tqdm，用于显示进度条
from PIL import Image  # 导入 PIL 图像模块（目前这个文件里没有使用）

import torch  # 导入 PyTorch 主模块
import torchvision  # 导入 torchvision
from torch.utils.data import DataLoader, Dataset  # 导入 DataLoader 和 Dataset 基类

from diffsynth.trainers.unified_dataset import (  # 从项目内部导入统一数据处理相关类
    DataProcessingPipeline,  # 数据处理流水线类（这里导入但未使用）
    DataProcessingOperator,  # 数据处理算子类（这里导入但未使用）
)

OSS_BUCKET = "hidream-dataset-embodied-ai"  # S3 / OSS 上的存储桶名称
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
    
def load_droid_metadata(metadata_name:str = "droid_metadata_with_annotations_success.pkl") -> dict[str, Droid_DAindex]:
    s3 = boto3.client('s3')
    
    # [新增 Hack] 动态修复 Pickle 找不到类的问题
    # 无论 Pickle 认为 Droid_DAindex 在哪里，我们都把它塞到 __main__ 里去
    import sys
    if not hasattr(sys.modules['__main__'], 'Droid_DAindex'):
        sys.modules['__main__'].Droid_DAindex = Droid_DAindex

    with temp_file_contextmanager() as temp_file:
        # 这里加个 try-except 处理本地/OSS 两种情况，方便调试
        try:
            # 优先尝试从 OSS 下载 (生产环境)
            s3.download_file(OSS_BUCKET, OSS_PREFIX+metadata_name, temp_file)
        except Exception:
            # 如果下载失败（比如没配 Key），且本地已有文件（刚才生成的），直接读本地
            if os.path.exists(metadata_name):
                print(f"⚠️ OSS download failed, falling back to local file: {metadata_name}")
                with open(metadata_name, "rb") as f:
                    return pickle.load(f)
            else:
                raise # 既没下载到也没本地文件，那就真报错了

        with open(temp_file, "rb") as f:
            return pickle.load(f)

def make_droid_metadata(output_path:str = "droid_metadata_with_annotations_success.pkl") -> None:
    s3 = boto3.client('s3')  # 创建 S3 客户端
    with temp_file_contextmanager(".pkl") as temp_file:  # 临时保存 index pkl 文件
        s3.download_file(OSS_BUCKET, OSS_PREFIX + "droid_meatadata_index.pkl", temp_file)  # 下载索引文件
        droid_meatadata_index = pickle.load(open(temp_file, "rb"))  # 反序列化得到索引列表或字典

    annotations : dict[str, dict]  # 声明注释数据的类型：uuid -> 注释字典
    with temp_file_contextmanager(".json") as temp_file:  # 临时保存注释 json 文件
        s3.download_file(OSS_BUCKET, OSS_PREFIX + "aggregated-annotations-030724.json", temp_file)  # 下载注释文件
        annotations = json.load(open(temp_file))  # 加载 json 内容为字典
        
    data = dict()  # 最终要保存的元数据字典：uuid -> Droid_DAindex
    metadata_file_prefix : str  # 每个 metadata 文件的路径前缀（类型声明）
    total_count = 0  # 统计成功记录的数量
    not_found = []  # 记录缺失文件的 uuid 列表
    for metadata_file_prefix in tqdm(droid_meatadata_index):  # 遍历索引中的每个 metadata 文件，并显示进度条
        with temp_file_contextmanager(".json") as temp_file:  # 临时保存单个 metadata json
            s3.download_file(OSS_BUCKET, metadata_file_prefix, temp_file)  # 从 S3 下载 metadata 文件
            with open(temp_file) as temp_file:  # 打开下载到本地的 json 文件
                metadata = json.load(temp_file)  # 加载为字典
        # print(metadata)  # 调试打印，当前注释掉
        if metadata["uuid"] not in annotations or not metadata["success"]:  # 如果没有对应注释或任务不成功
            continue  # 跳过该条记录
        
        try:
            # 判断数据是否存在：检查视频和 hdf5 文件是否在 OSS 上
            s3.head_object(Bucket=OSS_BUCKET, Key="droid/1.0.1/" + metadata["lab"] +"/"+ metadata["wrist_mp4_path"])  # 检查手腕视频
            s3.head_object(Bucket=OSS_BUCKET, Key="droid/1.0.1/" + metadata["lab"] +"/"+ metadata["hdf5_path"])  # 检查 hdf5 文件
            
            # 记录metadata：将该条记录整合成 Droid_DAindex 对象
            data[metadata["uuid"]] = Droid_DAindex(
                uuid                    = metadata["uuid"],  # uuid
                hdf5_path               = "droid/1.0.1/" + metadata["lab"] +"/"+ metadata["hdf5_path"],  # hdf5 路径
                success                 = metadata["success"],  # 成功标志
                trajectory_length       = metadata["trajectory_length"],  # 轨迹长度
                wrist_mp4_path          = "droid/1.0.1/" + metadata["lab"] +"/"+ metadata["wrist_mp4_path"],  # 手腕视频路径
                left_mp4_path           = "droid/1.0.1/" + metadata["lab"] +"/"+ metadata["left_mp4_path"],  # 左视角视频路径
                right_mp4_path          = "droid/1.0.1/" + metadata["lab"] +"/"+ metadata["right_mp4_path"],  # 右视角视频路径
                language_instruction1   = annotations.get(metadata["uuid"]).get("language_instruction1") or "",  # 文本指令 1，可能为空
                language_instruction2   = annotations.get(metadata["uuid"]).get("language_instruction2") or "",  # 文本指令 2
                language_instruction3   = annotations.get(metadata["uuid"]).get("language_instruction3") or "",  # 文本指令 3
            )
            total_count += 1  # 计数+1
            
        except Exception as e:  # 如果检查或处理过程中发生异常
            print(f"Error processing {metadata['uuid']}: {e}")  # 打印错误信息和对应 uuid
            not_found.append(metadata["uuid"])  # 记录该 uuid
            
    print(f"total count: {total_count}")  # 打印成功记录总数
    print(f"not found: {not_found}")  # 打印未找到的 uuid 列表
    if len(not_found) > 0:  # 如果存在未找到的文件
        with open(output_path + "_not_found.json", "w") as f:  # 将未找到的列表写入 json 文件
            json.dump(not_found, f)
    with open(output_path, "wb") as f:  # 打开输出 pkl 文件
        pickle.dump(data, f)  # 将数据字典序列化保存
        
class DroidVedioFrameDataset(Dataset):  # 定义 PyTorch 数据集类
    """
    提取Droid视频固定帧数据，一般是第一帧，做标注
    """
    def __init__(self, metadata_path:str = "droid_metadata_with_annotations_success.pkl", frame_index:int = 0) -> None:
        self.s3 = boto3.client('s3')  # 为该数据集实例创建一个 S3 客户端
        self.metadata = load_droid_metadata(metadata_path)  # 加载本地或 OSS 上生成的元数据 pkl
        self.metadata_keys = list(self.metadata.keys())  # 保存所有 uuid 的列表，方便按索引访问
        self.frame_index = frame_index  # 要抽取的视频帧索引（默认第 0 帧）
        self.length = len(self.metadata)  # 数据集大小，即轨迹总数
        
        
    def __len__(self) -> int:
        return self.length  # 返回数据集大小
    
    def __getitem__(self, index:int) -> dict[str, any]:
        metadata = self.metadata[self.metadata_keys[index]]  # 根据给定索引获取对应 uuid 的 metadata
        try:
            with temp_file_contextmanager(".mp4") as temp_file:  # 使用临时 mp4 文件保存左视角视频
                self.s3.download_file(OSS_BUCKET, metadata.left_mp4_path, temp_file)  # 从 OSS 下载左视角视频
                with videoReader_contextmanager(temp_file) as vr:  # 使用自定义上下文读取视频
                    left_frame = vr[self.frame_index].asnumpy()  # 读取指定帧并转为 numpy 数组
            
            with temp_file_contextmanager(".mp4") as temp_file:  # 再创建一个临时 mp4 文件保存右视角视频
                self.s3.download_file(OSS_BUCKET, metadata.left_mp4_path, temp_file)  # 这里下载的仍是 left_mp4_path（代码原样如此）
                with videoReader_contextmanager(temp_file) as vr:  # 再次使用视频读取上下文
                    right_frame = vr[self.frame_index].asnumpy()  # 读取同一帧的图像数据
        except Exception as e:  # 如果下载或读取视频出错
            print(e)  # 打印错误信息
            print(index)  # 打印出错的索引
            print(metadata)  # 打印对应的 metadata 信息
            exit(1)  # 直接退出程序
        
        return {
            "uuid": metadata.uuid,  # 当前样本对应的 uuid
            "frame": self.frame_index,  # 当前提取的帧编号
            "left_frame": left_frame,  # 左视角图像（numpy 数组）
            "right_frame": right_frame,  # 右视角图像（numpy 数组）
            "instruction1": metadata.language_instruction1,  # 对应的自然语言指令 1
        }
    


class DroidDataset(Dataset):
    def __init__(
        self, 
        metadata_path: str = "droid_metadata_with_annotations_success.pkl",
        width: int = 832,
        height: int = 480,
        num_frames: int = 49,
        sample_stride: int = 1,
        time_division_factor: int = 4,   # 对应 utils.py 中的 VAE 压缩倍率
        time_division_remainder: int = 1 # 对应 utils.py 中的余数 (k*4 + 1)
    ) -> None:
        self.s3 = boto3.client('s3')
        if not os.path.exists(metadata_path):
            try:
                print(f"Metadata not found, generating...")
                make_droid_metadata(metadata_path)
            except Exception as e:
                print(f"Warning: Make metadata failed: {e}")
            
        self.metadata = load_droid_metadata(metadata_path)
        self.metadata_keys = list(self.metadata.keys())
        
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.sample_stride = sample_stride
        self.length = len(self.metadata)

        self.load_from_cache = False 

        print(f"DroidDataset initialized. Target: {width}x{height}, Frames: {num_frames}")

    def __len__(self) -> int:
        return self.length

    # 直接移植 unified_dataset.py 中 ImageCropAndResize 类的核心逻辑
    def crop_and_resize(self, image: Image.Image, target_height: int, target_width: int):
        """
        Ref: ImageCropAndResize.crop_and_resize from unified_dataset.py
        确保与项目原生预处理逻辑完全一致：
        1. 计算 scale
        2. Bilinear Resize
        3. Center Crop
        """
        width, height = image.size
        # 计算缩放比例，保证短边填满目标尺寸
        scale = max(target_width / width, target_height / height)
        
        # 使用 torchvision 的 resize，插值方式为 BILINEAR
        # round() 会四舍五入到最近的整数，而不是简单的向下取整
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image

    def get_valid_frame_indices(self, total_frames):
        req_frames = self.num_frames
        
        # 如果视频太短，降级 num_frames
        if total_frames < req_frames:
            req_frames = total_frames
            # 确保满足 (n - 1) % 4 == 0 (例如 17, 13, 9, 5...)
            while req_frames > 1 and req_frames % self.time_division_factor != self.time_division_remainder:
                req_frames -= 1
        
        # 时序采样策略：长视频随机切片 (Random Crop)，短视频取全长
        actual_span = (req_frames - 1) * self.sample_stride + 1
        if total_frames >= actual_span:
            max_start = total_frames - actual_span
            start_idx = random.randint(0, max_start)
            indices = [start_idx + i * self.sample_stride for i in range(req_frames)]
        else:
            # 极少数情况：如果连最小的合法序列都凑不齐（例如视频只有3帧），直接返回空或报错处理
            # 这里简单返回从头开始的帧，后续由 collate_fn 或 loader 处理
            indices = list(range(req_frames))
            
        return indices
    
    # 根据视频总帧数，计算出需要提取的有效帧索引列表
    def process_video_frames(self, vr):
        total_frames = len(vr)
        indices = self.get_valid_frame_indices(total_frames)
        
        # 从 decord 读取
        video_data = vr.get_batch(indices).asnumpy()
        
        processed_frames = []
        for frame_arr in video_data:
            img = Image.fromarray(frame_arr) # Numpy -> PIL
            # 使用对齐后的 crop_and_resize
            img = self.crop_and_resize(img, self.height, self.width)
            processed_frames.append(img)
            
        return processed_frames

    def __getitem__(self, index: int) -> dict:
        # 重试机制
        for _ in range(10): 
            try:
                # 随机打乱 index 避免连续失败死循环
                if _ > 0: index = random.randint(0, self.length - 1)
                    
                metadata = self.metadata[self.metadata_keys[index]]
                
                # 双视角策略：随机选择 Left 或 Right
                # Droid 数据集提供了左右视角，随机选择相当于数据增强，不增加额外开销
                use_left = random.random() < 0.5
                video_path = metadata.left_mp4_path if use_left else metadata.right_mp4_path
                
                # 修正原版可能的路径错误（如果原metadata里路径不对，需在此处理）
                # 假设 metadata 中的路径是相对路径，需要拼接前缀（如果有的话）
                # 这里假设 load_droid_metadata 里已经处理好了，或者是相对路径
                
                with temp_file_contextmanager(".mp4") as temp_file:
                    self.s3.download_file(OSS_BUCKET, video_path, temp_file)
                    with videoReader_contextmanager(temp_file) as vr:
                        frames = self.process_video_frames(vr)
                
                if len(frames) == 0:
                    raise ValueError("Empty video frames")

                # 处理 Prompt
                prompt = metadata.language_instruction1
                if not prompt: prompt = "A robot performing a manipulation task."
                
                # 构建输出字典
                # 注：vace_reference_image 取第0帧，符合 R2V 逻辑
                return {
                    "video": frames,               
                    "vace_video": frames,          
                    "vace_reference_image": [frames[0]], 
                    "prompt": prompt,
                    # [Debug Info] 可以在这里返回视角信息，方便 Debug
                    # "view": "left" if use_left else "right"
                }
                
            except Exception as e:
                print(f"Error loading index {index}: {e}, retrying...")
        
        raise RuntimeError("Failed to load valid data after multiple attempts.")