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

import torchvision.transforms.functional as TF  
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
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, 
                    frame_processor=lambda x: x, sample_strategy="random", frame_interval=3):
            """
            :param sample_strategy: 采样策略
                - "start": 从头截取连续片段
                - "random": 随机截取连续片段
                - "uniform": 在整个视频时长内均匀采样
                - "interval": 按照 frame_interval 指定的间隔进行随机采样 (新增!)
            :param frame_interval: 当 sample_strategy 为 "interval" 时的采样间隔
            """
            self.num_frames = num_frames
            self.time_division_factor = time_division_factor
            self.time_division_remainder = time_division_remainder
            self.frame_processor = frame_processor
            self.sample_strategy = sample_strategy
            self.frame_interval = frame_interval # 新增参数
            self.s3 = boto3.client('s3')
            
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
            self.s3.download_file(OSS_BUCKET, oss_path, temp_file)
            
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

class DroidDataset(Dataset):
    def __init__(
        self, 
        metadata_path: str,
        video_operator, # 注入视频处理算子 (LoadVideoFromOSS)
    ) -> None:
        if not os.path.exists(metadata_path):
            try:
                print(f"Metadata not found, attempting to generate...")
                make_droid_metadata(metadata_path)
            except Exception:
                pass 
            
        self.metadata = load_droid_metadata(metadata_path)
        self.metadata_keys = list(self.metadata.keys())
        self.video_operator = video_operator
        self.length = len(self.metadata)
        
        # [DiffSynth 兼容] 必须属性
        self.load_from_cache = False 
        print(f"DroidDataset V2.0 initialized. Samples: {self.length}")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict:
        # [保留] 核心重试机制：保证训练不中断
        for _ in range(10): 
            try:
                # 随机打乱 index 避免死磕坏数据
                if _ > 0: index = random.randint(0, self.length - 1)
                    
                metadata = self.metadata[self.metadata_keys[index]]
                
                # 双视角随机
                use_left = random.random() < 0.5
                video_path = metadata.left_mp4_path if use_left else metadata.right_mp4_path
                
                # [核心差异] 调用 Operator 处理视频
                frames = self.video_operator(video_path)
                
                if not frames or len(frames) == 0:
                    raise ValueError("Empty video frames")

                prompt = metadata.language_instruction1
                if not prompt: prompt = "A robot performing a manipulation task."
                
                # [DiffSynth VACE 格式] 确保 vace_reference_image 是 List
                return {
                    "video": frames,               
                    "vace_video": frames,          
                    "vace_reference_image": [frames[0]], 
                    "prompt": prompt,
                }
                
            except Exception as e:
                print(f"Error loading index {index}: {e}, retrying...")
        
        raise RuntimeError("Failed to load valid data after multiple attempts.")