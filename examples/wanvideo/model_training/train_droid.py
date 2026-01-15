import torch, os, json  
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))
# os.path.dirname(__file__) 是当前脚本所在目录 (model_training)
# os.path.join(..., "../") 就是上一级目录 (examples/wanvideo)

# 添加项目根目录 DiffSynth-Studio 到路径 (为了找 diffsynth 包)
# 从 model_training 往上跳 3 级: ../../../
sys.path.append(os.path.join(current_dir, "../../../"))

from diffsynth import load_state_dict  # 从 diffsynth 包中导入权重加载工具
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig  # 导入 Wan 视频管线及模型配置类
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, launch_training_task, wan_parser  # 导入训练模块基类、日志器、训练启动函数和 WAN 专用参数解析器
from diffsynth.trainers.unified_dataset import UnifiedDataset, LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath  # 导入统一数据集和相关数据处理算子

try:
    # 导入 DroidvideoDataset
    from droid_load_dataset import DroidvideoDataset, LoadVideoFromOSS, ImageCropAndResize 
except ImportError:
    raise ImportError("Error: Could not import DroidvideoDataset. Please make sure!") 
    exit(1)

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 关闭 tokenizer 的并行化以避免多进程/多线程警告


class WanTrainingModule(DiffusionTrainingModule):  # 定义继承自 DiffusionTrainingModule 的训练模块
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None, audio_processor_config=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()  # 调用父类的初始化方法
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)  # 解析模型路径和配置，得到统一的 model_configs
        if audio_processor_config is not None:  # 如果提供了音频处理器的配置
            # audio_processor_config 的格式为 "model_id:origin_file_pattern"
            audio_processor_config = ModelConfig(model_id=audio_processor_config.split(":")[0], origin_file_pattern=audio_processor_config.split(":")[1])  # 构建 ModelConfig 对象
        # 使用 WanVideoPipeline.from_pretrained 加载推理管线，指定 bfloat16 精度，先在 CPU 上初始化
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, audio_processor_config=audio_processor_config)
        
        # Training mode
        # 切换管线到训练模式：设置可训练子模块、LoRA 相关配置等
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,  # 当前不启用 fp8 训练
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing  # 是否使用梯度检查点节省显存
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload  # 是否将梯度检查点 offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []  # 额外输入字段列表，逗号分隔
        self.max_timestep_boundary = max_timestep_boundary  # 最大时间步边界（控制时间采样范围）
        self.min_timestep_boundary = min_timestep_boundary  # 最小时间步边界
        
        
    def forward_preprocess(self, data):  # 在前向训练前，对原始数据做预处理，构造管线输入
        # CFG-sensitive parameters（对 classifier-free guidance 有区分的部分）
        inputs_posi = {"prompt": data["prompt"]}  # 正向条件（正提示词）
        inputs_nega = {}  # 负向条件（负提示词），这里暂为空
        
        # CFG-unsensitive parameters（对 CFG 不敏感的共享参数）（CFG: Classifier-Free Guidance）
        inputs_shared = {
            "input_video": data["video"], 
            "height": data["video"][0].size[1], 
            "width": data["video"][0].size[0], 
            "num_frames": len(data["video"]), 
            "cfg_scale": 1, 
            "tiled": False, 
            "rand_device": self.pipe.device, 
            "use_gradient_checkpointing": self.use_gradient_checkpointing, 
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload, 
            "cfg_merge": False, 
            "vace_scale": 1, 
            "max_timestep_boundary": self.max_timestep_boundary, 
            "min_timestep_boundary": self.min_timestep_boundary, 
        }
        
        # Extra inputs
        # 处理额外输入，例如输入帧、结束帧、参考图像等
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]  # 使用视频的第一帧作为 input_image
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]  # 使用视频的最后一帧作为 end_image
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                # inputs_shared[extra_input] = data[extra_input][0]  
                # DroidvideoDataset 返回的 vace_reference_image 已经是 Image 对象，无需 [0]
                inputs_shared[extra_input] = data[extra_input]
            else:
                inputs_shared[extra_input] = data[extra_input] 
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi} 
    
    
    def forward(self, data, inputs=None): 
        if inputs is None: inputs = self.forward_preprocess(data) 
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__": 
    parser = wan_parser() 

    parser.add_argument("--droid_metadata_path",type=str,default="droid_metadata_with_annotations_success.pkl", help="Path to droid metadata pkl file")

    parser.add_argument("--sample_strategy", type=str, default="random", help="start, random, or uniform")
    
    args = parser.parse_args()  
    print(f" Initializing Droid Dataset from: {args.droid_metadata_path}")
    print(f" Config: {args.width}x{args.height}, Frames: {args.num_frames}, Strategy: {args.sample_strategy}")

    # 对齐 train_wan.py 的 Dataset 初始化方式
    dataset = DroidvideoDataset(
        repeat=args.dataset_repeat,
        video_operator = LoadVideoFromOSS(
            args.num_frames, 
            4, 
            1, 
            frame_processor=ImageCropAndResize(args.height, args.width, None, 16, 16), 
            sample_strategy=args.sample_strategy
        ),
        video_list = ["left_mp4_path", "right_mp4_path"]
    )    
    print(f" Dataset loaded! Total samples: {len(dataset)}")

    model = WanTrainingModule( 
        model_paths=args.model_paths, 
        model_id_with_origin_paths=args.model_id_with_origin_paths, 
        audio_processor_config=args.audio_processor_config, 
        trainable_models=args.trainable_models, 
        lora_base_model=args.lora_base_model, 
        lora_target_modules=args.lora_target_modules, 
        lora_rank=args.lora_rank, 
        lora_checkpoint=args.lora_checkpoint, 
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload, 
        extra_inputs=args.extra_inputs, 
        max_timestep_boundary=args.max_timestep_boundary, 
        min_timestep_boundary=args.min_timestep_boundary, 
    )
    model_logger = ModelLogger( 
        args.output_path, 
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt 
    )
    launch_training_task(dataset, model, model_logger, args=args)