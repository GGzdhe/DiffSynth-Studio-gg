import torch, os, json  
import sys
import math
import argparse
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs

current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))
sys.path.append(os.path.join(current_dir, "../../../"))

from diffsynth import load_state_dict  # 从 diffsynth 包中导入权重加载工具
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig  # 导入 Wan 视频管线及模型配置类
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger # 导入训练模块基类、日志器、训练启动函数和 WAN 专用参数解析器

try:
    from droid_load_dataset_affordance import AffordanceDataset, LoadVideoFromOSS, ImageCropAndResize  # 尝试导入 DroidDataset 类
except ImportError:
    raise ImportError("Error: Could not import AffordanceDataset. Please make sure!") 

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 关闭 tokenizer 的并行化以避免多进程/多线程警告

def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame.")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Dataset repeat.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths.")
    parser.add_argument("--audio_processor_config", type=str, default=None, help="Audio processor config.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="LoRA target modules.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Offload gradient checkpointing.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary.")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Find unused parameters.")
    parser.add_argument("--save_steps", type=int, default=None, help="Checkpoint saving intervals.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    
    # [新增参数]支持 resume 和 sample_strategy
    parser.add_argument("--sample_strategy", type=str, default="random", help="Data Sample strategy.")
    parser.add_argument("--checkpoints_output_dir", type=str, required=True, help="checkpoints_output_dir")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="resume_from_checkpoint")
    return parser

def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    args = None,
    # 新增参数默认值
    output_dir: str = "./checkpoints", 
    resume_from_checkpoint: str = None 
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        gradient_accumulation_steps = args.gradient_accumulation_steps
        find_unused_parameters = args.find_unused_parameters
        if hasattr(args, "checkpoints_output_dir"): output_dir = args.checkpoints_output_dir
        if hasattr(args, "resume_from_checkpoint"): resume_from_checkpoint = args.resume_from_checkpoint

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
        project_dir=output_dir 
    )
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    global_step = 0
    first_epoch = 0

    # [断点续训逻辑]
    if resume_from_checkpoint:
        if resume_from_checkpoint != "":
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")
            accelerator.load_state(resume_from_checkpoint)
            try:
                step_from_path = int(os.path.basename(resume_from_checkpoint).split("-")[-1])
                global_step = step_from_path
            except ValueError:
                pass
            
            num_update_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step % num_update_steps_per_epoch
            print(f"Resuming at Epoch {first_epoch}, Step {resume_step} (Global Step {global_step})")

    for epoch_id in range(first_epoch, num_epochs):
        model.train()
        
        if resume_from_checkpoint and epoch_id == first_epoch and resume_step > 0:
            active_dataloader = accelerator.skip_first_batches(dataloader, resume_step * gradient_accumulation_steps)
        else:
            active_dataloader = dataloader

        progress_bar = tqdm(active_dataloader, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch_id}")

        for step, data in enumerate(progress_bar):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                
                if accelerator.sync_gradients:
                    global_step += 1
                    if save_steps and global_step % save_steps == 0:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        model_logger.on_step_end(accelerator, model, save_steps)

        # Epoch 结束保存
        accelerator.save_state(os.path.join(output_dir,  f"checkpoint-{global_step}"))
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
            
    model_logger.on_training_end(accelerator, model, save_steps)


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
        
        # CFG-unsensitive parameters（对 CFG 不敏感的共享参数）
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],  # 输入视频序列（通常是 PIL.Image 或类似对象列表）
            "height": data["video"][0].size[1],  # 从第一帧获取视频高度
            "width": data["video"][0].size[0],  # 从第一帧获取视频宽度
            "num_frames": len(data["video"]),  # 视频帧数
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,  # CFG scale，1 表示不做额外放大
            "tiled": False,  # 是否使用分块生成
            "rand_device": self.pipe.device,  # 随机数所在设备，使用管线当前设备
            "use_gradient_checkpointing": self.use_gradient_checkpointing,  # 是否使用梯度检查点
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,  # 是否 offload 梯度检查点
            "cfg_merge": False,  # 是否进行 CFG 合并（管线内部使用）
            "vace_scale": 1,  # VACE 相关缩放参数（特定算法使用）
            "max_timestep_boundary": self.max_timestep_boundary,  # 最大时间步边界
            "min_timestep_boundary": self.min_timestep_boundary,  # 最小时间步边界
        }
        
        # Extra inputs
        # 处理额外输入，例如输入帧、结束帧、参考图像等
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]  # 使用视频的第一帧作为 input_image
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]  # 使用视频的最后一帧作为 end_image
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input]  # 对于参考图像，通常从列表中取第 0 个
            else:
                inputs_shared[extra_input] = data[extra_input]  # 其他额外字段直接从 data 中取
        
        # Pipeline units will automatically process the input parameters.
        # 依次遍历管线中的每个 unit，由 unit_runner 根据 unit 类型处理输入
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}  # 将共享参数与正向条件合并成最终的模型输入（训练时通常只需要正向分支）
    
    
    def forward(self, data, inputs=None):  # 训练主前向函数
        if inputs is None: inputs = self.forward_preprocess(data)  # 如果未传入预处理结果，则先做前处理
        # 从管线中取出当前参与迭代优化的模型子模块（如 U-Net、VAE 等）
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        # 调用管线的 training_loss 接口计算损失
        loss = self.pipe.training_loss(**models, **inputs)
        return loss  # 返回标量或字典形式的损失


if __name__ == "__main__":  # 仅当本文件作为脚本直接运行时执行下面的代码
    parser = wan_parser()  # 创建 WAN 特定的命令行参数解析器
    args = parser.parse_args()

    print(f"Initializing Affordance Dataset (Strategy:{args.sample_strategy})")

    # 实例化图片处理器 (Processor)
    img_processor = ImageCropAndResize(
        height=args.height, 
        width=args.width
    )
    
    # 实例化视频加载器 (Operator)，注入处理器
    video_loader = LoadVideoFromOSS(
        num_frames=args.num_frames,
        sample_stride=1,
        sample_strategy=args.sample_strategy, # 使用命令行参数
        frame_processor=img_processor
    )
    
    # 3. 实例化数据集，注入加载器
    dataset = AffordanceDataset(
        video_operator = video_loader,
        width = args.width,
        height = args.height,
        repeat = args.dataset_repeat
    )

    print(f" Dataset loaded! Total samples: {len(dataset)}")

    model = WanTrainingModule(  # 实例化训练模块
        model_paths=args.model_paths,  # 模型权重路径（支持多个）
        model_id_with_origin_paths=args.model_id_with_origin_paths,  # model_id 与原始文件路径的映射
        audio_processor_config=args.audio_processor_config,  # 音频处理器配置（可选）
        trainable_models=args.trainable_models,  # 设置哪些子模块是可训练的
        lora_base_model=args.lora_base_model,  # LoRA 的基模型名称
        lora_target_modules=args.lora_target_modules,  # LoRA 作用的目标模块列表（逗号分隔）
        lora_rank=args.lora_rank,  # LoRA 的 rank
        lora_checkpoint=args.lora_checkpoint,  # 若有已有 LoRA checkpoint，可在此加载
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,  # 是否对梯度检查点做 offload
        extra_inputs=args.extra_inputs,  # 额外输入字段（逗号分隔）
        max_timestep_boundary=args.max_timestep_boundary,  # 最大时间步边界
        min_timestep_boundary=args.min_timestep_boundary,  # 最小时间步边界
    )
    model_logger = ModelLogger(  # 构建模型日志记录和保存工具
        args.output_path,  # 输出目录（保存日志、权重等）
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt  # 是否在保存 checkpoint 时移除参数前缀
    )

    print(f"Extra inputs configured : {args.extra_inputs}")

    launch_training_task(dataset, model, model_logger, args=args)  # 启动训练任务，内部会创建 DataLoader、优化器、训练循环等