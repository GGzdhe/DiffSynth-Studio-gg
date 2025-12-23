import os
import torch
import math
import argparse
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger
from diffsynth.tools.data_droid import Droid_DAindex, AffordenceMaskDataset, LoadVideoFromOSS, ImageCropAndResize
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--audio_processor_config", type=str, default=None, help="Model ID with origin paths to the audio processor config, e.g., Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    
    # 新增参数
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
    # 新增参数
    output_dir: str = "./checkpoints", 
    resume_from_checkpoint: str = None  # 传入 checkpoint 路径，例如 "./checkpoints/step_1000"
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        gradient_accumulation_steps = args.gradient_accumulation_steps
        find_unused_parameters = args.find_unused_parameters
        # 如果 args 里有相关配置
        if hasattr(args, "checkpoints_output_dir"): output_dir = args.checkpoints_output_dir
        if hasattr(args, "resume_from_checkpoint"): resume_from_checkpoint = args.resume_from_checkpoint

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    
    # 初始化 Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
        project_dir=output_dir # 设置项目目录
    )
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    global_step = 0
    first_epoch = 0

    if resume_from_checkpoint:
        if resume_from_checkpoint != "":
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")
            accelerator.load_state(resume_from_checkpoint)
            
            # 解析恢复的 step 数 (假设保存路径以 step_X 命名，或者你需要自己记录 step)
            # 这里简单通过路径名推断，或者通过计算
            try:
                # 假设路径结尾是数字，例如 checkpoint-1000
                step_from_path = int(os.path.basename(resume_from_checkpoint).split("-")[-1])
                global_step = step_from_path
            except ValueError:
                # 如果无法从路径获取，可能需要依赖保存的额外信息，这里暂定为0或估算
                pass
                
            # 计算应当从第几个 epoch 开始
            num_update_steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step % num_update_steps_per_epoch
            
            print(f"Resuming at Epoch {first_epoch}, Step {resume_step} (Global Step {global_step})")

    for epoch_id in range(first_epoch, num_epochs):
        model.train()
        
        # 如果是恢复的那个 epoch，需要跳过已经训练过的 batch
        if resume_from_checkpoint and epoch_id == first_epoch and resume_step > 0:
            # 计算需要跳过的原始 batch 数量
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
                
                # 更新全局步数 (仅在梯度更新发生时)
                if accelerator.sync_gradients:
                    global_step += 1
                    # 保存训练状态
                    if save_steps and global_step % save_steps == 0:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        print(f"Saved checkpoint at step {global_step}")
                        # 同时调用 logger 保存模型权重（用于推理）
                        model_logger.on_step_end(accelerator, model, save_steps)

        # 训练结束保存最后的 checkopint
        accelerator.save_state(os.path.join(output_dir,  f"checkpoint-{global_step}"))
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
            
        model_logger.on_training_end(accelerator, model, save_steps)


class WanTrainingModule(DiffusionTrainingModule):
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
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        if audio_processor_config is not None:
            audio_processor_config = ModelConfig(model_id=audio_processor_config.split(":")[0], origin_file_pattern=audio_processor_config.split(":")[1])
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, audio_processor_config=audio_processor_config)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
        
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
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
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
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
    args = parser.parse_args()
    # dataset = UnifiedDataset(
    #     base_path=args.dataset_base_path,
    #     metadata_path=args.dataset_metadata_path,
    #     repeat=args.dataset_repeat,
    #     data_file_keys=args.data_file_keys.split(","),
    #     main_data_operator=UnifiedDataset.default_video_operator(
    #         base_path=args.dataset_base_path,
    #         max_pixels=args.max_pixels,
    #         height=args.height,
    #         width=args.width,
    #         height_division_factor=16,
    #         width_division_factor=16,
    #         num_frames=args.num_frames,
    #         time_division_factor=4,
    #         time_division_remainder=1,
    #     ),
    #     special_operator_map={
    #         "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
    #         "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
    #     }
    # )
    dataset = AffordenceMaskDataset(
        repeat=args.dataset_repeat,
        video_operator = LoadVideoFromOSS(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(args.height, args.width, None, 16, 16), sample_strategy=args.sample_strategy),
        mask_reshape = (args.height, args.width),
    )
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
    print(f"Extra inputs: {args.extra_inputs}")
    launch_training_task(dataset, model, model_logger, args=args)
