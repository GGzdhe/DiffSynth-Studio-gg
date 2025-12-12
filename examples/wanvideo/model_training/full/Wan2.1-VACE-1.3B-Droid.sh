# 显式设置 Python 路径
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 本地模型路径
MODEL_PATHS='[
  "/mnt/gaoge/Wan2.1/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors",
  "/mnt/gaoge/Wan2.1/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
  "/mnt/gaoge/DiffSynth-Studio/Wan2.1_VAE.pth"
]'

accelerate launch examples/wanvideo/model_training/train_droid.py \
  --droid_metadata_path "droid_metadata_with_annotations_success.pkl" \
  --dataset_base_path "./" \
  --data_file_keys "video,vace_video,vace_reference_image" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_paths "$MODEL_PATHS" \
  --learning_rate 1e-4 \
  --num_epochs 2 \
  --save_steps 1000 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/Wan2.1-VACE-1.3B_Droid_Full" \
  --trainable_models "vace" \
  --extra_inputs "vace_video,vace_reference_image" \
  --use_gradient_checkpointing_offload