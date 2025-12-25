import os
import cv2
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def draw_text_on_image(image_np, text):
    """在图片左上角添加文字标签"""
    # 转为 PIL Image 以便绘制文字
    pil_img = Image.fromarray(image_np)
    draw = ImageDraw.Draw(pil_img)
    
    # 设置字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # 计算文字大小
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    
    # [修改 1] 绘制白色背景框 (255, 255, 255)
    draw.rectangle((0, 0, text_w + 10, text_h + 10), fill=(255, 255, 255))
    # [修改 2] 绘制黑色文字 (0, 0, 0)
    draw.text((5, 5), text, font=font, fill=(0, 0, 0))
    
    return np.array(pil_img)

def create_comparison_gif(
    video_paths, 
    output_path, 
    target_width=None, 
    fps=15, 
    add_label=True
):
    print(f"🎬 开始处理 {len(video_paths)} 个视频...")
    
    video_items = []
    max_frames = 0 # [修改 3] 改为寻找最大帧数
    
    # 1. 初始化读取器
    for path in video_paths:
        if not os.path.exists(path):
            print(f"❌ 警告: 文件不存在，跳过: {path}")
            continue
            
        reader = imageio.get_reader(path)
        length = reader.count_frames()
        
        # 更新最大帧数
        max_frames = max(max_frames, length)
        
        video_items.append({
            "reader": reader, 
            "path": path,
            "length": length,
            "last_frame": None # [修改 4] 用于缓存最后一帧
        })
        print(f"  - Loaded: {os.path.basename(path)} ({length} frames)")
    
    if not video_items:
        print("❌ 没有有效的视频文件！")
        return

    print(f"⏱️  输出 GIF 将延长至 {max_frames} 帧 (短视频播放结束后静止)")
    
    output_frames = []
    
    # 2. 逐帧处理 (循环直到最长的视频结束)
    for i in range(max_frames):
        current_frame_row = []
        
        for item in video_items:
            # [修改 5] 核心逻辑：如果视频没播完就读下一帧，播完了就用最后一帧
            if i < item["length"]:
                try:
                    frame = item["reader"].get_data(i)
                    item["last_frame"] = frame # 更新缓存
                except IndexError:
                    #以此防止可能的索引越界
                    frame = item["last_frame"]
            else:
                # 视频已结束，使用缓存的最后一帧
                frame = item["last_frame"]
            
            # 如果指定了宽度，进行缩放
            if target_width:
                h, w, _ = frame.shape
                scale = target_width / w
                new_h = int(h * scale)
                frame = cv2.resize(frame, (target_width, new_h))
            
            # 添加文字标签
            if add_label:
                # 优化文件名显示，去掉后缀
                label = os.path.basename(item["path"]).split(".")[0]
                frame = draw_text_on_image(frame, label)
                
            current_frame_row.append(frame)
        
        # 3. 横向拼接
        try:
            row_image = np.hstack(current_frame_row)
            output_frames.append(row_image)
        except ValueError as e:
            print(f"⚠️ 拼接失败 (分辨率不一致，尝试强制对齐): {e}")
            # 强制高度对齐策略
            base_h = current_frame_row[0].shape[0]
            resized_row = []
            for img in current_frame_row:
                if img.shape[0] != base_h:
                    img = cv2.resize(img, (img.shape[1], base_h))
                resized_row.append(img)
            row_image = np.hstack(resized_row)
            output_frames.append(row_image)

        if i % 10 == 0:
            print(f"  Processing frame {i}/{max_frames}...", end="\r")

    # 3. 保存 GIF
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"\n💾 正在保存 GIF 到: {output_path} ...")
    imageio.mimsave(output_path, output_frames, fps=fps, loop=0)
    print("✅ 完成！")

    # 清理资源
    for item in video_items:
        item["reader"].close()

if __name__ == "__main__":
    # ================= 配置区域 =================
    
    # 1. 待拼接的视频路径列表
    VIDEO_LIST = [
        "",
        "/mnt/gaoge/DiffSynth-Studio/output_video/Wan2.1-VACE-1.3B_full_interval_dit/epoch-0/left_mp4_path/frames_49_seed_42/1000_Pick up the marker and put it in the bowl.mp4",
        "/mnt/gaoge/DiffSynth-Studio/output_video/Wan2.1-VACE-1.3B_full_interval_dit/epoch-0/left_mp4_path/frames_81_seed_456/1000_Pick up the marker and put it in the bowl.mp4",
        "/mnt/gaoge/DiffSynth-Studio/output_video/Wan2.1-VACE-1.3B_full_interval_dit/epoch-0/left_mp4_path/frames_121_seed_42/1000_Pick up the marker and put it in the bowl.mp4",
    ]
    
    # 2. 输出目录设置
    OUTPUT_DIR = "/mnt/gaoge/DiffSynth-Studio/result_gif/"
    OUTPUT_FILENAME = "epoch0_interval_Pick up the marker and put it in the bowl.gif" 
    
    output_full_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    # 3. 执行拼接
    create_comparison_gif(
        VIDEO_LIST, 
        output_full_path, 
        target_width=400, 
        fps=8, 
        add_label=True
    )