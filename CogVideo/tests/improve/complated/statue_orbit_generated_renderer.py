import sys, math
from PIL import Image, ImageDraw
import numpy as np
import imageio

# Parameters
width, height = 720, 480
frames = 49
fps = 8
statue_color = (120, 120, 120)
bg_color = (34, 139, 34)  # green courtyard

def create_frame(angle_deg):
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Statue: a simple gray rectangle at center
    statue_w, statue_h = 100, 200
    cx, cy = width // 2, height // 2
    rect = [cx - statue_w//2, cy - statue_h//2,
            cx + statue_w//2, cy + statue_h//2]
    draw.rectangle(rect, fill=statue_color)

    # Rotate the whole image to simulate camera orbit (viewpoint change)
    rotated = img.rotate(angle_deg, resample=Image.BICUBIC, center=(cx, cy), expand=False)
    return np.array(rotated)

def main():
    out_path = sys.argv[1]
    frames_list = []
    for i in range(frames):
        angle = (360.0 / (frames - 1)) * i
        frame = create_frame(angle)
        frames_list.append(frame)

    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', bitrate=5000000,
                                ffmpeg_log_level="quiet")
    for f in frames_list:
        writer.append_data(f)
    writer.close()

if __name__ == "__main__":
    main()