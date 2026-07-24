import imageio_ffmpeg, numpy as np
from PIL import Image

reader = imageio_ffmpeg.read_frames('test3_original.mp4')
meta = next(reader)
w, h = meta['size']
frames = []
for raw in reader:
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
    frames.append(arr.copy())
print('total frames', len(frames))

idxs = [0, 4, 7, 9, 14, 20, 29, 32, 38, 40, 42, 48]
thumbs = [Image.fromarray(frames[i]).resize((320, 213)) for i in idxs]

cols = 4
rows = (len(thumbs) + cols - 1) // cols
sheet = Image.new('RGB', (320 * cols, 213 * rows), (0, 0, 0))
for i, th in enumerate(thumbs):
    x = (i % cols) * 320
    y = (i // cols) * 213
    sheet.paste(th, (x, y))
sheet.save('test3_check.jpg', quality=90)
print('indices used:', idxs)
