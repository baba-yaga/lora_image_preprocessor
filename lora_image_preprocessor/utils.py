# lora_image_preprocessor/utils.py
#%%
from PIL import Image
import os

#%%
def load_image(path):
    return Image.open(path).convert("RGB")

def save_image(img, path):
    img.save(path, quality=95)

#%%
def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)

def resize_with_aspect_ratio(img, max_size):
    w, h = img.size
    scale = min(max_size / w, max_size / h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
