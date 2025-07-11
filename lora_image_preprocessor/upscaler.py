# lora_image_preprocessor/upscaler.py
#%%
from PIL import Image
import numpy as np


#%%
def upscale_image(img, upscale_if_large=False, resolution=512):
    w, h = img.size
    max_dim = max(w, h)

    if max_dim >= resolution and not upscale_if_large:
        return img

    scale_factor = resolution / max_dim
    if scale_factor <= 2:
        return img.resize((resolution, resolution), Image.LANCZOS)

    return img.resize((resolution, resolution), Image.LANCZOS)
