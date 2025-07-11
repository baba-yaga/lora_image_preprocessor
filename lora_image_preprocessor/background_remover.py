# lora_image_preprocessor/background_remover.py
#%%
from PIL import Image
import numpy as np

#%%
try:
    from rembg import new_session, remove
    rembg_available = True
    cpu_session = new_session(providers=["CPUExecutionProvider"])
except ImportError:
    rembg_available = False
    cpu_session = None

#%%
def remove_background(img, bg_color=(0, 177, 64)):
    if not rembg_available:
        print("[WARN] rembg not available, skipping background removal")
        return img

    # Remove background
    result = remove(img, session=cpu_session)

    # Convert RGBA to RGB with solid background
    if result.mode == "RGBA":
        bg = Image.new("RGB", result.size, bg_color)
        bg.paste(result, mask=result.split()[3])
        return bg

    return result.convert("RGB")
