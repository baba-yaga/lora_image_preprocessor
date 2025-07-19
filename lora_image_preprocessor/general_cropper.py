# lora_image_preprocessor/general_cropper.py
#%%
from PIL import Image, ImageOps
from lora_image_preprocessor.utils import resize_with_aspect_ratio

#%%
def make_square(img, bg_color=(0, 0, 0)):
    w, h = img.size
    size = max(w, h)
    new_img = Image.new("RGB", (size, size), bg_color)
    offset = ((size - w) // 2, (size - h) // 2)
    new_img.paste(img, offset)
    return new_img

#%%
def process_whole_image(img, base_name, output_dir, resolution=512):
    padded = make_square(img, bg_color=(0, 0, 0))
    resized = padded.resize((resolution, resolution), Image.LANCZOS)
    return resized
