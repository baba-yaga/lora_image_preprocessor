# lora_image_preprocessor/general_cropper.py
#%%
from PIL import Image, ImageOps
from lora_image_preprocessor.utils import resize_with_aspect_ratio
from lora_image_preprocessor.upscaler import upscale_image
from lora_image_preprocessor.captioner import generate_caption

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
    resized = upscale_image(resized, upscale_if_large=True)
    return resized
