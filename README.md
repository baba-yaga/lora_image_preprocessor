# LoRA Image Preprocessor

A Python tool to preprocess images for LoRA training, with optional face cropping, RealESRGAN/GFPGAN upscaling, background removal, and captioning using BLIP.

## Features
- Face-aware square cropping
- RealESRGAN and GFPGAN upscaling
- Background removal with `rembg`
- BLIP captioning for full images
- Easy CLI interface

## Installation

```bash
git clone https://github.com/yourusername/lora-image-preprocessor.git
cd lora-image-preprocessor
pip install -r requirements.txt


## Running
Assuming the images are in the subdirectory `images` and the output should go to `output`, and we start python in lora