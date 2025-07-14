# LoRA Image Preprocessor

A Python tool to preprocess images for LoRA training, with optional face cropping, background removal, and captioning.

## Features
- Face-aware square cropping
- Background removal with `rembg`
- LLaVA captioning for full images
- Blendshape detection for facial expressions
- Easy CLI interface

## Installation

```bash
git clone https://github.com/baba-yaga/lora-image-preprocessor.git
cd lora-image-preprocessor
pip install -r requirements.txt
```

## Running

The main script is `lora_image_preprocessor/main.py`. You can run it from the project root directory.

### CLI Options

*   `--input_dir`: (Required) The directory containing the images to process.
*   `--output_dir`: (Required) The directory where the processed images and captions will be saved.
*   `--face_only`: If specified, the script will crop the image to the detected face. Note that in images where both eyes are not visible, or in cartoon-like images, the face is usually not detected.
*   `--remove_bg`: If specified, the background of the images will be removed.
*   `--resolution`: The target output resolution for the images (default: 512).
*   `--output_format`: The output image format (e.g., png, jpg) (default: png).
*   `--face_crop_padding`: The padding factor for face cropping (default: 1.8).
*   `--no_caption`: If specified, the script will not generate captions.

### Examples

**Process all images in a directory:**

```bash
python -m lora_image_preprocessor.main --input_dir input_images --output_dir output_images
```

**Process only images with faces:**

```bash
python -m lora_image_preprocessor.main --input_dir input_images --output_dir output_images --face_only
```

**Remove the background from all images:**

```bash
python -m lora_image_preprocessor.main --input_dir input_images --output_dir output_images --remove_bg
```

**Process images without generating captions:**

```bash
python -m lora_image_preprocessor.main --input_dir input_images --output_dir output_images --no_caption
