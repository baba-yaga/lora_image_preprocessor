# lora_image_preprocessor/main.py
import os
import argparse
import warnings
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

from lora_image_preprocessor.utils import load_image, save_image, ensure_dir

def preload_models(face_only: bool, remove_bg: bool, general_processing: bool):
    print("Pre-loading models...")
    if face_only:
        print("- Loading face detection model...")
        from lora_image_preprocessor.face_cropper import process_face_image
        import numpy as np
        from PIL import Image
        dummy_img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        process_face_image(dummy_img)
        print("  - Face detection model loaded.")

    if general_processing:
        from lora_image_preprocessor.captioner import load_caption_model
        load_caption_model()

    if remove_bg:
        from lora_image_preprocessor.background_remover import load_background_remover_model
        load_background_remover_model()

    print("Model pre-loading complete.")


def process_images(input_dir, output_dir, face_only, remove_bg, resolution=512, output_format="png", face_crop_padding=1.5):
    from lora_image_preprocessor.face_cropper import process_face_image
    from lora_image_preprocessor.general_cropper import process_whole_image
    from lora_image_preprocessor.background_remover import remove_background
    from lora_image_preprocessor.captioner import generate_caption
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    allowed_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in allowed_extensions]

    if not image_files:
        print("No images found in the input directory.")
        return

    general_processing = not face_only
    preload_models(face_only, remove_bg, general_processing)

    for img_path in tqdm(image_files, desc="Processing images"):
        print(f"\nWorking on {img_path.name}:")
        img = load_image(img_path)
        base_name = img_path.stem

        final_img = None
        prompt = None
        blendshapes = None
        out_path = None
        txt_path = None

        if face_only:
            print("- Detecting faces...", end="", flush=True)
            processed_img, face_detected, blendshapes = process_face_image(img, resolution, padding_factor=face_crop_padding)
            if face_detected:
                print("Done.")
                print("  - Detected one face, cropping.")
                final_img = processed_img
                out_path = output_dir / f"{base_name}_face_{resolution}.{output_format}"
                txt_path = output_dir / f"{base_name}_face_{resolution}.txt"
            else:
                print("Done.")
                print("  - No faces detected. Skipping.")
                continue
        else:
            print("- Processing whole image...")
            processed_img, prompt = process_whole_image(img, base_name, output_dir, resolution)
            final_img = processed_img
            out_path = output_dir / f"{base_name}_{resolution}.{output_format}"
            txt_path = output_dir / f"{base_name}_{resolution}.txt"

        if final_img and remove_bg:
            print("- Removing background...", end="", flush=True)
            final_img = remove_background(final_img)
            print("Done.")

        if final_img and out_path:
            save_image(final_img, out_path)

        if txt_path:
            print("- Creating image description...", end="", flush=True)
            prompt = generate_caption(final_img, blendshapes)
            txt_path.write_text(prompt, encoding="utf-8")
            print("Done.")

    print("\nProcessing complete!")
    print(f"Processed {len(image_files)} images.")
    print(f"Output saved to: {output_dir.resolve()}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--face_only", action="store_true")
    parser.add_argument("--remove_bg", action="store_true")
    parser.add_argument("--resolution", type=int, default=512, help="Target output resolution (default: 512)")
    parser.add_argument("--output_format", type=str, default="png", help="Output image format (e.g., png, jpg)")
    parser.add_argument("--face_crop_padding", type=float, default=1.8, help="Padding factor for face cropping.")
    args = parser.parse_args()

    process_images(args.input_dir, args.output_dir, args.face_only, args.remove_bg, args.resolution, args.output_format, args.face_crop_padding)
