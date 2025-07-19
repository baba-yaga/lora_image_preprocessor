# lora_image_preprocessor/main.py
import os
import argparse
import warnings
from lora_image_preprocessor.blur_bg import blur_background
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

from lora_image_preprocessor.utils import load_image, save_image, ensure_dir

def preload_models(face_only: bool, remove_bg: bool, general_processing: bool, no_caption: bool):
    print("Pre-loading models...")
    if face_only:
        print("- Loading face detection model...")
        from lora_image_preprocessor.face_cropper import process_face_image
        import numpy as np
        from PIL import Image
        dummy_img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        process_face_image(dummy_img)
        print("  - Face detection model loaded.")

    if general_processing and not no_caption:
        from lora_image_preprocessor.captioner import load_blip_model, load_llava_model
        load_blip_model()
        load_llava_model()

    if remove_bg:
        from lora_image_preprocessor.background_remover import remove_background

    print("Model pre-loading complete.")


def process_images(input_dir, output_dir, face_only, remove_bg, no_caption, resolution=512, output_format="png", face_crop_padding=1.8):
    from lora_image_preprocessor.face_cropper import process_face_image
    from lora_image_preprocessor.general_cropper import process_whole_image
    
    if output_dir is None:
        output_dir = Path(input_dir)
    
    
    from lora_image_preprocessor.background_remover import remove_background
    from lora_image_preprocessor.captioner import generate_captions
    
    # Save processed files to directory with the resolution value, face or background indication appended
    input_dir = Path(input_dir)
    output_dir_parts = [str(output_dir), str(resolution)]

    if face_only:
        output_dir_parts.append("face")
    if remove_bg:
        output_dir_parts.append("no_bg")
    if args.blur_bg:
        output_dir_parts.append("blurred_bg")

    output_dir = Path("_".join(output_dir_parts))
    ensure_dir(output_dir)


    allowed_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in allowed_extensions]

    if not image_files:
        print("No images found in the input directory.")
        return

    general_processing = not face_only
    preload_models(face_only, remove_bg, general_processing, no_caption)

    for img_path in tqdm(image_files, desc="Processing images"):
        print(f"\nWorking on {img_path.name}:")
        img = load_image(img_path)
        base_name = img_path.stem
        out_path = output_dir / f"{base_name}.{output_format}"
        txt_path = output_dir / f"{base_name}.txt"


        final_img = None
        prompt = None
        blendshapes = None

        if face_only:
            print("- Detecting faces...", end="", flush=True)
            processed_img, face_detected, blendshapes = process_face_image(img, resolution, padding_factor=face_crop_padding)
            if face_detected:
                print("Done.")
                print("  - A face detected, cropping.")
                final_img = processed_img
            else:
                print("Done.")
                print("  - No faces detected. Skipping.") 
                continue
        else:
            print("- Processing whole image...")
            _, _, blendshapes = process_face_image(img, resolution, padding_factor=face_crop_padding)
            processed_img = process_whole_image(img, base_name, output_dir, resolution)
            final_img = processed_img

        if txt_path and not no_caption:
            print("- Creating image captions...")
            prompt = generate_captions(final_img)
            if blendshapes:
                emotion_prompt = "\n\nThe ARKit expression blendshape scores:"
                for emotion, score in blendshapes.items():
                    if score > 0.15:
                        emotion_prompt += f"\n{emotion} ({score:.2f}),"
                if emotion_prompt.endswith(","):
                    prompt += emotion_prompt[:-1] + "." # remove last comma
            txt_path.write_text(prompt, encoding="utf-8")
            print("Captions and blendshapes saved.")

        if final_img and remove_bg:
            print("- Removing background...", end="", flush=True)
            final_img = remove_background(final_img)
            print("Done.")

        if args.blur_bg and final_img is not None:
            print("- Blurring background...", end="", flush=True)
            final_img = blur_background(final_img)
            print("Done.")

        base_name = img_path.stem
        out_path = output_dir / f"{base_name}.{output_format}"

        if final_img and out_path:
            save_image(final_img, out_path)

    print("\nProcessing complete!")
    print(f"Processed {len(image_files)} images.")
    print(f"Output saved to: {output_dir.resolve()}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=False)
    parser.add_argument("--face_only", action="store_true")
    parser.add_argument("--remove_bg", action="store_true")
    parser.add_argument("--blur_bg", action="store_true", help="Apply background blur.")
    parser.add_argument("--resolution", type=int, default=512, help="Target output resolution (default: 512)")
    parser.add_argument("--output_format", type=str, default="png", help="Output image format (e.g., png, jpg)")
    parser.add_argument("--face_crop_padding", type=float, default=1.8, help="Padding factor for face cropping.")
    parser.add_argument("--no_caption", action="store_true", help="Do not generate captions.")
    args = parser.parse_args()

    process_images(args.input_dir, args.output_dir, args.face_only, args.remove_bg, args.no_caption, args.resolution, args.output_format, args.face_crop_padding)
