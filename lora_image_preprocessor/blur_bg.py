import cv2
import mediapipe as mp
import numpy as np
import os

# Input and output folders
INPUT_DIR = 'input_images'
OUTPUT_DIR = 'input_images_512_bg_blured'
os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_face_detection = mp.solutions.face_detection
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# For each image in the input folder
for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        continue

    path = os.path.join(INPUT_DIR, fname)
    image = cv2.imread(path)
    if image is None:
        print(f"Cannot open {fname}")
        continue

    # Face detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detector:
        results = face_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.detections:
            print(f"No face found in {fname}")
            continue

        # Use the first detected face
        det = results.detections[0]
        bbox = det.location_data.relative_bounding_box
        h, w, _ = image.shape
        # Original bounding box (as float)
        bx = bbox.xmin * w
        by = bbox.ymin * h
        bw = bbox.width * w
        bh = bbox.height * h

        # Center of the box
        cx = bx + bw / 2
        cy = by + bh / 2

        # New size with 1.5x padding (box size × 1.5)
        scale = 1.5
        new_bw = bw * scale
        new_bh = bh * scale

        # New top-left and bottom-right, ensuring we stay within image bounds
        x_min = int(max(cx - new_bw / 2, 0))
        x_max = int(min(cx + new_bw / 2, w))
        y_min = int(max(cy - new_bh / 2, 0))
        y_max = int(min(cy + new_bh / 2, h))

        crop = image[y_min:y_max, x_min:x_max]

    # Person segmentation (to create a foreground mask)
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter:
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = segmenter.process(rgb_crop)
        mask = result.segmentation_mask
        # Clean binary mask for the person
        mask = (mask > 0.1).astype(np.uint8)

        # Blur background with a random kernel size
        sigma = np.random.uniform(5, 25)
        blurred = cv2.GaussianBlur(crop, (0, 0), sigma)
        mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        combined = crop * mask_3ch + blurred * (1 - mask_3ch)
        combined = combined.astype(np.uint8)

    # Resize to 512x512 (padding if necessary)
    target_size = 512
    h_c, w_c = combined.shape[:2]
    scale = target_size / max(h_c, w_c)
    resized = cv2.resize(combined, (int(w_c * scale), int(h_c * scale)))
    top = (target_size - resized.shape[0]) // 2
    bottom = target_size - resized.shape[0] - top
    left = (target_size - resized.shape[1]) // 2
    right = target_size - resized.shape[1] - left
    final_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(127,127,127))

    # Save result
    out_path = os.path.join(OUTPUT_DIR, fname)
    cv2.imwrite(out_path, final_img)
    print(f"Processed {fname} → {out_path}")

print("All images processed.")
