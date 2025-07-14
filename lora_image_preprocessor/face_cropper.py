# lora_image_preprocessor/face_cropper.py
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from lora_image_preprocessor.general_cropper import make_square

# Import the new API components
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Create a FaceLandmarker object
base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       num_faces=1)
landmarker = vision.FaceLandmarker.create_from_options(options)

def process_face_image(pil_img, resolution=512, padding_factor=1.5):
    np_img = np.array(pil_img.convert('RGB'))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_img)
    
    blendshapes = []
    face_detected = False

    # Run the face landmarker
    detection_result = landmarker.detect(mp_image)

    if detection_result.face_landmarks:
        face_detected = True
        face_landmarks = detection_result.face_landmarks[0]
        
        # Extract blendshapes
        if detection_result.face_blendshapes:
            blendshapes = {shape.category_name: shape.score for shape in detection_result.face_blendshapes[0]}

        # Get bounding box from landmarks
        h, w, _ = np_img.shape
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        for landmark in face_landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            if x < x_min:
                x_min = x
            if x > x_max:
                x_max = x
            if y < y_min:
                y_min = y
            if y > y_max:
                y_max = y
        
        # Add padding
        face_width = x_max - x_min
        face_height = y_max - y_min
        center_x = x_min + face_width // 2
        center_y = y_min + face_height // 2
        
        crop_size = max(face_width, face_height)
        padded_crop_size = int(crop_size * padding_factor)

        crop_left = max(0, center_x - padded_crop_size // 2)
        crop_top = max(0, center_y - padded_crop_size // 2)
        crop_right = min(w, crop_left + padded_crop_size)
        crop_bottom = min(h, crop_top + padded_crop_size)

        # Adjust crop box to be square
        crop_width = crop_right - crop_left
        crop_height = crop_bottom - crop_top

        if crop_width > crop_height:
            diff = crop_width - crop_height
            crop_top = max(0, crop_top - diff // 2)
            crop_bottom = min(h, crop_bottom + (diff - diff // 2))
        elif crop_height > crop_width:
            diff = crop_height - crop_width
            crop_left = max(0, crop_left - diff // 2)
            crop_right = min(w, crop_right + (diff - diff // 2))

        cropped = pil_img.crop((crop_left, crop_top, crop_right, crop_bottom))
    else:
        print("[WARN] No face detected in image. Using fallback square crop.")
        cropped = make_square(pil_img)


    resized = cropped.resize((resolution, resolution), Image.LANCZOS)
    return resized, face_detected, blendshapes
