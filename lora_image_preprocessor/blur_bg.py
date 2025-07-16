import cv2
import numpy as np
from PIL import Image
from rembg import remove
from rembg.bg import new_session

# Create a session that forces CPU usage to avoid CUDA errors
cpu_session = new_session(providers=['CPUExecutionProvider'])

def blur_background(pil_img: Image.Image) -> Image.Image:
    """Blurs the background of an image using a Gaussian blur, keeping the foreground sharp.

    Args:
        pil_img: The input image (PIL.Image).

    Returns:
        The image with a blurred background (PIL.Image).
    """
    # Get the foreground mask from the image using the CPU-only session
    mask_pil = remove(pil_img, only_mask=True, session=cpu_session)

    # Convert PIL image to OpenCV format (NumPy array) and from RGB to BGR
    ocv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Apply Gaussian blur to the entire image
    sigma = np.random.uniform(3, 15)
    blurred_ocv_img = cv2.GaussianBlur(ocv_img, (0, 0), sigma)

    # Convert the single-channel mask to a 3-channel version
    mask_ocv = cv2.cvtColor(np.array(mask_pil), cv2.COLOR_GRAY2BGR)
    
    # Normalize mask values to the 0.0-1.0 range
    mask_float = mask_ocv.astype(np.float32) / 255.0

    # Combine the original foreground with the blurred background.
    # Where the mask is white (1.0), we use the original image pixel.
    # Where the mask is black (0.0), we use the blurred image pixel.
    combined_ocv = ocv_img * mask_float + blurred_ocv_img * (1 - mask_float)
    combined_ocv = combined_ocv.astype(np.uint8)

    # Convert the final image back to PIL format from BGR to RGB
    blurred_pil_img = Image.fromarray(cv2.cvtColor(combined_ocv, cv2.COLOR_BGR2RGB))
    
    return blurred_pil_img
