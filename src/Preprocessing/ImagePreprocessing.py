import cv2
import numpy as np
from PIL import Image


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE to a handwritten text image.

    CLAHE stands for Contrast Limited Adaptive Histogram Equalization.
    It improves local contrast so that faint handwriting may become
    easier to distinguish from the background.

    Parameters:
        image: Input image (PIL Image or NumPy array).
        clip_limit: Limits excessive contrast enhancement.
        tile_grid_size: Divides the image into smaller regions.

    Returns:
        Enhanced image in PIL format.
    """
    if isinstance(image, Image.Image): # convert pil to numpy arrayy
        image = np.array(image)

    if len(image.shape) == 3: #conert the image to grayscale if it has rgb value
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)    

    clahe = cv2.createCLAHE(  #create CLAHE object with specified parameters
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )    

    enhanced_image = clahe.apply(image)  #apply CLAHE to the image

    return Image.fromarray(enhanced_image)  #convert the enhanced image back to PIL format