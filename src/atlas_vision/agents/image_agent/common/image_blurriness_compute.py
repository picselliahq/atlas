import logging
import os
from typing import Any

import cv2
import numpy as np
from numpy import floating
from PIL import Image

logger = logging.getLogger(__name__)


def assess_bluriness_and_corruption(
    filename: str, blur_threshold: float = 90.0
) -> tuple[
    bool | Any,
    bool,
    float | Any,
    int | None,
    int | None,
    int | None,
    Any | None,
    Any,
    floating[Any] | None,
]:
    """
    Download the image from asset_url, then assess whether it is corrupted and/or blurry.
    A blur score is computed using the variance of the Laplacian of the grayscale image.
    Additionally, the average color, luminance, and contrast are computed.
    """
    try:
        # Open image with PIL
        file_size_bytes = os.stat(filename).st_size
        image = Image.open(filename)
        image.load()  # Force loading the image to catch potential corruption
        width, height = image.size
        # Convert the image to a NumPy array
        image_cv = np.array(image)
        # Convert to grayscale if the image is colored (3 or 4 channels)
        if len(image_cv.shape) == 3 and image_cv.shape[2] in [3, 4]:
            image_gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image_cv
        # Compute the variance of the Laplacian (a measure of blur)
        blur_score = cv2.Laplacian(image_gray, cv2.CV_64F).var()
        is_blurry = blur_score < blur_threshold
        is_corrupted = False

        avg_color = image_cv.mean(axis=(0, 1))

        # Compute the luminance for each pixel using Rec. 709:
        # Luminance Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
        luminance = (
            0.2126 * image_cv[:, :, 0]
            + 0.7152 * image_cv[:, :, 1]
            + 0.0722 * image_cv[:, :, 2]
        )

        # Average luminance over the entire image
        luminance_value = luminance.mean(axis=(0, 1))

        # Contrast can be estimated as the standard deviation of the luminance values
        contrast = np.std(luminance)
    except Exception as e:
        # If any error occurs, mark the asset as corrupted
        logger.info(f"Error processing image {filename}: {e}")
        is_corrupted = True
        is_blurry = False
        blur_score = 0.0
        width, height, file_size_bytes = None, None, None
        avg_color = None
        luminance_value = None
        contrast = None

    return (
        is_blurry,
        is_corrupted,
        blur_score,
        width,
        height,
        file_size_bytes,
        avg_color,
        luminance_value,
        contrast,
    )
