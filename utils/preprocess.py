from typing import Optional

from PIL import Image
import numpy as np  # Lazy usage; avoid heavy ops at import


def pil_to_cv(img: Image.Image) -> np.ndarray:
    # Import cv2 lazily to avoid triggering heavy native loads at startup
    import cv2 as cv
    arr = np.array(img)
    return cv.cvtColor(arr, cv.COLOR_RGB2BGR)


def cv_to_pil(img_cv: np.ndarray) -> Image.Image:
    import cv2 as cv
    rgb = cv.cvtColor(img_cv, cv.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def preprocess_roi(img: Image.Image, contrast: bool = True, binarize: bool = True, denoise: bool = True) -> Image.Image:
    """Apply simple preprocessing to aid OCR: grayscale, CLAHE, threshold, denoise.
    Imports OpenCV lazily to improve app stability on Windows environments.
    """
    import cv2 as cv

    cv_img = pil_to_cv(img)

    # Grayscale
    gray = cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY)

    # Contrast via CLAHE
    if contrast:
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # Denoise (median blur)
    if denoise:
        gray = cv.medianBlur(gray, 3)

    # Binarize (adaptive threshold)
    if binarize:
        th = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 31, 2)
    else:
        th = gray

    return cv_to_pil(th)