from typing import List

from PIL import Image

def _pil_to_np(img: Image.Image):
    import numpy as np  # lazy import to avoid heavy init on Windows
    return np.array(img)


def _load_easyocr_reader():
    # Lazy import to keep app responsive; torch loads may be heavy.
    import easyocr
    # English and common Latin scripts; extend as needed
    return easyocr.Reader(["en"], gpu=False)


_reader = None


def extract_text(img: Image.Image, engine: str = "EasyOCR") -> str:
    """Extract text from preprocessed ROI using EasyOCR.
    Tesseract could be added later if installed; for hackathon speed we stick to EasyOCR.
    """
    global _reader
    if engine != "EasyOCR":
        engine = "EasyOCR"

    if _reader is None:
        _reader = _load_easyocr_reader()

    np_img = _pil_to_np(img)
    # EasyOCR expects RGB NumPy arrays
    results = _reader.readtext(np_img, detail=0)
    # Join lines into single text block
    text = "\n".join([r.strip() for r in results if isinstance(r, str)])
    return text