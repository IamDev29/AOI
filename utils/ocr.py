from typing import List
import os

from PIL import Image


def _pil_to_np(img: Image.Image):
    import numpy as np  # lazy import to avoid heavy init on Windows
    return np.array(img)


def _load_easyocr_reader():
    # Lazy import to keep app responsive; torch loads may be heavy.
    import easyocr
    # English and common Latin scripts; extend as needed
    return easyocr.Reader(["en"], gpu=False)


def _tesseract_default_path() -> str:
    # Prefer Windows default if present; otherwise fall back to common Linux path
    if os.name == "nt":
        return r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    return "/usr/bin/tesseract"


def _run_tesseract(img: Image.Image) -> str:
    # Lazy import to avoid overhead when not selected
    try:
        import pytesseract
    except Exception as e:
        raise RuntimeError(f"pytesseract not available: {e}")

    # Configure executable path from env or sensible default
    tess_cmd = os.getenv("TESSERACT_CMD") or _tesseract_default_path()
    if tess_cmd:
        try:
            pytesseract.pytesseract.tesseract_cmd = tess_cmd
        except Exception:
            # Non-fatal; pytesseract may still find tesseract on PATH
            pass

    # Use a reasonable PSM for block of text; adjust if needed
    config = "--psm 6"
    text = pytesseract.image_to_string(img, lang="eng", config=config)
    return text.strip()


_reader = None


def extract_text(img: Image.Image, engine: str = "EasyOCR") -> str:
    """Extract text from preprocessed ROI using selected OCR engine.
    Supports EasyOCR and Tesseract (pytesseract).
    """
    # If Tesseract requested, try it first; fall back to EasyOCR on errors
    if str(engine).strip().lower() == "tesseract":
        try:
            t_text = _run_tesseract(img)
            if t_text:
                return t_text
        except Exception:
            # Fall back to EasyOCR if Tesseract path or import fails
            pass

    # EasyOCR path
    global _reader
    if _reader is None:
        _reader = _load_easyocr_reader()

    np_img = _pil_to_np(img)
    # EasyOCR expects RGB NumPy arrays
    results = _reader.readtext(np_img, detail=0)
    # Join lines into single text block
    text = "\n".join([r.strip() for r in results if isinstance(r, str)])
    return text