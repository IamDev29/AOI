import os
import io
from typing import Optional, Tuple

import streamlit as st
from PIL import Image
import numpy as np

from utils.preprocess import preprocess_roi
from utils.ocr import extract_text
from utils.validation import validate_text, ValidationResult


st.set_page_config(page_title="IC Marking OCR & Validation", layout="wide")

def load_image(upload) -> Optional[Image.Image]:
    try:
        return Image.open(upload).convert("RGB")
    except Exception:
        return None


def main():
    st.title("IC Marking OCR & Validation")
    st.caption("Upload an IC image, crop the marking ROI, run OCR, and validate.")

    with st.sidebar:
        st.header("Options")
        ocr_engine = st.selectbox(
            "OCR Engine",
            ["EasyOCR", "Tesseract"],
            index=0,
            help="EasyOCR included. Tesseract requires local install of tesseract.exe."
        )
        # Allow specifying Tesseract executable path (Windows default shown)
        default_tess = os.getenv("TESSERACT_CMD") or (
            r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe" if os.name == "nt" else "/usr/bin/tesseract"
        )
        tess_cmd = st.text_input(
            "Tesseract executable path (optional)",
            value=default_tess,
            help="Example on Windows: C\\\Program Files\\\Tesseract-OCR\\\tesseract.exe"
        )
        if tess_cmd:
            os.environ["TESSERACT_CMD"] = tess_cmd
        apply_contrast = st.checkbox("Increase contrast (CLAHE)", value=True)
        apply_binarize = st.checkbox("Binarize (adaptive threshold)", value=True)
        apply_denoise = st.checkbox("Denoise (median blur)", value=True)
        n8n_url = st.text_input("n8n webhook URL (optional)", value=os.getenv("N8N_WEBHOOK_URL", ""))
        st.caption("When Gemini key is set, validation uses Gemini only (no web search).")
        gemini_key = st.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""))
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key
        gemini_model = st.text_input("Gemini Model (optional)", value=os.getenv("GEMINI_MODEL", ""),
                                     help="Use names like gemini-1.5-flash or gemini-1.5-pro (no 'models/' prefix).")
        if gemini_model:
            os.environ["GEMINI_MODEL"] = gemini_model
        st.caption("If Gemini key is missing, DeepSeek/SerpAPI may be used as fallback.")
        serpapi_key = st.text_input("SerpAPI API Key", value=os.getenv("SERPAPI_KEY", ""))
        if serpapi_key:
            os.environ["SERPAPI_KEY"] = serpapi_key
        deepseek_key = st.text_input("DeepSeek API Key", value=os.getenv("DEEPSEEK_API_KEY", ""))
        if deepseek_key:
            os.environ["DEEPSEEK_API_KEY"] = deepseek_key
        deepseek_model = st.text_input("DeepSeek Model (optional)", value=os.getenv("DEEPSEEK_MODEL", ""),
                                      help="If 404 occurs, try deepseek-chat or deepseek-reasoner.")
        if deepseek_model:
            os.environ["DEEPSEEK_MODEL"] = deepseek_model

    uploaded = st.file_uploader("Upload IC image", type=["png", "jpg", "jpeg", "bmp", "tiff"]) 
    if uploaded is None:
        st.info("Upload an image to get started.")
        return

    image = load_image(uploaded)
    if image is None:
        st.error("Could not read the uploaded image.")
        return

    st.subheader("Select Region of Interest (ROI)")
    st.write("Use the handles to crop tightly around the marking.")
    # Try to use streamlit-cropper; fallback to manual sliders if unavailable
    cropped_img = None
    try:
        from streamlit_cropper import st_cropper  # lazy import to avoid module-level crashes
        cropped_img = st_cropper(image, realtime_update=True, box_color=(0, 255, 0), aspect_ratio=None)
    except Exception as e:
        st.warning(f"Interactive cropper unavailable, using manual crop sliders. ({e.__class__.__name__})")
        w, h = image.size
        colA, colB = st.columns(2)
        with colA:
            x = st.slider("Left (x)", 0, max(0, w - 10), 0)
            y = st.slider("Top (y)", 0, max(0, h - 10), 0)
        with colB:
            cw = st.slider("Width", 10, w - x, min(200, w - x))
            ch = st.slider("Height", 10, h - y, min(100, h - y))
        try:
            cropped_img = image.crop((x, y, x + cw, y + ch))
        except Exception:
            cropped_img = image

    if cropped_img is None:
        st.warning("No ROI selected yet.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image, caption="Original", use_column_width=True)
    with col2:
        st.image(cropped_img, caption="Cropped ROI", use_column_width=True)

    # Preprocess
    pre_img = preprocess_roi(cropped_img, contrast=apply_contrast, binarize=apply_binarize, denoise=apply_denoise)
    with col3:
        st.image(pre_img, caption="Preprocessed ROI", use_column_width=True)

    # OCR
    with st.spinner("Running OCR..."):
        text = extract_text(pre_img, engine=ocr_engine)

    st.subheader("Detected Text")
    if text.strip():
        st.code(text)
    else:
        st.warning("No text detected. Try adjusting crop or preprocessing options.")

    # Validation
    with st.spinner("Validating marking..."):
        result: ValidationResult = validate_text(text, webhook_url=n8n_url, mode="serpapi")

    st.subheader("Validation Result")
    status_color = {"PASS": "#2ecc71", "FAIL": "#e74c3c", "WARNING": "#f1c40f"}.get(result.status, "#3498db")
    st.markdown(f"<div style='padding:0.75rem;border-radius:8px;background:{status_color};color:white;font-weight:600;'>"
                f"Status: {result.status}</div>", unsafe_allow_html=True)
    if result.details:
        # Show search results and details
        st.write(result.details)
        # Explainer section parses lines after 'Explainer:' marker
        if "Explainer:" in result.details:
            try:
                expl = result.details.split("Explainer:", 1)[1].strip()
                if expl:
                    with st.expander("Explainer"):
                        st.markdown(expl)
            except Exception:
                pass
        # LLM Analysis section parses lines after 'LLM Analysis:' marker
        if "LLM Analysis:" in result.details:
            try:
                llm = result.details.split("LLM Analysis:", 1)[1].strip()
                if llm:
                    with st.expander("LLM Analysis"):
                        st.markdown(llm)
            except Exception:
                pass
        # Brief summary line
        if "Summary:" in result.details:
            try:
                summary_line = result.details.split("Summary:", 1)[1].splitlines()[0].strip()
                if summary_line:
                    st.info(summary_line)
            except Exception:
                pass
    if result.reference:
        with st.expander("Reference / Source"):
            st.write(result.reference)

    st.divider()
    st.caption("Tip: If you have an n8n workflow, provide its webhook URL to validate against web search or a knowledge base.")


if __name__ == "__main__":
    main()