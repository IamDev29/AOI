# IC Marking OCR & Validation

Streamlit app to extract IC (integrated circuit) marking text via OCR and validate authenticity using LLMs and/or web search.

## Overview

- Upload an IC image, crop the marking ROI, and run OCR (EasyOCR or Tesseract).
- Validate the detected text via:
  - Gemini (Generative Language API) — preferred when `GEMINI_API_KEY` is set.
  - DeepSeek (LLM) — optional, can use SerpAPI search context.
  - SerpAPI Google Search — heuristic validation using top results.
  - n8n webhook — custom pipeline if configured.
  - Local heuristics — simple pattern matching as last resort.
- Clear summary and analysis are shown, with error hints for common API issues.

## Features

- Interactive ROI cropper (falls back to sliders if unavailable).
- Preprocessing: contrast (CLAHE), binarize (adaptive threshold), denoise (median blur).
- OCR via EasyOCR or Tesseract (pytesseract).
- LLM validation with Gemini or DeepSeek.
- Optional web validation with SerpAPI.
- Robust error diagnostics (e.g., 400/401/402/404/429) with guidance.

## Requirements

- Python 3.10+ recommended.
- Windows, macOS, or Linux. The app runs CPU-only; GPU is optional.
 - Optional: Tesseract OCR installed if you want to use the Tesseract engine.

## Setup

On Windows PowerShell:

1) Create and activate a virtual environment

```
python -m venv .venv
.\.venv\Scripts\activate
```

2) Install dependencies

```
pip install -r requirements.txt
```

3) Run the app

```
streamlit run app.py
```

Then open the displayed `Local URL` in your browser.

## Configuration

All keys can be entered directly in the app sidebar (they’re stored in-memory during the session) or set as environment variables.

- `Gemini API Key`: Generative Language API key from Google AI Studio.
- `Gemini Model (optional)`: use names like `gemini-1.5-flash`, `gemini-1.5-pro`, or `gemini-2.5-flash`.
  - Do not include the `models/` prefix.
- `DeepSeek API Key`: DeepSeek API key.
- `DeepSeek Model (optional)`: e.g., `deepseek-reasoner` or `deepseek-chat`.
- `SerpAPI API Key`: Google Search proxy for web validation.
- `n8n webhook URL (optional)`: POST endpoint to a custom validation workflow.

Environment variables (optional):

```
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-1.5-flash
DEEPSEEK_API_KEY=...
DEEPSEEK_MODEL=deepseek-reasoner
SERPAPI_KEY=...
N8N_WEBHOOK_URL=https://...
```

### Tesseract (optional)

- Install Tesseract OCR locally.
  - Windows default path: `C:\\Program Files\\Tesseract-OCR\\tesseract.exe`
  - Linux default path: `/usr/bin/tesseract`
- Configure the executable:
  - In the app sidebar, set “Tesseract executable path (optional)” to your `tesseract.exe` path.
  - Or set `TESSERACT_CMD` in the environment before running the app.

## Validation Flow

The app prefers Gemini-only when `GEMINI_API_KEY` is present:

1) Gemini (LLM-only, OCR text)
2) DeepSeek (LLM, optionally with SerpAPI web context)
3) SerpAPI-only heuristics
4) n8n webhook (if provided)
5) Local heuristics

Gemini client behavior:
- Tries the official SDK (`google.genai`) when installed; falls back to REST if needed.
- Normalizes model names by stripping a leading `models/` prefix.
- On model-related errors (400/404), automatically retries with `gemini-1.5-flash` and `gemini-1.5-pro`.
- Adds “Response Body” under “LLM Analysis” for clearer diagnostics.

DeepSeek client behavior:
- Uses `v1/chat/completions` with automatic endpoint fallback to the legacy path if needed.
- On 404, tries an alternate model (`deepseek-chat` vs `deepseek-reasoner`).
- Provides specific error hints for 401, 402, 404, 429.

## Using the App

1) Upload an IC image.
2) Choose OCR Engine: EasyOCR or Tesseract.
3) If using Tesseract, set the executable path (Windows example shown in the sidebar).
4) Crop the marking ROI using the cropper (or sliders).
5) Adjust preprocessing options as needed.
6) Enter the API keys in the sidebar as desired.
7) Run OCR and validation; review:
   - Status card (PASS/FAIL/WARNING)
   - “Detected Text”
   - “LLM Analysis” (includes response body on errors)
   - “Search results” and “Explainer” (when web search is used)
   - “Reference / Source”

## Troubleshooting

Common Gemini errors:
- `HTTP 400 Bad request: unexpected model name format` — remove `models/` prefix; use `gemini-1.5-flash`, `gemini-1.5-pro`, or `gemini-2.5-flash`.
- `API key not valid for Generative Language API` — create a new key in Google AI Studio (not Vertex-only keys).
- `Rate limited` — reduce request rate or check quota.
- `Unsupported location` — use `gemini-1.5-flash` or enable the model in your region.

Common DeepSeek errors:
- `HTTP 401 Unauthorized` — invalid key.
- `HTTP 402 Payment required` — check billing/credits.
- `HTTP 404 Not found` — wrong endpoint/model; try `deepseek-reasoner` or `deepseek-chat`.
- `HTTP 429 Rate limited` — slow down or check quota.

OCR / UI notes:
- CPU-only is fine; GPU warnings can be ignored.
- If “No text detected”, tighten the crop and try different preprocessing options.

Tesseract issues:
- `tesseract.exe not found` — set the correct path in the sidebar or `TESSERACT_CMD`.
- Empty results — adjust ROI/preprocessing; default config uses `--psm 6`.

## Project Structure

```
app.py                      # Streamlit UI and workflow
utils/
  preprocess.py             # ROI preprocessing (contrast/binarize/denoise)
  ocr.py                    # OCR via EasyOCR and Tesseract
  validation.py             # Validation orchestrator & fallbacks
  gemini_client.py          # Gemini SDK/REST client with model normalization & fallbacks
  deepseek_client.py        # DeepSeek client with endpoint/model fallbacks
  search_client.py          # SerpAPI query and caching
```

## Customization

- Add known good markings in `utils/validation.py` under `KNOWN_MARKINGS`.
- Adjust prompts/policy in `utils/gemini_client.py` or `utils/deepseek_client.py`.
- Integrate your own webhook via `N8N_WEBHOOK_URL`.

## Notes

- Keys entered in the sidebar are injected into `os.environ` for the current session only.
- No license file is included; add one if needed for distribution.