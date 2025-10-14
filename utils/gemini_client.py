import os
import json
import requests


def _build_payload(ocr_text: str) -> dict:
    system_instruction = (
        "You are an expert IC authenticity auditor. Given ONLY the OCR text from an IC marking, "
        "classify the chip as REAL (genuine), FAKE (counterfeit/clone), or UNCERTAIN. "
        "Return a compact JSON with keys: status in [PASS, FAIL, WARNING] and reason (one short sentence)."
    )
    user_text = f"OCR text:\n{ocr_text.strip()}\n"
    return {
        "systemInstruction": {"role": "system", "parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {"temperature": 0.2},
    }


def _call_gemini(api_key: str, model: str, payload: dict) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    resp = requests.post(url, json=payload, timeout=25)
    if not resp.ok:
        hint = ""
        body = resp.text or ""
        if resp.status_code == 401:
            hint = "Unauthorized: check GEMINI_API_KEY"
        elif resp.status_code == 404:
            hint = "Model not found: set GEMINI_MODEL (e.g., gemini-1.5-flash or gemini-1.5-pro)"
        elif resp.status_code == 429:
            hint = "Rate limited: slow down or check quota"
        elif resp.status_code == 400:
            if "API key not valid" in body or "not valid for this API" in body:
                hint = "API key not valid for Generative Language API: create a new key in Google AI Studio"
            elif "unsupported location" in body.lower():
                hint = "Model unsupported in region: try gemini-1.5-flash or enable billing/region"
            elif "model" in body.lower():
                hint = "Bad request: verify GEMINI_MODEL (e.g., gemini-1.5-flash or gemini-1.5-pro)"
            else:
                hint = "Bad request: verify payload and model name"
        return {"error": f"HTTP {resp.status_code} {hint}".strip(), "text": body}

    data = resp.json()
    # Safety block handling
    candidates = data.get("candidates") or []
    if candidates and str(candidates[0].get("finishReason", "")).upper() == "SAFETY":
        return {"status": "WARNING", "reason": "Content blocked by safety filters"}

    text_out = ""
    if candidates:
        parts = candidates[0].get("content", {}).get("parts") or []
        text_out = "\n".join([p.get("text", "") for p in parts])

    if not text_out:
        return {"error": "Empty response from Gemini"}

    # Parse JSON block; fallback to keyword heuristic
    try:
        parsed = json.loads(text_out)
        status = str(parsed.get("status", "WARNING")).upper()
        reason = parsed.get("reason") or "No reason provided."
        if status not in {"PASS", "FAIL", "WARNING"}:
            status = "WARNING"
        return {"status": status, "reason": reason}
    except Exception:
        low = text_out.lower()
        status = "WARNING"
        if any(k in low for k in ["pass", "real", "genuine"]):
            status = "PASS"
        elif any(k in low for k in ["fail", "fake", "counterfeit"]):
            status = "FAIL"
        return {"status": status, "reason": text_out.strip()[:300]}


def _normalize_model(model: str) -> str:
    m = (model or "").strip()
    # Accept both "gemini-1.5-flash" and "models/gemini-1.5-flash"; normalize to bare name
    if m.lower().startswith("models/"):
        m = m.split("/", 1)[1]
    return m


def _classify_with_sdk(model: str, ocr_text: str) -> dict | None:
    """Attempt classification via official google.genai SDK.
    Returns a result dict on success or an error dict on failure. If SDK is missing, returns None.
    """
    try:
        from google import genai
    except Exception:
        return None

    prompt = (
        "You are an expert IC authenticity auditor. Given ONLY the OCR text from an IC marking, "
        "classify the chip as REAL (genuine), FAKE (counterfeit/clone), or UNCERTAIN. "
        "Return a compact JSON with keys: status in [PASS, FAIL, WARNING] and reason (one short sentence).\n\n"
        f"OCR text:\n{ocr_text.strip()}\n"
    )

    try:
        client = genai.Client()  # Reads GEMINI_API_KEY from environment
        resp = client.models.generate_content(model=model, contents=prompt)
        text_out = getattr(resp, "text", None) or getattr(resp, "output_text", None) or ""
        if not text_out:
            # Try candidates shape if present
            candidates = getattr(resp, "candidates", None) or []
            if candidates:
                parts = (candidates[0].get("content") or {}).get("parts") or []
                text_out = "\n".join([p.get("text", "") for p in parts])
        if not text_out:
            return {"error": "Empty response from Gemini SDK"}

        try:
            parsed = json.loads(text_out)
            status = str(parsed.get("status", "WARNING")).upper()
            reason = parsed.get("reason") or "No reason provided."
            if status not in {"PASS", "FAIL", "WARNING"}:
                status = "WARNING"
            return {"status": status, "reason": reason}
        except Exception:
            low = text_out.lower()
            status = "WARNING"
            if any(k in low for k in ["pass", "real", "genuine"]):
                status = "PASS"
            elif any(k in low for k in ["fail", "fake", "counterfeit"]):
                status = "FAIL"
            return {"status": status, "reason": text_out.strip()[:300]}
    except Exception as e:
        msg = str(e)
        # Provide model-specific hinting
        if "unexpected model name format" in msg.lower() or "not found" in msg.lower():
            return {"error": "HTTP 400 Bad request: verify GEMINI_MODEL (e.g., gemini-1.5-flash or gemini-1.5-pro)", "text": msg}
        if "not valid for this api" in msg.lower():
            return {"error": "API key not valid for Generative Language API: create a new key in Google AI Studio", "text": msg}
        return {"error": f"SDK error: {e}", "text": msg}


def classify_genuineness(ocr_text: str) -> dict:
    """Call Gemini API to classify IC genuineness using OCR text only.
    Returns: {"status": "PASS|FAIL|WARNING", "reason": "..."} or {"error": "..."}.
    Implements model fallback on common 400/404 model errors.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY missing"}

    primary = _normalize_model(os.getenv("GEMINI_MODEL") or "gemini-1.5-flash")
    fallbacks = [m for m in ["gemini-1.5-flash", "gemini-1.5-pro"] if m != primary]

    # First, try SDK path if available
    sdk_res = _classify_with_sdk(primary, ocr_text)
    if isinstance(sdk_res, dict):
        if sdk_res.get("error"):
            body = (sdk_res.get("text") or "").lower()
            if ("model" in body) or ("not found" in body) or ("unsupported" in body) or ("unexpected model name format" in body):
                for alt in fallbacks:
                    sdk_res2 = _classify_with_sdk(alt, ocr_text)
                    if isinstance(sdk_res2, dict) and not sdk_res2.get("error"):
                        return sdk_res2
                sdk_res["error"] += " (model fallback attempted)"
            # If SDK errored for non-model reasons, fall back to REST
        else:
            return sdk_res

    # REST path
    payload = _build_payload(ocr_text)
    res = _call_gemini(api_key, primary, payload)
    if res.get("error"):
        body = (res.get("text") or "").lower()
        # Fallback on model-related errors (404 or 400 mentioning model)
        if ("model" in body) or ("not found" in body) or ("unsupported" in body) or ("unexpected model name format" in body):
            for alt in fallbacks:
                res2 = _call_gemini(api_key, alt, payload)
                if not res2.get("error"):
                    return res2
            # If fallbacks failed, keep original error but annotate
            res["error"] += " (model fallback attempted)"
        return res

    return res