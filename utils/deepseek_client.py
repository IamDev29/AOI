import os
import json
import requests


# Prefer v1 endpoint; also support legacy path via automatic fallback
DEFAULT_API_URLS = [
    os.getenv("DEEPSEEK_API_URL") or "https://api.deepseek.com/v1/chat/completions",
    "https://api.deepseek.com/chat/completions",
]


def classify_genuineness(ocr_text: str, organic_results: list | None = None) -> dict:
    """Call DeepSeek API to classify IC genuineness using OCR text and optional search results.
    Returns a dict: {"status": "PASS|FAIL|WARNING", "reason": "..."} or {"error": "..."}.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return {"error": "DEEPSEEK_API_KEY missing"}

    context_lines = []
    if isinstance(organic_results, list):
        for item in organic_results[:5]:
            title = item.get("title") or ""
            link = item.get("link") or item.get("url") or ""
            snippet = item.get("snippet") or ""
            context_lines.append(f"- {title} | {link}\n{snippet}")
    context_block = "\n".join(context_lines) if context_lines else "(no web context available)"

    system_prompt = (
        "You are an expert IC authenticity auditor. Classify OCR-marked ICs as REAL (genuine), "
        "FAKE (counterfeit/clone), or UNCERTAIN. Use provided web search snippets when available."
    )
    user_prompt = (
        "OCR text:\n" + ocr_text.strip() + "\n\n" +
        "Web search context (top results):\n" + context_block + "\n\n" +
        "Return a compact JSON with keys: status in [PASS, FAIL, WARNING] and reason (one short sentence)."
    )

    # Allow overriding model via env var; default to a reasoning-capable model
    model = os.getenv("DEEPSEEK_MODEL") or "deepseek-reasoner"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        last_error = None
        # Try each API URL; within each, try the selected model then an alternate if 404
        for base_url in DEFAULT_API_URLS:
            resp = requests.post(base_url, headers=headers, data=json.dumps(payload), timeout=25)
            if resp.status_code == 404:
                alt_model = "deepseek-chat" if model != "deepseek-chat" else "deepseek-reasoner"
                payload_alt = dict(payload)
                payload_alt["model"] = alt_model
                resp_alt = requests.post(base_url, headers=headers, data=json.dumps(payload_alt), timeout=25)
                if resp_alt.ok:
                    data = resp_alt.json()
                    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
                    try:
                        parsed = json.loads(content)
                        status = str(parsed.get("status", "WARNING")).upper()
                        reason = parsed.get("reason") or "No reason provided."
                        if status not in {"PASS", "FAIL", "WARNING"}:
                            status = "WARNING"
                        return {"status": status, "reason": reason}
                    except Exception:
                        low = content.lower()
                        status = "WARNING"
                        if "pass" in low or "real" in low or "genuine" in low:
                            status = "PASS"
                        elif "fail" in low or "fake" in low or "counterfeit" in low:
                            status = "FAIL"
                        return {"status": status, "reason": content.strip()[:300]}
                else:
                    hint = "Not found: verify API endpoint/model (try deepseek-reasoner or deepseek-chat)"
                    last_error = {"error": f"HTTP 404 {hint}", "text": resp_alt.text}
                    continue

            if not resp.ok:
                hint = ""
                if resp.status_code == 401:
                    hint = "Unauthorized: check DEEPSEEK_API_KEY"
                elif resp.status_code == 402:
                    hint = "Payment required: check credits/billing status"
                elif resp.status_code == 429:
                    hint = "Rate limited: slow down requests or check quota"
                elif resp.status_code == 404:
                    hint = "Not found: verify API endpoint/model (try deepseek-reasoner or deepseek-chat)"
                last_error = {"error": f"HTTP {resp.status_code} {hint}".strip(), "text": resp.text}
                continue

            # Success path
            data = resp.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            try:
                parsed = json.loads(content)
                status = str(parsed.get("status", "WARNING")).upper()
                reason = parsed.get("reason") or "No reason provided."
                if status not in {"PASS", "FAIL", "WARNING"}:
                    status = "WARNING"
                return {"status": status, "reason": reason}
            except Exception:
                low = content.lower()
                status = "WARNING"
                if "pass" in low or "real" in low or "genuine" in low:
                    status = "PASS"
                elif "fail" in low or "fake" in low or "counterfeit" in low:
                    status = "FAIL"
                return {"status": status, "reason": content.strip()[:300]}

        return last_error or {"error": "Unknown error"}
    except Exception as e:
        return {"error": f"Network/Client error: {e}"}