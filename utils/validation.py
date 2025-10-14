from dataclasses import dataclass
from typing import Optional
import os
import re
import requests
from urllib.parse import urlparse
from utils.search_client import google_search_marking_cached
from utils.deepseek_client import classify_genuineness
from utils.gemini_client import classify_genuineness as classify_genuineness_gemini


@dataclass
class ValidationResult:
    status: str  # PASS | FAIL | WARNING
    details: Optional[str] = None
    reference: Optional[str] = None


KNOWN_MARKINGS = {
    # Demo references: part -> list of valid marking substrings
    "ATMEGA328P": ["MEGA328", "MEGA 328P", "ATMEGA328P"],
    "LM7805": ["7805", "LM7805"],
    "NE555": ["NE555", "LM555"],
    "TDA1060A": ["TDA1060A", "TDA 1060 A", "HSH92184 Y", "HSH92184", "4728"],
}


def _local_validation(text: str) -> ValidationResult:
    t = text.upper()
    hits = []
    for part, patterns in KNOWN_MARKINGS.items():
        for p in patterns:
            if p in t:
                hits.append((part, p))
                break

    if not t.strip():
        return ValidationResult(status="WARNING", details="Empty OCR result.")

    if hits:
        parts = ", ".join([h[0] for h in hits])
        return ValidationResult(status="PASS", details=f"Matched known parts: {parts}")
    else:
        return ValidationResult(status="WARNING", details="No local match; consider validating via web or n8n.")


def validate_text(text: str, webhook_url: Optional[str] = None) -> ValidationResult:
    """Validate OCR text either via n8n webhook or local/SerpAPI heuristics."""
    webhook = webhook_url or os.getenv("N8N_WEBHOOK_URL")
    if webhook:
        try:
            resp = requests.post(webhook, json={"ocr_text": text}, timeout=10)
            if resp.ok:
                data = resp.json() if "application/json" in resp.headers.get("Content-Type", "") else {}
                status = (data.get("status") or "WARNING").upper()
                details = data.get("details") or "Validated via n8n workflow."
                reference = data.get("reference")
                return ValidationResult(status=status, details=details, reference=reference)
            else:
                return ValidationResult(status="WARNING", details=f"n8n webhook error: {resp.status_code}")
        except Exception as e:
            return ValidationResult(status="WARNING", details=f"n8n webhook failed: {e}")

    # Fallback to local validation
    return _local_validation(text)


# --- SerpAPI-based validation ---
VENDOR_DOMAINS = {
    "microchip.com",
    "ti.com",
    "st.com",
    "nxp.com",
    "analog.com",
    "infineon.com",
    "onsemi.com",
    "renesas.com",
    "maximintegrated.com",
}


def _validate_via_serpapi(text: str) -> ValidationResult:
    q = f"{text} IC marking genuine datasheet"
    res = google_search_marking_cached(q, num=5)
    if isinstance(res, dict) and res.get("error"):
        return ValidationResult(status="WARNING", details=f"SerpAPI error: {res['error']}")

    organic = res.get("organic_results") or []
    if not organic:
        return ValidationResult(status="WARNING", details="No search results found.", reference="SerpAPI")

    status = "WARNING"
    details_lines = []
    pass_item = None
    fail_item = None
    for item in organic[:5]:
        title = item.get("title") or ""
        link = item.get("link") or item.get("url") or ""
        snippet = item.get("snippet") or ""
        dom = urlparse(link).netloc if link else ""
        details_lines.append(f"- {title} | {link}")
        low = (title + " " + snippet).lower()
        if not pass_item and "datasheet" in low and any(v in dom for v in VENDOR_DOMAINS):
            pass_item = {"title": title, "link": link, "domain": dom}
        if not fail_item and any(k in low for k in ["fake", "counterfeit", "clone"]):
            fail_item = {"title": title, "link": link, "domain": dom, "snippet": snippet}

    if pass_item:
        status = "PASS"
    elif fail_item:
        status = "FAIL"
    else:
        status = "WARNING"

    explainer_lines = []
    if status == "PASS" and pass_item:
        explainer_lines.append(
            f"PASS triggered by vendor datasheet on {pass_item['domain']}: {pass_item['title']} | {pass_item['link']}"
        )
    elif status == "FAIL" and fail_item:
        explainer_lines.append(
            f"FAIL triggered by suspicious keywords in: {fail_item['title']} | {fail_item['link']}"
        )
    else:
        explainer_lines.append("No decisive signal found in top results.")

    # Add a brief summary line explaining why it's real/fake/uncertain
    if status == "PASS" and pass_item:
        summary = f"Summary: REAL — vendor datasheet found on {pass_item['domain']}"
    elif status == "FAIL" and fail_item:
        summary = "Summary: FAKE — counterfeit keywords present in top results"
    else:
        summary = "Summary: UNCERTAIN — no decisive signal in top results"

    details = summary + "\n\nSearch results:\n" + "\n".join(details_lines) + "\n\nExplainer:\n" + "\n".join(explainer_lines)
    return ValidationResult(status=status, details=details, reference="SerpAPI Google Search")


def _validate_via_deepseek(text: str) -> ValidationResult:
    """Validate via DeepSeek LLM, optionally using SerpAPI results for context.
    Produces a brief summary and includes LLM Analysis, Search results, and Explainer when available.
    """
    organic = None
    search_details_lines = []
    explainer_lines = []
    reference = "DeepSeek"

    # If SerpAPI key is present, fetch web context for LLM and build explainer
    if os.getenv("SERPAPI_KEY"):
        q = f"{text} IC marking genuine datasheet"
        res = google_search_marking_cached(q, num=5)
        if isinstance(res, dict) and res.get("error"):
            search_details_lines.append(f"SerpAPI error: {res['error']}")
        else:
            organic = res.get("organic_results") or []
            reference = "DeepSeek + SerpAPI Google Search"
            pass_item = None
            fail_item = None
            for item in organic[:5]:
                title = item.get("title") or ""
                link = item.get("link") or item.get("url") or ""
                snippet = item.get("snippet") or ""
                dom = urlparse(link).netloc if link else ""
                search_details_lines.append(f"- {title} | {link}")
                low = (title + " " + snippet).lower()
                if not pass_item and "datasheet" in low and any(v in dom for v in VENDOR_DOMAINS):
                    pass_item = {"title": title, "link": link, "domain": dom}
                if not fail_item and any(k in low for k in ["fake", "counterfeit", "clone"]):
                    fail_item = {"title": title, "link": link, "domain": dom, "snippet": snippet}

            if pass_item:
                explainer_lines.append(
                    f"PASS indicator: vendor datasheet on {pass_item['domain']}: {pass_item['title']} | {pass_item['link']}"
                )
            elif fail_item:
                explainer_lines.append(
                    f"FAIL indicator: suspicious keywords in: {fail_item['title']} | {fail_item['link']}"
                )
            else:
                explainer_lines.append("No decisive signal found in top results.")

    # Call DeepSeek LLM to classify
    llm = classify_genuineness(text, organic_results=organic)
    if llm.get("error"):
        details = "Summary: UNCERTAIN — LLM error encountered\n\nLLM Analysis:\n" + llm.get("error")
        # Include search details if any
        if search_details_lines:
            details += "\n\nSearch results:\n" + "\n".join(search_details_lines)
        if explainer_lines:
            details += "\n\nExplainer:\n" + "\n".join(explainer_lines)
        return ValidationResult(status="WARNING", details=details, reference=reference)

    llm_status = (llm.get("status") or "WARNING").upper()
    llm_reason = llm.get("reason") or "No reason provided."

    # Build summary from LLM decision
    if llm_status == "PASS":
        summary = f"Summary: REAL — {llm_reason}"
    elif llm_status == "FAIL":
        summary = f"Summary: FAKE — {llm_reason}"
    else:
        summary = f"Summary: UNCERTAIN — {llm_reason}"

    details_parts = [summary, "", "LLM Analysis:", f"Status: {llm_status}", f"Reason: {llm_reason}"]
    if search_details_lines:
        details_parts += ["", "Search results:", *search_details_lines]
    if explainer_lines:
        details_parts += ["", "Explainer:", *explainer_lines]

    details = "\n".join(details_parts)
    return ValidationResult(status=llm_status, details=details, reference=reference)


def _validate_via_gemini(text: str) -> ValidationResult:
    """Validate via Gemini LLM using OCR text only (no web context)."""
    llm = classify_genuineness_gemini(text)
    if llm.get("error"):
        details_lines = [
            "Summary: UNCERTAIN — LLM error encountered",
            "",
            "LLM Analysis:",
            llm.get("error")
        ]
        body = llm.get("text")
        if body:
            details_lines += ["", "Response Body:", body]
        details = "\n".join(details_lines)
        return ValidationResult(status="WARNING", details=details, reference="Gemini")

    llm_status = (llm.get("status") or "WARNING").upper()
    llm_reason = llm.get("reason") or "No reason provided."

    if llm_status == "PASS":
        summary = f"Summary: REAL — {llm_reason}"
    elif llm_status == "FAIL":
        summary = f"Summary: FAKE — {llm_reason}"
    else:
        summary = f"Summary: UNCERTAIN — {llm_reason}"

    details = "\n".join([summary, "", "LLM Analysis:", f"Status: {llm_status}", f"Reason: {llm_reason}"])
    return ValidationResult(status=llm_status, details=details, reference="Gemini")


def validate_text(text: str, webhook_url: Optional[str] = None, mode: Optional[str] = None) -> ValidationResult:
    """Prefer Gemini-only validation; fall back to webhook or local heuristics.
    If Gemini key is set, do NOT use SerpAPI or DeepSeek.
    """
    # Gemini-only path when key present
    if os.getenv("GEMINI_API_KEY"):
        return _validate_via_gemini(text)

    # Optional alternate web validations when Gemini missing
    if os.getenv("DEEPSEEK_API_KEY"):
        return _validate_via_deepseek(text)
    if os.getenv("SERPAPI_KEY"):
        return _validate_via_serpapi(text)

    # Fallbacks when web validation not available
    webhook = webhook_url or os.getenv("N8N_WEBHOOK_URL")
    if webhook:
        try:
            resp = requests.post(webhook, json={"ocr_text": text}, timeout=10)
            if resp.ok:
                data = resp.json() if "application/json" in resp.headers.get("Content-Type", "") else {}
                status = (data.get("status") or "WARNING").upper()
                details = data.get("details") or "Validated via n8n workflow."
                reference = data.get("reference")
                return ValidationResult(status=status, details=details, reference=reference)
            else:
                return ValidationResult(status="WARNING", details=f"n8n webhook error: {resp.status_code}")
        except Exception as e:
            return ValidationResult(status="WARNING", details=f"n8n webhook failed: {e}")

    # Final fallback to local heuristics
    return _local_validation(text)