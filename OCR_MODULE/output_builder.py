import re

def _strip_html(text: str) -> str:
    return re.sub(r'<[^>]+>', '', text)

def build_page_output(page_number: int, full_text: str, text_confidence: float, tables: list[dict]) -> dict:
    return {
        "page": page_number,
        "raw_text": _strip_html(full_text).strip(),
        "confidence": round(text_confidence, 2),
        "tables": tables
    }
