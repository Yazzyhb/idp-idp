import re
from typing import Optional


def _strip_html(text: str) -> str:
    return re.sub(r'<[^>]+>', '', text)


# =============================================================================
# REGEX PATTERNS
# =============================================================================

_REF_CODE = (
    r"[A-Z]{2,}(?:[/-][A-Z0-9]{2,})*"
    r"(?:"
    r"[/-]N[°º]\s*\d*[/-]\d{4}"
    r"|[/-]N[°º]\s*\d+"
    r"|[/-]\d{4}"
    r")"
)

_REF_HEADER_RE = re.compile(_REF_CODE)
_REF_BODY_RE = re.compile(
    r"(?:Réf|REF)\s*:\s*(" + _REF_CODE + r"(?:\s+du\s+[\d/]+)?)",
    re.IGNORECASE
)

_DATE_RE = re.compile(
    r"Alger\s*,?\s*le\s*[.\-_\s]*\n?\s*([A-Za-z0-9][^\n]+?)(?:\n|$)",
    re.IGNORECASE
)

_RECEIVER_RE = re.compile(
    r"(?:^|\n)\s*[AÀ]\s*\n((?:(?:Messieurs?|Mesdames?|Monsieur)\s+[^\n]+\n?)+)",
    re.IGNORECASE | re.MULTILINE
)

_OBJET_RE = re.compile(r"Objet\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_PJ_RE    = re.compile(r"P\.?\s*J\.?\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)

# Mesdames et Messieurs removed — it matches receiver lines
_BODY_START_RE = re.compile(
    r"(?:^|\n)(?:Monsieur\s*[;:,]|"
    r"Faisant\s+suite|Dans\s+le\s+cadre|Suite\s+\u00e0|"
    r"J['\u2019]ai\s+l['\u2019]honneur|Je\s+vous\s+prie)\s*\n?",
    re.IGNORECASE | re.MULTILINE
)

_BODY_END_RE = re.compile(
    r"(?:Veuillez\s+agréer|Veuillez\s+recevoir|Veuillez\s+trouver|"
    r"Recevez|Je\s+vous\s+prie\s+d['\u2019]agréer|Nous\s+vous\s+prions|"
    r"Dans\s+l.attente|Comptant\s+sur|En\s+vous\s+remerciant)",
    re.IGNORECASE
)

_SENDER_LINES = 4


# =============================================================================
# FIELD EXTRACTORS
# =============================================================================

def _extract_sender(raw_text: str) -> Optional[str]:
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
    sender_parts = []
    for line in lines[:_SENDER_LINES]:
        if re.search(_REF_CODE, line):
            break
        if re.match(r'^[A-Z]{2,}[/-]', line):
            break
        if re.match(r'^/?\d{4}$', line):
            break
        if re.match(r'^REF\s*:', line, re.IGNORECASE):
            break
        sender_parts.append(line)
    return "\n".join(sender_parts).strip() or None


def _extract_ref_header(raw_text: str) -> Optional[str]:
    lines = raw_text.split("\n")
    first_lines = [l.strip() for l in lines[:15]]

    for line in first_lines:
        m = re.search(r"REF\s*:\s*(" + _REF_CODE + r")", line, re.IGNORECASE)
        if m:
            return " ".join(m.group(1).split())

    for line in first_lines:
        if re.match(r"^Réf\s*:", line, re.IGNORECASE):
            continue
        match = re.search(_REF_CODE, line)
        if match:
            return " ".join(match.group(0).split())

    for i in range(len(first_lines) - 1):
        for combo in [
            first_lines[i] + first_lines[i + 1],
            first_lines[i + 1] + first_lines[i]
        ]:
            if re.match(r"^Réf\s*:", combo, re.IGNORECASE):
                continue
            match = re.search(_REF_CODE, combo)
            if match:
                return " ".join(match.group(0).split())

    dept_line = None
    year_line = None
    for line in first_lines:
        if re.match(r"^Réf\s*:", line, re.IGNORECASE):
            continue
        if re.match(
            r'^[A-Z]{2,}(?:[/-][A-Z0-9]{2,})+(?:[/-]N[°º]\s*\d*|[/-])?$', line
        ) and not dept_line:
            dept_line = line.rstrip("/")
        if re.match(r'^/?\d{4}$', line) and not year_line:
            year_line = line.lstrip("/")

    if dept_line and year_line:
        return re.sub(r'\s+', '', dept_line) + "/" + year_line

    return None


def _extract_ref_body(raw_text: str) -> Optional[str]:
    match = _REF_BODY_RE.search(raw_text)
    return match.group(1).strip() if match else None


def _extract_date(raw_text: str) -> Optional[str]:
    match = re.search(
        r"Alger\s*,?\s*le\s*([.\-_\s]{0,20})\n?\s*([A-Za-z0-9][^\n]{3,}?)(?:\n|$)",
        raw_text, re.IGNORECASE
    )
    if not match:
        return None
    candidate = match.group(2).strip()
    date_str = re.sub(r'^[.\-_\s]+', '', candidate).strip()
    if not date_str or len(date_str.replace(" ", "")) < 4:
        return None
    if re.match(r'^[\W\s]+$', date_str):
        return None
    if re.match(r"(?:Messieurs?|Mesdames?|Monsieur)\s+", date_str, re.IGNORECASE):
        return None
    if len(date_str) <= 2:
        return None
    return date_str


def _extract_receiver(raw_text: str) -> Optional[str]:
    match = _RECEIVER_RE.search(raw_text)
    if match:
        lines = [l.strip() for l in match.group(1).split("\n") if l.strip()]
        return "\n".join(lines)

    lines = raw_text.split("\n")
    for i, line in enumerate(lines):
        line = line.strip()
        if re.match(
            r"(?:Messieurs?|Mesdames?|Monsieur)\s+(?:le|les|la|l['\u2019])\s+",
            line, re.IGNORECASE
        ):
            receiver_parts = [line]
            for j in range(i + 1, min(i + 4, len(lines))):
                next_line = lines[j].strip()
                if not next_line:
                    break
                if re.match(
                    r"(?:Objet|Réf|P\.J|Monsieur\s*[;:,]|Alger|DGAP|REF)",
                    next_line, re.IGNORECASE
                ):
                    break
                receiver_parts.append(next_line)
            return "\n".join(receiver_parts)
    return None


def _extract_objet(raw_text: str) -> Optional[str]:
    match = _OBJET_RE.search(raw_text)
    return match.group(1).strip() if match else None


def _extract_pj(raw_text: str) -> Optional[str]:
    match = _PJ_RE.search(raw_text)
    return match.group(1).strip() if match else None


def _extract_body(raw_text: str) -> Optional[str]:
    all_matches = list(_BODY_START_RE.finditer(raw_text))

    if not all_matches:
        m = re.search(r"(?:^|\n)Monsieur\s*,\s*\n", raw_text, re.IGNORECASE | re.MULTILINE)
        if not m:
            return None
        all_matches = [m]

    # pick first match not followed by header-like content
    best_match = None
    for match in all_matches:
        remaining = raw_text[match.end():].strip()
        first_line = remaining.split("\n")[0].strip() if remaining else ""
        if re.match(r"(?:Objet|Réf|P\.J|Messieurs?|Mesdames?)", first_line, re.IGNORECASE):
            continue
        best_match = match
        break

    if not best_match:
        best_match = all_matches[-1]

    # use start() to include the trigger phrase in the body
    remaining = raw_text[best_match.start():].lstrip("\n")
    end_match = _BODY_END_RE.search(remaining)
    body = remaining[:end_match.start()].strip() if end_match else remaining.strip()
    return body or None


def extract_fields(raw_text: str) -> dict:
    clean_text = _strip_html(raw_text)
    return {
        "sender":     _extract_sender(clean_text),
        "ref_header": _extract_ref_header(clean_text),
        "ref_body":   _extract_ref_body(clean_text),
        "date":       _extract_date(clean_text),
        "receiver":   _extract_receiver(clean_text),
        "objet":      _extract_objet(clean_text),
        "pj":         _extract_pj(clean_text),
        "body":       _extract_body(clean_text),
    }
