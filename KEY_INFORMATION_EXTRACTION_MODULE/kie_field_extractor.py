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
    r"|[/-]\d{4}[/-]\d+"
    r"|[/-]\d{4}"
    r")"
)

_REF_BODY_RE = re.compile(
    r"(?:Réf|REF)\s*:\s*(" + _REF_CODE + r"(?:\s+du\s+[\d/]+)?)",
    re.IGNORECASE
)

_OBJET_RE = re.compile(r"Objet\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_PJ_RE    = re.compile(
    r"(?:P\.?\s*J\.?\s*:|Pi[eè]ces?\s+jointes?\s*:)\s*(.+?)(?:\n|$)",
    re.IGNORECASE
)

_BODY_START_RE = re.compile(
    r"(?:^|\n)(?:Monsieur\s*[;:,]|"
    r"Faisant\s+suite|Dans\s+le\s+cadre|Suite\s+\u00e0|"
    r"J['\u2019]ai\s+l['\u2019]honneur|Je\s+vous\s+prie|"
    r"Nous\s+avons\s+l['\u2019]honneur|Conform[eé]ment|"
    r"Suite\s+aux|Suite\s+à|"
    r"Nous\s+avons\s+constat[eé]|Nous\s+vous\s+sollicitons|"
    r"Nous\s+vous\s+informons|Nous\s+vous\s+transmettons|"
    r"Le\s+D[eé]partement|Nous\s+avons\s+proc[eé]d[eé]|"
    r"Suite\s+aux\s+r[eé]centes|Nous\s+vous\s+adressons)\s*\n?",
    re.IGNORECASE | re.MULTILINE
)

# NOTE: "Nous vous prions" removed — it appears inside body text too
_BODY_END_RE = re.compile(
    r"(?:Veuillez\s+agréer|Veuillez\s+recevoir|Veuillez\s+trouver|"
    r"Recevez|Je\s+vous\s+prie\s+d['\u2019]agréer|"
    r"Dans\s+l.attente|Comptant\s+sur|En\s+vous\s+remerciant|"
    r"Copie\s+[aà]\s*:|Direction\s+G[eé]n[eé]rale\s*,\s*Service|"
    r"Direction\s+des\s+Op[eé]rations\s*$|"
    r"D[eé]partement\s+Risques\s*,)",
    re.IGNORECASE
)

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_URL_RE    = re.compile(r"(?:www\.|http|\.dz|\.com|\.org)", re.IGNORECASE)
_NOISE_RE  = re.compile(r"^[^A-Za-zÀ-ÿ]{0,3}[^A-Za-zÀ-ÿ\s]{3,}")
_FOOTER_RE = re.compile(r"(?:Quartier|Tél\s*:|Bab\s+Ezzouar|BP\s*\d)", re.IGNORECASE)


def _is_noise(line: str) -> bool:
    if _ARABIC_RE.search(line): return True
    if _URL_RE.search(line): return True
    if _NOISE_RE.search(line): return True
    if _FOOTER_RE.search(line): return True
    if sum(c.isalpha() for c in line) / max(len(line), 1) < 0.4: return True
    if len(line) < 3: return True
    return False


def _is_ref_line(line: str) -> bool:
    return bool(re.search(_REF_CODE, line)) or bool(re.match(r'^/?REF\s*:?$', line, re.IGNORECASE))


def _is_keyword_line(line: str) -> bool:
    return bool(re.match(
        r"(?:Alger|REF|Réf|Objet|P\.J|Copie|www\.|http)",
        line, re.IGNORECASE
    ))


# =============================================================================
# FIELD EXTRACTORS
# =============================================================================

def _extract_sender(raw_text: str) -> Optional[str]:
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]

    # pattern 1: explicit "De: <sender>" label
    for line in lines[:12]:
        m = re.match(r"De\s*:\s*(.+)", line, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if not _is_noise(val) and not _is_ref_line(val):
                return val

    # pattern 2: first clean non-noise line before ref/date block
    for line in lines[:8]:
        if _is_ref_line(line): break
        if _is_keyword_line(line): break
        if _is_noise(line): continue
        return line

    return None


def _extract_ref_header(raw_text: str) -> Optional[str]:
    lines = raw_text.split("\n")
    first_lines = [l.strip() for l in lines[:25]]

    # pass 1: "REF:" alone on line, value on next
    for i, line in enumerate(first_lines[:-1]):
        if re.match(r"^REF\s*:?\s*$", line, re.IGNORECASE):
            m = re.search(_REF_CODE, first_lines[i + 1])
            if m:
                return " ".join(m.group(0).split())

    # pass 2: "REF: value" on same line
    for line in first_lines:
        m = re.search(r"REF\s*:\s*(" + _REF_CODE + r")", line, re.IGNORECASE)
        if m:
            return " ".join(m.group(1).split())

    # pass 3: ref code standalone
    for line in first_lines:
        if re.match(r"^Réf\s*:", line, re.IGNORECASE):
            continue
        m = re.search(_REF_CODE, line)
        if m:
            return " ".join(m.group(0).split())

    # pass 4: split across two lines
    for i in range(len(first_lines) - 1):
        for combo in [first_lines[i] + first_lines[i+1],
                      first_lines[i+1] + first_lines[i]]:
            if re.match(r"^Réf\s*:", combo, re.IGNORECASE):
                continue
            m = re.search(_REF_CODE, combo)
            if m:
                return " ".join(m.group(0).split())

    return None


def _extract_ref_body(raw_text: str) -> Optional[str]:
    match = _REF_BODY_RE.search(raw_text)
    return match.group(1).strip() if match else None


def _extract_date(raw_text: str) -> Optional[str]:
    # pattern 1: "Alger:\n DD/MM/YYYY"
    m = re.search(
        r"Alger\s*:?\s*,?\s*(?:le\s*)?\n\s*(\d{2}[/\-]\d{2}[/\-]\d{4})",
        raw_text, re.IGNORECASE
    )
    if m:
        return m.group(1)

    # pattern 2: DD/MM/YYYY in first 25 lines
    for line in raw_text.split("\n")[:25]:
        m = re.search(r"\b(\d{2}[/\-]\d{2}[/\-]\d{4})\b", line)
        if m:
            return m.group(1)

    # pattern 3: "Alger, le ..." same line
    m = re.search(
        r"Alger\s*,?\s*le\s*([.\-_\s]{0,20})\n?\s*([A-Za-z0-9][^\n]{3,}?)(?:\n|$)",
        raw_text, re.IGNORECASE
    )
    if not m:
        return None
    candidate = re.sub(r'^[.\-_\s]+', '', m.group(2)).strip()
    if (not candidate or len(candidate.replace(" ", "")) < 4
            or re.match(r'^[\W\s]+$', candidate)
            or re.match(r"(?:Messieurs?|Mesdames?|Monsieur)\s+", candidate, re.IGNORECASE)):
        return None
    return candidate


def _extract_receiver(raw_text: str) -> Optional[str]:
    lines = [l.strip() for l in raw_text.split("\n")]

    # Find the "De:" sender line index
    de_idx = None
    for i, line in enumerate(lines[:15]):
        if re.match(r"De\s*:", line, re.IGNORECASE):
            de_idx = i
            break

    if de_idx is not None:
        # Look for receiver: skip lines that are ref codes, noise, or keywords
        # Receiver is the first clean department/service name after De: that is NOT a ref
        for j in range(de_idx + 1, min(de_idx + 6, len(lines))):
            candidate = lines[j]
            if not candidate:
                continue
            if _is_noise(candidate):
                continue
            if _is_keyword_line(candidate):
                continue
            if _is_ref_line(candidate):
                continue
            # must look like a department/service name (has letters, reasonable length)
            if len(candidate) > 5 and sum(c.isalpha() for c in candidate) / len(candidate) > 0.5:
                return candidate

    # pattern 2: "À l'attention de ..." / "À: ..."
    m = re.search(
        r"[AÀ]\s*(?:l['\u2019]attention\s+(?:de|du|des)|:)\s*(.+?)(?:\n|$)",
        raw_text, re.IGNORECASE
    )
    if m:
        receiver = m.group(1).strip()
        for line in raw_text[m.end():].split("\n")[:3]:
            line = line.strip()
            if not line: break
            if re.match(r"(?:Objet|Réf|P\.J|Alger|REF)", line, re.IGNORECASE): break
            receiver += "\n" + line
        return receiver.strip() or None

    # pattern 3: "Messieurs le / Monsieur le ..."
    for i, line in enumerate(lines):
        if re.match(
            r"(?:Messieurs?|Mesdames?|Monsieur)\s+(?:le|les|la|l['\u2019])\s+",
            line, re.IGNORECASE
        ):
            parts = [line]
            for j in range(i + 1, min(i + 4, len(lines))):
                nxt = lines[j]
                if not nxt: break
                if re.match(r"(?:Objet|Réf|P\.J|Alger|REF)", nxt, re.IGNORECASE): break
                parts.append(nxt)
            return "\n".join(parts)

    # pattern 4: no De: line — second clean line before ref block (template 2)
    clean_lines = []
    for line in lines[:12]:
        if _is_ref_line(line): break
        if _is_keyword_line(line): break
        if _is_noise(line): continue
        clean_lines.append(line)
        if len(clean_lines) == 2:
            break
    if len(clean_lines) >= 2:
        return clean_lines[1]

    return None


def _extract_objet(raw_text: str) -> Optional[str]:
    match = _OBJET_RE.search(raw_text)
    if not match:
        return None
    first_line = match.group(1).strip()

    # collect continuation lines — stop at body-start or structural keywords
    pos = match.end()
    extra = []
    for line in raw_text[pos:].split("\n")[:4]:
        line = line.strip()
        if not line:
            break
        # stop at any body-start trigger or structural keyword
        if re.match(
            r"(?:P\.J|Réf|REF|Alger|Monsieur|Messieurs?|Mesdames?|"
            r"Dans\s+le\s+cadre|Faisant\s+suite|Suite\s+[aà]|"
            r"J['\u2019]ai\s+l['\u2019]honneur|Nous\s+vous|Le\s+D[eé]partement|"
            r"Nous\s+avons|Conform[eé]ment|Suite\s+aux)",
            line, re.IGNORECASE
        ):
            break
        extra.append(line)

    return (first_line + (" " + " ".join(extra) if extra else "")).strip() or None


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
