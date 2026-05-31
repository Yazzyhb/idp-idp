import re
from difflib import SequenceMatcher
from typing import Optional

try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None


def _strip_html(text: str) -> str:
    return re.sub(r'<[^>]+>', '', text)


def _norm_ocr_label(line: str) -> str:
    """
    Lightweight OCR normalization for structural labels.
    Keeps content mostly intact while fixing frequent label confusion.
    """
    s = (line or "").replace("’", "'").strip()
    # Frequent OCR punctuation confusions around labels.
    s = re.sub(r"[|¦]+", ":", s)
    # Only normalize at beginning of structural lines to avoid altering words
    # like "Relations" in sender/receiver values.
    s = re.sub(r"^\s*R[EF]{1,2}\s*[:\-]?\s*", "REF: ", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*R[ée]f\s*[:\-]?\s*", "Réf : ", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*Obje[tl1]\s*[:\-]?\s*", "Objet : ", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*P[\.\s]*[Jj1][\.\s]*[:\-]?\s*", "P.J : ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s{2,}", " ", s)
    return s


def _normalized_lines(raw_text: str) -> list[str]:
    return [_norm_ocr_label(l) for l in raw_text.split("\n")]


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
    r"(?:P\.?\s*J\.?\s*:?\s*|Pi[eè]ces?\s+jointes?\s*:?\s*)(.+?)(?:\n|$)",
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
    # OCR artifacts frequently seen in decorative headers/logos.
    if re.search(r"(?:\+{1,}|%{1,}|N8X|oloX|米)", line):
        return True
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
    lines = [l.strip() for l in _normalized_lines(raw_text) if l.strip()]

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

    # pattern 3: explicit department/service/direction anywhere in header window
    for line in lines[:14]:
        if _is_noise(line) or _is_ref_line(line):
            continue
        if re.search(r"\b(Service|Département|Direction)\b", line, re.IGNORECASE):
            return line

    return None


def _extract_ref_header(raw_text: str) -> Optional[str]:
    lines = _normalized_lines(raw_text)
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
    norm_text = "\n".join(_normalized_lines(raw_text))
    match = _REF_BODY_RE.search(norm_text)
    return match.group(1).strip() if match else None


def _extract_date(raw_text: str) -> Optional[str]:
    norm_text = "\n".join(_normalized_lines(raw_text))
    # pattern 1: "Alger:\n DD/MM/YYYY"
    m = re.search(
        r"Alger\s*:?\s*,?\s*(?:le\s*)?\n\s*(\d{2}[/\-]\d{2}[/\-]\d{4})",
        norm_text, re.IGNORECASE
    )
    if m:
        return m.group(1)

    # pattern 2: DD/MM/YYYY in first 25 lines
    for line in norm_text.split("\n")[:25]:
        m = re.search(r"\b(\d{2}[/\-]\d{2}[/\-]\d{4})\b", line)
        if m:
            return m.group(1)

    # pattern 3: "Alger, le ..." same line
    m = re.search(
        r"Alger\s*,?\s*le\s*([.\-_\s]{0,20})\n?\s*([A-Za-z0-9][^\n]{3,}?)(?:\n|$)",
        norm_text, re.IGNORECASE
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
    lines = [l.strip() for l in _normalized_lines(raw_text)]

    def _clean_receiver(s: str) -> str:
        s = re.sub(r"^[\W_]+", "", (s or "").strip())
        s = re.sub(r"^\bA\s*:\s*", "À: ", s, flags=re.IGNORECASE)
        return s.strip()

    def _valid_receiver_candidate(s: str) -> bool:
        s = (s or "").strip()
        if not s:
            return False
        if _is_noise(s) or _is_ref_line(s) or _is_keyword_line(s):
            return False
        if _URL_RE.search(s) or _FOOTER_RE.search(s):
            return False
        return len(s) > 5

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
            if _valid_receiver_candidate(candidate):
                if sum(c.isalpha() for c in candidate) / len(candidate) > 0.5:
                    return _clean_receiver(candidate)

    # pattern 2: explicit "À l'attention ..." / "À: ..."
    norm_text = "\n".join(lines)
    m = re.search(
        r"(?im)^\s*[AÀ]\s*(?:l['\u2019]attention\s+(?:de|du|des)|:)\s*(.+?)\s*$",
        norm_text,
    )
    if m:
        receiver = _clean_receiver(m.group(1))
        if not _valid_receiver_candidate(receiver):
            receiver = ""
        for line in norm_text[m.end():].split("\n")[:3]:
            line = line.strip()
            if not line: break
            if re.match(r"(?:Objet|Réf|P\.J|Alger|REF)", line, re.IGNORECASE): break
            if _valid_receiver_candidate(line):
                receiver += "\n" + _clean_receiver(line)
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
                if not _valid_receiver_candidate(nxt): break
                parts.append(nxt)
            merged = "\n".join(parts).strip()
            return _clean_receiver(merged) if _valid_receiver_candidate(merged) else None

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
        candidate = _clean_receiver(clean_lines[1])
        if _valid_receiver_candidate(candidate):
            return candidate

    return None


def _extract_objet(raw_text: str) -> Optional[str]:
    norm_text = "\n".join(_normalized_lines(raw_text))
    lines = [l.strip() for l in norm_text.split("\n")]

    obj_idx = None
    first_line = ""
    for i, line in enumerate(lines):
        m = re.match(r"^\s*Objet\s*:?\s*(.*)$", line, re.IGNORECASE)
        if m:
            obj_idx = i
            first_line = (m.group(1) or "").strip()
            break
    if obj_idx is None:
        return None

    # If label is alone/near-empty, read from next non-empty line.
    if not first_line:
        for line in lines[obj_idx + 1 : obj_idx + 4]:
            line = line.strip()
            if line:
                first_line = line
                break

    # collect continuation lines — stop at body-start or structural keywords.
    extra = []
    for line in lines[obj_idx + 1 : obj_idx + 5]:
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

    # Candidate A: direct label-based extraction.
    merged_a = (first_line + (" " + " ".join(extra) if extra else "")).strip()
    merged_a = re.sub(r"\s+", " ", merged_a).strip()
    merged_a = re.sub(r"^\s*Réf\s*:\s*", "", merged_a, flags=re.IGNORECASE).strip()

    # If we captured a tiny trailing fragment, recover from nearby lines.
    if len(merged_a.split()) <= 3:
        tail = []
        scan_lines = lines[max(0, obj_idx - 1) : obj_idx + 6]
        started = False
        for ln in scan_lines:
            ln = ln.strip()
            if not ln:
                if started:
                    break
                continue
            if not started and re.search(r"^Objet\s*:", ln, re.IGNORECASE):
                ln = re.sub(r"^Objet\s*:\s*", "", ln, flags=re.IGNORECASE).strip()
                started = True
            elif not started:
                continue
            if re.match(r"(?:P\.J|Réf|REF|Alger|Monsieur|Messieurs?|Mesdames?)", ln, re.IGNORECASE):
                break
            if re.search(r"(?:Dans\s+le\s+cadre|Faisant\s+suite|Suite\s+[aà])", ln, re.IGNORECASE):
                break
            if ln:
                tail.append(ln)
        if tail:
            merged_a = " ".join(tail).strip()

    # Candidate C: subject line appears BEFORE "Objet:" in some templates.
    prev_parts = []
    for j in range(obj_idx - 1, max(-1, obj_idx - 3), -1):
        if j < 0:
            break
        p = lines[j].strip()
        if not p:
            break
        if re.match(r"(?:REF|Réf|Alger|De\s*:|[AÀ]\s*(?:l['\u2019]attention|:)|P\.J|Copie\s+[aà]\s*:)", p, re.IGNORECASE):
            break
        if _is_noise(p) or _is_ref_line(p) or _is_keyword_line(p):
            break
        prev_parts.append(p)
    prev_parts.reverse()
    prev_subject = " ".join(prev_parts).strip()

    merged_c = ""
    if prev_subject:
        # If line after "Objet:" is a tiny continuation token, append it.
        if first_line and len(first_line.split()) <= 3 and not re.search(r"^(?:Dans\s+le\s+cadre|Faisant\s+suite|Suite\s+[aà])", first_line, re.IGNORECASE):
            merged_c = f"{prev_subject} {first_line}".strip()
        # If line after "Objet:" already starts body, keep pre-label subject only.
        elif first_line and re.search(r"^(?:Dans\s+le\s+cadre|Faisant\s+suite|Suite\s+[aà])", first_line, re.IGNORECASE):
            merged_c = prev_subject

    # Candidate B: first compact sentence-like chunk after "Objet:".
    chunk_lines = []
    for ln in lines[obj_idx + 1 : obj_idx + 6]:
        s = ln.strip()
        if not s:
            if chunk_lines:
                break
            continue
        if re.match(r"(?:P\.J|Réf|REF|Alger|Copie\s+[aà]\s*:)", s, re.IGNORECASE):
            break
        if re.search(r"^(?:Dans\s+le\s+cadre|Faisant\s+suite|Suite\s+[aà]|J['\u2019]ai\s+l['\u2019]honneur)", s, re.IGNORECASE):
            break
        chunk_lines.append(s)
        if s.endswith("."):
            break
    merged_b = re.sub(r"\s+", " ", " ".join(chunk_lines)).strip()

    candidates = [c for c in [merged_a, merged_b, merged_c] if c]
    if not candidates:
        return None

    def _score(c: str) -> tuple[int, int]:
        toks = len(c.split())
        # Prefer medium-length title phrases, avoid very long body-like strings.
        length_penalty = abs(toks - 10)
        body_penalty = 6 if toks > 22 else 0
        return (length_penalty + body_penalty, toks)

    best = sorted(candidates, key=_score)[0]
    return best or None


def _extract_pj(raw_text: str) -> Optional[str]:
    norm_text = "\n".join(_normalized_lines(raw_text))
    lines = [l.strip() for l in norm_text.split("\n")]
    text_low = norm_text.lower()

    # Prefer explicit PJ labels at line start (avoid catching body phrases).
    for i, line in enumerate(lines):
        m = re.match(r"^(?:P\.?\s*J\.?|Pi[eè]ces?\s+jointes?)\s*:?\s*(.*)$", line, re.IGNORECASE)
        if not m:
            continue
        val = (m.group(1) or "").strip()
        if not val and i + 1 < len(lines):
            val = lines[i + 1].strip()
        if val:
            n = re.search(r"\b(\d{1,2})\b", val)
            if n:
                k = int(n.group(1))
                return f"{k:02d} pièce jointe" if k == 1 else f"{k:02d} pièces jointes"
            # Keep short values only; long values are often body spill.
            if len(val.split()) <= 4:
                return val

    # OCR may drop the explicit label; recover "NN pièces jointes" anywhere.
    m2 = re.search(r"\b(\d{1,2})\s*pi[eè]ce?s?\s+jointe?s?\b", norm_text, re.IGNORECASE)
    if m2:
        k = int(m2.group(1))
        return f"{k:02d} pièce jointe" if k == 1 else f"{k:02d} pièces jointes"

    # Spelled-out counts.
    m3 = re.search(r"\b(deux|trois|quatre)\s+pi[eè]ce?s?\s+jointe?s?\b", norm_text, re.IGNORECASE)
    if m3:
        mp = {"deux": 2, "trois": 3, "quatre": 4}
        k = mp[m3.group(1).lower()]
        return f"{k:02d} pièces jointes"

    # Body-level attachment cues: "ci-joint" with explicit count.
    num_map = {
        "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4,
    }
    # e.g. "ci-joint les quatre dossiers", "ci joint 02 pièces"
    m4 = re.search(
        r"ci[\-\s]?joint(?:e|es)?(?:\s+\w+){0,4}\s+(\d{1,2}|un|une|deux|trois|quatre)\s+"
        r"(?:pi[eè]ces?|dossiers?|documents?|fichiers?)",
        text_low,
        re.IGNORECASE,
    )
    if m4:
        tok = m4.group(1).lower()
        k = int(tok) if tok.isdigit() else num_map.get(tok, 1)
        return f"{k:02d} pièce jointe" if k == 1 else f"{k:02d} pièces jointes"

    # Reverse order e.g. "les quatre dossiers ... ci-joint"
    m5 = re.search(
        r"(\d{1,2}|un|une|deux|trois|quatre)\s+"
        r"(?:pi[eè]ces?|dossiers?|documents?|fichiers?)(?:\s+\w+){0,6}\s+ci[\-\s]?joint(?:e|es)?",
        text_low,
        re.IGNORECASE,
    )
    if m5:
        tok = m5.group(1).lower()
        k = int(tok) if tok.isdigit() else num_map.get(tok, 1)
        return f"{k:02d} pièce jointe" if k == 1 else f"{k:02d} pièces jointes"

    # Fallback from "Copie à" block or attachment tables (often on page 2):
    # infer at least one attached element when explicit PJ label is absent.
    if re.search(r"\bCopie\s+[aà]\s*:", norm_text, re.IGNORECASE):
        return "01 pièce jointe"
    if re.search(r"\bannexes?\b|\bpi[eè]ces?\b", norm_text, re.IGNORECASE):
        return "01 pièce jointe"
    return None


def _extract_body(raw_text: str) -> Optional[str]:
    norm_text = "\n".join(_normalized_lines(raw_text))
    all_matches = list(_BODY_START_RE.finditer(norm_text))

    if not all_matches:
        m = re.search(r"(?:^|\n)Monsieur\s*,\s*\n", norm_text, re.IGNORECASE | re.MULTILINE)
        if not m:
            return None
        all_matches = [m]

    best_match = None
    for match in all_matches:
        remaining = norm_text[match.end():].strip()
        first_line = remaining.split("\n")[0].strip() if remaining else ""
        if re.match(r"(?:Objet|Réf|P\.J|Messieurs?|Mesdames?)", first_line, re.IGNORECASE):
            continue
        best_match = match
        break

    if not best_match:
        best_match = all_matches[-1]

    remaining = norm_text[best_match.start():].lstrip("\n")
    end_match = _BODY_END_RE.search(remaining)
    body = remaining[:end_match.start()].strip() if end_match else remaining.strip()
    return body or None


def _crf_like_header_decode(raw_text: str) -> tuple[Optional[str], Optional[str]]:
    """
    CRF-inspired line-sequence decoder for sender/receiver in top document region.
    Not a learned model, but uses transition + emission scoring to reduce regex misses.
    """
    lines = [l.strip() for l in _normalized_lines(raw_text) if l.strip()]
    top = lines[:18]
    if not top:
        return None, None

    sender_idx = None
    receiver_idx = None
    best_sender = -10**9
    best_receiver = -10**9

    for i, line in enumerate(top):
        score = 0
        if _is_noise(line) or _is_ref_line(line) or _is_keyword_line(line):
            score -= 4
        if re.search(r"\b(Service|Département|Direction)\b", line, re.IGNORECASE):
            score += 3
        if re.match(r"De\s*:", line, re.IGNORECASE):
            score += 6
        if i <= 5:
            score += 1
        if score > best_sender:
            best_sender = score
            sender_idx = i

    for i, line in enumerate(top):
        score = 0
        if _is_noise(line) or _is_ref_line(line) or _is_keyword_line(line):
            score -= 4
        if re.search(r"\b(Service|Département|Direction|Monsieur|Mesdames?)\b", line, re.IGNORECASE):
            score += 2
        if sender_idx is not None and i > sender_idx:
            score += 2
        if re.match(r"[AÀ]\s*(?:l['’]attention|:)", line, re.IGNORECASE):
            score += 5
        if score > best_receiver:
            best_receiver = score
            receiver_idx = i

    sender = top[sender_idx] if sender_idx is not None and best_sender > 0 else None
    receiver = top[receiver_idx] if receiver_idx is not None and best_receiver > 0 else None
    if sender and receiver and sender == receiver:
        receiver = None
    return sender, receiver


def _normalize_confidence(value: float | int | None) -> float:
    if value is None:
        return 0.0
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.0
    if conf > 1.0:
        conf /= 100.0
    return max(0.0, min(conf, 1.0))


def _text_similarity(left: str, right: str) -> float:
    left_norm = (left or "").strip().lower()
    right_norm = (right or "").strip().lower()
    if not left_norm or not right_norm:
        return 0.0
    if fuzz is not None:
        return fuzz.partial_ratio(left_norm, right_norm) / 100.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def _strip_html_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _make_candidate(
    field_name: str,
    strategy: str,
    value: Optional[str],
    strategy_score: float,
    page_confidence: float,
    reason: str = "",
) -> Optional[dict]:
    if not value:
        return None
    cleaned = _strip_html_whitespace(str(value))
    if not cleaned:
        return None

    strategy_score = max(0.0, min(float(strategy_score), 1.0))
    page_confidence = _normalize_confidence(page_confidence)
    combined = (0.8 * strategy_score) + (0.2 * page_confidence)

    if field_name == "objet":
        token_count = len(cleaned.split())
        if 3 <= token_count <= 16:
            combined += 0.04
        elif token_count > 24:
            combined -= 0.08
    elif field_name in {"sender", "receiver"}:
        if len(cleaned) > 5:
            combined += 0.02
        if _is_noise(cleaned):
            combined -= 0.12
    elif field_name == "body" and len(cleaned.split()) > 20:
        combined += 0.03

    combined = max(0.0, min(combined, 1.0))
    return {
        "field": field_name,
        "strategy": strategy,
        "value": cleaned,
        "strategy_score": round(strategy_score, 4),
        "page_confidence": round(page_confidence, 4),
        "score": round(combined, 4),
        "reason": reason,
    }


def _pick_best_candidate(candidates: list[dict | None]) -> Optional[dict]:
    valid = [candidate for candidate in candidates if candidate and candidate.get("value")]
    if not valid:
        return None
    return sorted(valid, key=lambda c: (c["score"], c["strategy_score"], len(c["value"])), reverse=True)[0]


def _value_after_label(line: str, label_patterns: list[str]) -> Optional[str]:
    cleaned = (line or "").strip()
    if not cleaned:
        return None
    if ":" in cleaned:
        value = cleaned.split(":", 1)[1].strip()
        return value or None
    for pattern in label_patterns:
        match = re.match(pattern, cleaned, re.IGNORECASE)
        if match:
            value = cleaned[match.end():].strip()
            return value or None
    return None


def _best_fuzzy_label_line(lines: list[str], aliases: list[str], window: int = 18) -> tuple[Optional[int], Optional[str], float]:
    best_idx = None
    best_line = None
    best_score = 0.0
    for idx, line in enumerate(lines[:window]):
        candidate = (line or "").strip()
        if not candidate:
            continue
        score = max((_text_similarity(candidate, alias) for alias in aliases), default=0.0)
        if score > best_score:
            best_score = score
            best_idx = idx
            best_line = candidate
    return best_idx, best_line, best_score


def _extract_sender_regex(raw_text: str) -> Optional[str]:
    lines = [line.strip() for line in _normalized_lines(raw_text) if line.strip()]
    for line in lines[:12]:
        match = re.match(r"(?:De|Source|Exp[eé]diteur)\s*:\s*(.+)", line, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if value and not _is_noise(value):
                return value
    return None


def _extract_sender_fuzzy(raw_text: str) -> Optional[str]:
    lines = [line.strip() for line in _normalized_lines(raw_text) if line.strip()]
    idx, line, score = _best_fuzzy_label_line(lines, ["de", "source", "expediteur"], window=12)
    if line is None or score < 0.68:
        return None
    value = _value_after_label(line, [r"(?:De|Source|Exp[eé]diteur)\s*:\s*"])
    if not value and idx is not None and idx + 1 < len(lines):
        value = lines[idx + 1].strip()
    return value or None


def _extract_receiver_regex(raw_text: str) -> Optional[str]:
    lines = [line.strip() for line in _normalized_lines(raw_text) if line.strip()]
    for line in lines[:15]:
        match = re.match(r"(?:A|À)\s*(?:l['\u2019]attention\s+(?:de|du|des)|:)\s*(.+)", line, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if value and not _is_noise(value):
                return value
    return None


def _extract_receiver_fuzzy(raw_text: str) -> Optional[str]:
    lines = [line.strip() for line in _normalized_lines(raw_text) if line.strip()]
    idx, line, score = _best_fuzzy_label_line(lines, ["a l'attention", "destinataire", "destination", "a"], window=16)
    if line is None or score < 0.66:
        return None
    value = _value_after_label(line, [r"(?:A|À)\s*(?:l['\u2019]attention\s+(?:de|du|des)|:)"])
    if not value and idx is not None and idx + 1 < len(lines):
        value = lines[idx + 1].strip()
    return value or None


def _extract_ref_header_regex(raw_text: str) -> Optional[str]:
    norm_text = "\n".join(_normalized_lines(raw_text))
    match = _REF_BODY_RE.search(norm_text)
    if match:
        return match.group(1).strip()
    for line in norm_text.split("\n")[:25]:
        if re.match(r"^Réf\s*:|^REF\s*:", line, re.IGNORECASE):
            value = re.sub(r"^(?:Réf|REF)\s*:\s*", "", line, flags=re.IGNORECASE).strip()
            if value:
                return value
    return None


def _extract_ref_header_fuzzy(raw_text: str) -> Optional[str]:
    lines = [line.strip() for line in _normalized_lines(raw_text) if line.strip()]
    idx, line, score = _best_fuzzy_label_line(lines, ["ref", "réf", "reference"], window=20)
    if line is None or score < 0.70:
        return None
    value = _value_after_label(line, [r"(?:Réf|REF|Reference)\s*:\s*"])
    if not value and idx is not None and idx + 1 < len(lines):
        value = lines[idx + 1].strip()
    return value or None


def _extract_date_regex(raw_text: str) -> Optional[str]:
    norm_text = "\n".join(_normalized_lines(raw_text))
    match = re.search(r"\b(\d{2}[/\-]\d{2}[/\-]\d{4})\b", norm_text)
    if match:
        return match.group(1)
    match = re.search(r"Alger\s*:?,?\s*(?:le\s*)?\n\s*(\d{2}[/\-]\d{2}[/\-]\d{4})", norm_text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _extract_date_fuzzy(raw_text: str) -> Optional[str]:
    lines = [line.strip() for line in _normalized_lines(raw_text) if line.strip()]
    idx, line, score = _best_fuzzy_label_line(lines, ["date", "alger", "le"], window=18)
    if line is None or score < 0.60:
        return None
    match = re.search(r"\b(\d{2}[/\-]\d{2}[/\-]\d{4})\b", line)
    if match:
        return match.group(1)
    if idx is not None and idx + 1 < len(lines):
        match = re.search(r"\b(\d{2}[/\-]\d{2}[/\-]\d{4})\b", lines[idx + 1])
        if match:
            return match.group(1)
    return None


def _extract_objet_regex(raw_text: str) -> Optional[str]:
    norm_text = "\n".join(_normalized_lines(raw_text))
    match = _OBJET_RE.search(norm_text)
    if match:
        value = match.group(1).strip()
        return value or None
    return None


def _extract_objet_fuzzy(raw_text: str) -> Optional[str]:
    lines = [line.strip() for line in _normalized_lines(raw_text) if line.strip()]
    idx, line, score = _best_fuzzy_label_line(lines, ["objet", "subject"], window=24)
    if line is None or score < 0.72:
        return None
    value = _value_after_label(line, [r"Objet\s*:\s*", r"Subject\s*:\s*"])
    if not value and idx is not None and idx + 1 < len(lines):
        value = lines[idx + 1].strip()
    return value or None


def _extract_pj_regex(raw_text: str) -> Optional[str]:
    norm_text = "\n".join(_normalized_lines(raw_text))
    match = _PJ_RE.search(norm_text)
    if match:
        value = match.group(1).strip()
        return value or None
    return None


def _extract_pj_fuzzy(raw_text: str) -> Optional[str]:
    lines = [line.strip() for line in _normalized_lines(raw_text) if line.strip()]
    idx, line, score = _best_fuzzy_label_line(lines, ["pj", "pieces jointes", "annexes", "copie"], window=24)
    if line is None or score < 0.64:
        return None
    value = _value_after_label(line, [r"(?:P\.?\s*J\.?|Pi[eè]ces?\s+jointes?|Annexes?|Copie\s+[aà])\s*:?"])
    if not value and idx is not None and idx + 1 < len(lines):
        value = lines[idx + 1].strip()
    return value or None


def _extract_body_regex(raw_text: str) -> Optional[str]:
    norm_text = "\n".join(_normalized_lines(raw_text))
    match = _BODY_START_RE.search(norm_text)
    if not match:
        match = re.search(r"(?:^|\n)Monsieur\s*,\s*\n", norm_text, re.IGNORECASE | re.MULTILINE)
        if not match:
            return None

    remaining = norm_text[match.start():].lstrip("\n")
    end_match = _BODY_END_RE.search(remaining)
    body = remaining[:end_match.start()].strip() if end_match else remaining.strip()
    return body or None


def _extract_body_fuzzy(raw_text: str) -> Optional[str]:
    lines = [line.strip() for line in _normalized_lines(raw_text)]
    compact_lines = [line for line in lines if line]
    idx, line, score = _best_fuzzy_label_line(
        compact_lines,
        ["monsieur", "madame", "j'ai l'honneur", "veuillez", "nous vous"],
        window=40,
    )
    if line is None or score < 0.58:
        return None
    body_lines = []
    started = False
    for current in compact_lines[idx:]:
        if not current:
            if started:
                break
            continue
        if not started:
            started = True
        body_lines.append(current)
    body = "\n".join(body_lines).strip()
    end_match = _BODY_END_RE.search(body)
    if end_match:
        body = body[:end_match.start()].strip()
    return body or None


def _extract_sender_detailed(raw_text: str, page_confidence: float) -> tuple[Optional[str], list[dict]]:
    candidates = [
        _make_candidate("sender", "regex", _extract_sender_regex(raw_text), 0.85, page_confidence, "explicit sender label"),
        _make_candidate("sender", "heuristic", _extract_sender(raw_text), 0.78, page_confidence, "header window and cleanup rules"),
        _make_candidate("sender", "rapid_fuzzy", _extract_sender_fuzzy(raw_text), 0.72, page_confidence, "fuzzy label match"),
    ]
    best = _pick_best_candidate(candidates)
    return (best["value"] if best else None, [candidate for candidate in candidates if candidate])


def _extract_receiver_detailed(raw_text: str, page_confidence: float) -> tuple[Optional[str], list[dict]]:
    candidates = [
        _make_candidate("receiver", "regex", _extract_receiver_regex(raw_text), 0.84, page_confidence, "explicit receiver label"),
        _make_candidate("receiver", "heuristic", _extract_receiver(raw_text), 0.77, page_confidence, "contextual header scan"),
        _make_candidate("receiver", "rapid_fuzzy", _extract_receiver_fuzzy(raw_text), 0.70, page_confidence, "fuzzy label match"),
    ]
    best = _pick_best_candidate(candidates)
    return (best["value"] if best else None, [candidate for candidate in candidates if candidate])


def _extract_date_detailed(raw_text: str, page_confidence: float) -> tuple[Optional[str], list[dict]]:
    candidates = [
        _make_candidate("date", "regex", _extract_date_regex(raw_text), 0.82, page_confidence, "explicit date pattern"),
        _make_candidate("date", "heuristic", _extract_date(raw_text), 0.74, page_confidence, "alger/date window heuristics"),
        _make_candidate("date", "rapid_fuzzy", _extract_date_fuzzy(raw_text), 0.62, page_confidence, "fuzzy date label match"),
    ]
    best = _pick_best_candidate(candidates)
    return (best["value"] if best else None, [candidate for candidate in candidates if candidate])


def _extract_ref_header_detailed(raw_text: str, page_confidence: float) -> tuple[Optional[str], list[dict]]:
    candidates = [
        _make_candidate("ref_header", "regex", _extract_ref_header_regex(raw_text), 0.86, page_confidence, "explicit reference pattern"),
        _make_candidate("ref_header", "heuristic", _extract_ref_header(raw_text), 0.80, page_confidence, "reference window scan"),
        _make_candidate("ref_header", "rapid_fuzzy", _extract_ref_header_fuzzy(raw_text), 0.68, page_confidence, "fuzzy label match"),
    ]
    best = _pick_best_candidate(candidates)
    return (best["value"] if best else None, [candidate for candidate in candidates if candidate])


def _extract_objet_detailed(raw_text: str, page_confidence: float) -> tuple[Optional[str], list[dict]]:
    candidates = [
        _make_candidate("objet", "regex", _extract_objet_regex(raw_text), 0.84, page_confidence, "explicit objet label"),
        _make_candidate("objet", "heuristic", _extract_objet(raw_text), 0.79, page_confidence, "multi-candidate subject heuristics"),
        _make_candidate("objet", "rapid_fuzzy", _extract_objet_fuzzy(raw_text), 0.73, page_confidence, "fuzzy label match"),
    ]
    best = _pick_best_candidate(candidates)
    return (best["value"] if best else None, [candidate for candidate in candidates if candidate])


def _extract_pj_detailed(raw_text: str, page_confidence: float) -> tuple[Optional[str], list[dict]]:
    candidates = [
        _make_candidate("pj", "regex", _extract_pj_regex(raw_text), 0.83, page_confidence, "explicit PJ label"),
        _make_candidate("pj", "heuristic", _extract_pj(raw_text), 0.78, page_confidence, "attachment count and spillover heuristics"),
        _make_candidate("pj", "rapid_fuzzy", _extract_pj_fuzzy(raw_text), 0.65, page_confidence, "fuzzy label match"),
    ]
    best = _pick_best_candidate(candidates)
    return (best["value"] if best else None, [candidate for candidate in candidates if candidate])


def _extract_body_detailed(raw_text: str, page_confidence: float) -> tuple[Optional[str], list[dict]]:
    candidates = [
        _make_candidate("body", "regex", _extract_body_regex(raw_text), 0.81, page_confidence, "body start/end regex"),
        _make_candidate("body", "heuristic", _extract_body(raw_text), 0.76, page_confidence, "body window heuristics"),
        _make_candidate("body", "rapid_fuzzy", _extract_body_fuzzy(raw_text), 0.60, page_confidence, "fuzzy greeting/body cue"),
    ]
    best = _pick_best_candidate(candidates)
    return (best["value"] if best else None, [candidate for candidate in candidates if candidate])


def extract_fields_detailed(raw_text: str, page_confidence: float = 0.0) -> dict:
    clean_text = _strip_html(raw_text)
    seq_sender, seq_receiver = _crf_like_header_decode(clean_text)

    fields = {}
    field_details = {}

    sender_value, sender_candidates = _extract_sender_detailed(clean_text, page_confidence)
    if not sender_value:
        sender_value = seq_sender
    fields["sender"] = sender_value
    field_details["sender"] = {"selected": sender_value, "candidates": sender_candidates}

    ref_value, ref_candidates = _extract_ref_header_detailed(clean_text, page_confidence)
    fields["ref_header"] = ref_value
    field_details["ref_header"] = {"selected": ref_value, "candidates": ref_candidates}

    fields["ref_body"] = _extract_ref_body(clean_text)
    field_details["ref_body"] = {"selected": fields["ref_body"], "candidates": []}

    date_value, date_candidates = _extract_date_detailed(clean_text, page_confidence)
    fields["date"] = date_value
    field_details["date"] = {"selected": date_value, "candidates": date_candidates}

    receiver_value, receiver_candidates = _extract_receiver_detailed(clean_text, page_confidence)
    if not receiver_value:
        receiver_value = seq_receiver
    fields["receiver"] = receiver_value
    field_details["receiver"] = {"selected": receiver_value, "candidates": receiver_candidates}

    objet_value, objet_candidates = _extract_objet_detailed(clean_text, page_confidence)
    fields["objet"] = objet_value
    field_details["objet"] = {"selected": objet_value, "candidates": objet_candidates}

    pj_value, pj_candidates = _extract_pj_detailed(clean_text, page_confidence)
    fields["pj"] = pj_value
    field_details["pj"] = {"selected": pj_value, "candidates": pj_candidates}

    body_value, body_candidates = _extract_body_detailed(clean_text, page_confidence)
    fields["body"] = body_value
    field_details["body"] = {"selected": body_value, "candidates": body_candidates}

    return {
        "fields": fields,
        "field_details": field_details,
        "page_confidence": _normalize_confidence(page_confidence),
    }


def extract_fields(raw_text: str, page_confidence: float = 0.0) -> dict:
    return extract_fields_detailed(raw_text, page_confidence=page_confidence)["fields"]
