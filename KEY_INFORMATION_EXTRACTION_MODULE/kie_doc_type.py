import re

_SUBTYPE_PATTERNS = [
    ("demande", [
        r"\bdemande\b", r"\bdemander\b", r"\bsollicite\b", r"\bprie\b",
        r"\bveuille[zs]\b", r"\bprière\b"
    ]),
    ("transmission", [
        r"\btransmet[s]?\b", r"\bci[- ]joint\b", r"\bci[- ]après\b",
        r"\bvous\s+adresse\b", r"\bfaire\s+parvenir\b"
    ]),
    ("information", [
        r"\bporte[r]?\s+à\s+(votre\s+)?connaissance\b",
        r"\binform\w+\b", r"\bnotif\w+\b", r"\bsignale[r]?\b"
    ]),
]


def detect_doc_type(raw_text: str) -> dict:
    """
    Detect document type and subtype from raw text.
    All documents in this system are lettre_administrative.
    Subtype is inferred from the body language.
    """
    text_lower = raw_text.lower()
    subtype = "autre"

    for name, patterns in _SUBTYPE_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, text_lower):
                subtype = name
                break
        if subtype != "autre":
            break

    return {
        "doc_type": "lettre_administrative",
        "doc_subtype": subtype
    }
