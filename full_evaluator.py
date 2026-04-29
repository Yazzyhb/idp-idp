"""
Full IDP Evaluator
==================
OCR metrics : CER, WER, Levenshtein distance, Text Accuracy, processing time
KIE metrics : Precision, Recall, F1, Exact Match, Field-level Accuracy,
              IoU (bounding-box quality proxy), processing time

Output
------
- Per-document console progress
- full_ocr_eval.csv   — one row per document, all OCR metrics
- full_kie_eval.csv   — one row per document × field, all KIE metrics
- Summary tables printed at the end
"""

import multiprocessing
multiprocessing.freeze_support()

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import csv
import importlib.util
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT     = Path(__file__).parent
KIE_PATH = ROOT / "KEY_INFORMATION_EXTRACTION_MODULE"
OCR_PATH = ROOT / "OCR_MODULE"
DOCS_DIR = ROOT / "documents"
GT_CSV   = DOCS_DIR / "generated_documents.csv"

sys.path.insert(0, str(KIE_PATH))
sys.path.insert(0, str(OCR_PATH))

# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------

def _load_pipeline():
    spec = importlib.util.spec_from_file_location("ocr_main", OCR_PATH / "main.py")
    ocr_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ocr_mod)
    # skip table extraction for speed (pure OCR+KIE eval)
    ocr_mod.extract_tables = lambda *a, **kw: ([], [])

    spec2 = importlib.util.spec_from_file_location("kie_extractor", KIE_PATH / "extractor.py")
    kie_mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(kie_mod)

    return ocr_mod.process_document, kie_mod.extract


# ---------------------------------------------------------------------------
# Ground-truth loader
# ---------------------------------------------------------------------------

def _load_gt() -> dict:
    gt = {}
    with open(GT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = re.search(r"doc_(\d+)", str(row.get("filename", "")))
            if m:
                gt[m.group(1)] = row
    return gt


def _gt_full_text(row: dict) -> str:
    parts = [
        row.get("Source", ""),
        row.get("Destination", ""),
        row.get("Date", ""),
        row.get("Ref", ""),
        row.get("Objet", ""),
        row.get("Pj", ""),
        row.get("Content", ""),
        row.get("Copie", ""),
        row.get("Table", ""),
    ]
    return " ".join(p.strip() for p in parts if p and str(p).strip())


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").lower()).strip()

def _ref_norm(s: str) -> str:
    return re.sub(r"[-\s]+", "/", _norm(s)).strip("/")


# ---------------------------------------------------------------------------
# OCR metrics
# ---------------------------------------------------------------------------

def _edit_distance(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = tmp
    return dp[n]


def calc_cer(hyp: str, ref: str) -> float:
    h, r = list(_norm(hyp)), list(_norm(ref))
    if not r:
        return 0.0 if not h else 1.0
    return min(_edit_distance(h, r) / len(r), 1.0)


def calc_wer(hyp: str, ref: str) -> float:
    h, r = _norm(hyp).split(), _norm(ref).split()
    if not r:
        return 0.0 if not h else 1.0
    return min(_edit_distance(h, r) / len(r), 1.0)


def calc_levenshtein(hyp: str, ref: str) -> int:
    """Raw character-level Levenshtein distance (not normalised)."""
    return _edit_distance(list(_norm(hyp)), list(_norm(ref)))


def calc_text_accuracy(hyp: str, ref: str) -> float:
    """1 - CER, clamped to [0, 1]."""
    return max(0.0, 1.0 - calc_cer(hyp, ref))


# ---------------------------------------------------------------------------
# KIE metrics
# ---------------------------------------------------------------------------

def calc_exact_match(pred: str, gold: str, field: str = "") -> float:
    if not gold:
        return float("nan")
    p = _ref_norm(pred) if "ref" in field else _norm(pred)
    g = _ref_norm(gold) if "ref" in field else _norm(gold)
    return 1.0 if p == g else 0.0


def calc_token_prf(pred: str, gold: str) -> tuple[float, float, float]:
    """Token-level precision, recall, F1."""
    if not gold:
        return float("nan"), float("nan"), float("nan")
    p_toks = _norm(pred).split()
    g_toks = _norm(gold).split()
    if not p_toks and not g_toks:
        return 1.0, 1.0, 1.0
    if not p_toks or not g_toks:
        return 0.0, 0.0, 0.0
    common = sum((Counter(p_toks) & Counter(g_toks)).values())
    prec   = common / len(p_toks)
    rec    = common / len(g_toks)
    f1     = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def calc_field_accuracy(pred: str, gold: str, threshold: float = 0.5) -> float:
    """
    Field-level accuracy: 1.0 if token F1 >= threshold, else 0.0.
    NaN when no ground truth.
    """
    if not gold:
        return float("nan")
    _, _, f1 = calc_token_prf(pred, gold)
    return 1.0 if (f1 == f1 and f1 >= threshold) else 0.0


def calc_iou_text(pred: str, gold: str) -> float:
    """
    Text-based IoU proxy: |intersection of tokens| / |union of tokens|.
    Used as a bounding-box quality proxy when pixel coords are unavailable.
    NaN when no ground truth.
    """
    if not gold:
        return float("nan")
    p_set = set(_norm(pred).split())
    g_set = set(_norm(gold).split())
    if not p_set and not g_set:
        return 1.0
    inter = len(p_set & g_set)
    union = len(p_set | g_set)
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_mean(vals: list) -> float:
    clean = [v for v in vals if v == v]
    return sum(clean) / len(clean) if clean else float("nan")


def _fmt(v) -> str:
    if v != v:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


# ---------------------------------------------------------------------------
# Field mapping  KIE-key → GT-column
# ---------------------------------------------------------------------------

KIE_FIELDS = {
    "sender":     "Source",
    "receiver":   "Destination",
    "date":       "Date",
    "ref_header": "Ref",
    "objet":      "Objet",
    "pj":         "Pj",
    "body":       "Content",
}


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(max_docs: int | None = None):
    print("Loading pipeline…")
    process_document, extract = _load_pipeline()
    gt_data = _load_gt()

    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    if max_docs:
        pdf_files = pdf_files[:max_docs]

    total = len(pdf_files)
    print(f"Found {total} document(s) to evaluate.\n")

    # ── accumulators ────────────────────────────────────────────────────────
    ocr_rows  = []   # one dict per doc
    kie_rows  = []   # one dict per doc × field

    ocr_acc   = defaultdict(list)   # metric_name → [values]
    kie_acc   = defaultdict(lambda: defaultdict(list))  # field → metric → [values]

    # ── per-document loop ───────────────────────────────────────────────────
    for idx, pdf_path in enumerate(pdf_files, 1):
        doc_name = pdf_path.name
        m        = re.search(r"doc_(\d+)", doc_name)
        doc_num  = m.group(1) if m else "???"
        gt_row   = gt_data.get(doc_num, {})
        gt_full  = _gt_full_text(gt_row) if gt_row else ""

        print(f"[{idx:3d}/{total}] {doc_name}", end=" … ", flush=True)

        # ── OCR ─────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            ocr_out = process_document(str(pdf_path))
        except Exception as exc:
            print(f"OCR ERROR: {exc}")
            continue
        ocr_time = time.perf_counter() - t0

        ocr_text = " ".join(p.get("raw_text", "") for p in ocr_out)
        ocr_conf = sum(p.get("confidence", 0.0) for p in ocr_out) / max(len(ocr_out), 1)

        if gt_full:
            doc_cer  = calc_cer(ocr_text, gt_full)
            doc_wer  = calc_wer(ocr_text, gt_full)
            doc_lev  = calc_levenshtein(ocr_text, gt_full)
            doc_tacc = calc_text_accuracy(ocr_text, gt_full)
        else:
            doc_cer = doc_wer = doc_lev = doc_tacc = float("nan")

        ocr_row = {
            "doc":           doc_name,
            "doc_num":       doc_num,
            "ocr_conf":      round(ocr_conf, 4),
            "ocr_time_s":    round(ocr_time, 3),
            "CER":           _fmt(doc_cer),
            "WER":           _fmt(doc_wer),
            "Levenshtein":   _fmt(doc_lev),
            "Text_Accuracy": _fmt(doc_tacc),
        }
        ocr_rows.append(ocr_row)

        if gt_full:
            ocr_acc["CER"].append(doc_cer)
            ocr_acc["WER"].append(doc_wer)
            ocr_acc["Levenshtein"].append(doc_lev)
            ocr_acc["Text_Accuracy"].append(doc_tacc)
            ocr_acc["ocr_time_s"].append(ocr_time)

        # ── KIE ─────────────────────────────────────────────────────────────
        t1 = time.perf_counter()
        try:
            kie_out = extract(ocr_out, doc_id=pdf_path.stem)
        except Exception as exc:
            print(f"KIE ERROR: {exc}")
            kie_out = {"pages": []}
        kie_time = time.perf_counter() - t1

        fields = kie_out["pages"][0]["fields"] if kie_out.get("pages") else {}

        for kie_field, gt_col in KIE_FIELDS.items():
            pred = fields.get(kie_field) or ""
            gold = gt_row.get(gt_col, "") or ""

            em   = calc_exact_match(pred, gold, kie_field)
            prec, rec, f1 = calc_token_prf(pred, gold)
            facc = calc_field_accuracy(pred, gold)
            iou  = calc_iou_text(pred, gold)

            kie_row = {
                "doc":            doc_name,
                "doc_num":        doc_num,
                "field":          kie_field,
                "predicted":      pred[:120],
                "ground_truth":   gold[:120],
                "kie_time_s":     round(kie_time, 3),
                "Exact_Match":    _fmt(em),
                "Precision":      _fmt(prec),
                "Recall":         _fmt(rec),
                "F1":             _fmt(f1),
                "Field_Accuracy": _fmt(facc),
                "IoU":            _fmt(iou),
            }
            kie_rows.append(kie_row)

            if gold:
                kie_acc[kie_field]["Exact_Match"].append(em)
                kie_acc[kie_field]["Precision"].append(prec)
                kie_acc[kie_field]["Recall"].append(rec)
                kie_acc[kie_field]["F1"].append(f1)
                kie_acc[kie_field]["Field_Accuracy"].append(facc)
                kie_acc[kie_field]["IoU"].append(iou)

        # ── per-doc console line ─────────────────────────────────────────────
        cer_s  = f"CER={doc_cer:.3f}" if doc_cer == doc_cer else "CER=n/a"
        wer_s  = f"WER={doc_wer:.3f}" if doc_wer == doc_wer else "WER=n/a"
        tacc_s = f"Acc={doc_tacc:.1%}" if doc_tacc == doc_tacc else "Acc=n/a"
        avg_f1 = _safe_mean([
            _safe_mean(kie_acc[f]["F1"]) for f in KIE_FIELDS
            if kie_acc[f]["F1"]
        ])
        f1_s = f"KIE-F1={avg_f1:.3f}" if avg_f1 == avg_f1 else "KIE-F1=n/a"
        print(f"{cer_s}  {wer_s}  {tacc_s}  {f1_s}  OCR={ocr_time:.1f}s  KIE={kie_time:.2f}s")

    # ── save CSVs ────────────────────────────────────────────────────────────
    ocr_csv = ROOT / "full_ocr_eval.csv"
    kie_csv = ROOT / "full_kie_eval.csv"

    with open(ocr_csv, "w", newline="", encoding="utf-8-sig") as f:
        if ocr_rows:
            w = csv.DictWriter(f, fieldnames=ocr_rows[0].keys())
            w.writeheader(); w.writerows(ocr_rows)

    with open(kie_csv, "w", newline="", encoding="utf-8-sig") as f:
        if kie_rows:
            w = csv.DictWriter(f, fieldnames=kie_rows[0].keys())
            w.writeheader(); w.writerows(kie_rows)

    # ── SUMMARY TABLES ───────────────────────────────────────────────────────
    sep = "=" * 70

    print(f"\n{sep}")
    print("  OCR EVALUATION SUMMARY")
    print(sep)
    print(f"  {'Metric':<20} {'Mean':>10} {'Min':>10} {'Max':>10}")
    print("  " + "-" * 54)
    for metric in ["CER", "WER", "Levenshtein", "Text_Accuracy", "ocr_time_s"]:
        vals = ocr_acc[metric]
        if not vals:
            continue
        clean = [v for v in vals if v == v]
        if not clean:
            continue
        mean_v = sum(clean) / len(clean)
        print(f"  {metric:<20} {mean_v:>10.4f} {min(clean):>10.4f} {max(clean):>10.4f}")
    print(f"  Docs evaluated: {len(ocr_rows)}")
    print(f"  Saved → {ocr_csv}")

    print(f"\n{sep}")
    print("  KIE EVALUATION SUMMARY  (mean over all documents)")
    print(sep)
    header = f"  {'Field':<14} {'N':>4} {'ExactMatch':>11} {'Precision':>10} {'Recall':>8} {'F1':>8} {'FieldAcc':>9} {'IoU':>8}"
    print(header)
    print("  " + "-" * 76)

    macro = defaultdict(list)
    for field in KIE_FIELDS:
        fa = kie_acc[field]
        n  = len([v for v in fa["F1"] if v == v])
        em = _safe_mean(fa["Exact_Match"])
        pr = _safe_mean(fa["Precision"])
        rc = _safe_mean(fa["Recall"])
        f1 = _safe_mean(fa["F1"])
        ac = _safe_mean(fa["Field_Accuracy"])
        iu = _safe_mean(fa["IoU"])
        for k, v in [("EM", em), ("P", pr), ("R", rc), ("F1", f1), ("Acc", ac), ("IoU", iu)]:
            if v == v:
                macro[k].append(v)
        print(
            f"  {field:<14} {n:>4} "
            f"{_fmt(em):>11} {_fmt(pr):>10} {_fmt(rc):>8} "
            f"{_fmt(f1):>8} {_fmt(ac):>9} {_fmt(iu):>8}"
        )

    print("  " + "-" * 76)
    print(
        f"  {'Macro avg':<14} {'':>4} "
        f"{_fmt(_safe_mean(macro['EM'])):>11} "
        f"{_fmt(_safe_mean(macro['P'])):>10} "
        f"{_fmt(_safe_mean(macro['R'])):>8} "
        f"{_fmt(_safe_mean(macro['F1'])):>8} "
        f"{_fmt(_safe_mean(macro['Acc'])):>9} "
        f"{_fmt(_safe_mean(macro['IoU'])):>8}"
    )
    kie_time_all = [r["kie_time_s"] for r in kie_rows if r["field"] == "body"]
    if kie_time_all:
        print(f"\n  Mean KIE time/doc : {sum(kie_time_all)/len(kie_time_all):.3f}s")
    print(f"  Saved → {kie_csv}\n")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Full IDP Evaluator (OCR + KIE)")
    parser.add_argument("--docs", type=int, default=None,
                        help="Limit number of documents (default: all)")
    args = parser.parse_args()
    evaluate(max_docs=args.docs)
