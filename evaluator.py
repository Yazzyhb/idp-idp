# Windows multiprocessing guard — must be FIRST before any other import
import multiprocessing
multiprocessing.freeze_support()
if __name__ == "__main__":
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict, Counter

ROOT     = Path(__file__).parent
KIE_PATH = ROOT / "KEY_INFORMATION_EXTRACTION_MODULE"
OCR_PATH = ROOT / "OCR_MODULE"
DOCS_DIR = ROOT / "documents"
GT_CSV   = DOCS_DIR / "generated_documents.csv"

sys.path.insert(0, str(KIE_PATH))
sys.path.insert(0, str(OCR_PATH))

import importlib.util


# =============================================================================
# PIPELINE LOADER
# =============================================================================

def load_pipeline():
    spec = importlib.util.spec_from_file_location("ocr_main", OCR_PATH / "main.py")
    ocr_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ocr_mod)

    spec2 = importlib.util.spec_from_file_location("kie_extractor", KIE_PATH / "extractor.py")
    kie_mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(kie_mod)

    return ocr_mod.process_document, kie_mod.extract


def load_gt() -> dict:
    out = {}
    with open(GT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = re.search(r"doc_(\d+)", str(row["filename"]))
            if m:
                out[m.group(1)] = row
    return out


def _has_table(gt_row: dict) -> bool:
    return str(gt_row.get("has_table", "")).strip().lower() == "yes"


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def _norm(s: str) -> str:
    """Lowercase, collapse whitespace, strip."""
    return re.sub(r"\s+", " ", str(s or "").lower()).strip()


def _ref_norm(s: str) -> str:
    return re.sub(r"[-\s]+", "/", _norm(s)).strip("/")


def _gt_full_text(gt_row: dict) -> str:
    """
    Reconstruct full document text from GT fields in reading order.
    This matches what the OCR should produce top-to-bottom.
    """
    parts = [
        gt_row.get("Source", ""),
        gt_row.get("Destination", ""),
        gt_row.get("Date", ""),
        gt_row.get("Ref", ""),
        gt_row.get("Objet", ""),
        gt_row.get("Pj", ""),
        gt_row.get("Content", ""),
    ]
    return " ".join(p.strip() for p in parts if p and p.strip())


# =============================================================================
# OCR METRICS — CER / WER
# Uses only the extracted body text vs GT body (Content field)
# Rationale: the raw OCR text includes letterhead, ref, date etc.
# which are not in the GT body. Comparing body-to-body is fair.
# =============================================================================

def _edit_distance(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def cer(hyp: str, ref: str) -> float:
    """Character Error Rate — lower is better, 0 = perfect."""
    h = list(_norm(hyp))
    r = list(_norm(ref))
    if not r:
        return 0.0 if not h else 1.0
    return min(_edit_distance(h, r) / len(r), 1.0)


def wer(hyp: str, ref: str) -> float:
    """Word Error Rate — lower is better, 0 = perfect."""
    h = _norm(hyp).split()
    r = _norm(ref).split()
    if not r:
        return 0.0 if not h else 1.0
    return min(_edit_distance(h, r) / len(r), 1.0)


# =============================================================================
# KIE METRICS — Exact Match / Token F1
# =============================================================================

def exact_match(pred: str, gold: str, field: str = "") -> float:
    """1.0 if normalised strings match, 0.0 otherwise. NaN if no GT."""
    if not gold:
        return float("nan")
    p = _ref_norm(pred) if "ref" in field else _norm(pred)
    g = _ref_norm(gold) if "ref" in field else _norm(gold)
    return 1.0 if p == g else 0.0


def token_f1(pred: str, gold: str) -> tuple[float, float, float]:
    """Token-level precision, recall, F1. Returns (nan,nan,nan) if no GT."""
    if not gold:
        return float("nan"), float("nan"), float("nan")
    p_toks = _norm(pred).split()
    g_toks = _norm(gold).split()
    if not p_toks and not g_toks:
        return 1.0, 1.0, 1.0
    if not p_toks:
        return 0.0, 0.0, 0.0
    if not g_toks:
        return 0.0, 0.0, 0.0
    common   = sum((Counter(p_toks) & Counter(g_toks)).values())
    prec     = common / len(p_toks)
    rec      = common / len(g_toks)
    f1       = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


# =============================================================================
# FIELD MAP
# =============================================================================

KIE_FIELDS = {
    "ref_header": "Ref",
    "date":       "Date",
    "sender":     "Source",
    "receiver":   "Destination",
    "objet":      "Objet",
    "pj":         "Pj",
    "body":       "Content",
}


# =============================================================================
# MAIN EVALUATION LOOP
# =============================================================================

def _safe_mean(vals: list) -> float:
    clean = [v for v in vals if v == v]  # filter NaN
    return sum(clean) / len(clean) if clean else float("nan")


def evaluate(max_docs: int = None, no_tables_only: bool = False):
    print("Loading pipeline…")
    process_document, extract = load_pipeline()
    gt_data = load_gt()

    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    if max_docs:
        pdf_files = pdf_files[:max_docs]

    # accumulators
    ocr_cer_list, ocr_wer_list = [], []
    kie_em   = defaultdict(list)
    kie_f1   = defaultdict(list)
    kie_prec = defaultdict(list)
    kie_rec  = defaultdict(list)
    results_rows = []

    out_path = ROOT / "evaluation_results.csv"
    csv_file  = open(out_path, "w", newline="", encoding="utf-8-sig")
    csv_writer = None  # initialised on first row

    total = len(pdf_files)
    skipped_table_docs = 0
    for idx, pdf_path in enumerate(pdf_files, 1):
        doc_name = pdf_path.name
        m        = re.search(r"doc_(\d+)", doc_name)
        doc_num  = m.group(1) if m else None
        gt_row   = gt_data.get(doc_num, {})

        if no_tables_only and gt_row and _has_table(gt_row):
            skipped_table_docs += 1
            print(f"[{idx:3d}/{total}] {doc_name} … skipped (table present)")
            continue

        print(f"[{idx:3d}/{total}] {doc_name}", end=" … ", flush=True)

        try:
            ocr_out = process_document(str(pdf_path))
            kie_out = extract(ocr_out, doc_id=pdf_path.stem)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        # ── OCR: compare full raw OCR text vs full reconstructed GT text ───
        fields   = kie_out["pages"][0]["fields"] if kie_out.get("pages") else {}
        raw_text = " ".join(p.get("raw_text", "") for p in ocr_out)
        gt_full  = _gt_full_text(gt_row)

        doc_cer = cer(raw_text, gt_full) if gt_full else float("nan")
        doc_wer = wer(raw_text, gt_full) if gt_full else float("nan")
        if gt_full:
            ocr_cer_list.append(doc_cer)
            ocr_wer_list.append(doc_wer)

        # ── KIE: per-field metrics ─────────────────────────────────────────
        row = {
            "doc": doc_name, "doc_num": doc_num,
            "ocr_confidence": round(ocr_out[0].get("confidence", 0), 4),
            "full_cer": round(doc_cer, 4) if gt_full else "",
            "full_wer": round(doc_wer, 4) if gt_full else "",
        }

        for kie_field, gt_col in KIE_FIELDS.items():
            pred = fields.get(kie_field) or ""
            gold = gt_row.get(gt_col, "") or ""
            em   = exact_match(pred, gold, kie_field)
            p, r, f = token_f1(pred, gold)
            kie_em[kie_field].append(em)
            kie_f1[kie_field].append(f)
            kie_prec[kie_field].append(p)
            kie_rec[kie_field].append(r)
            row[f"{kie_field}_em"] = "" if em != em else int(em)
            row[f"{kie_field}_f1"] = "" if f != f  else round(f, 4)

        results_rows.append(row)

        # write row immediately (intermediate save)
        if csv_writer is None:
            csv_writer = csv.DictWriter(csv_file, fieldnames=row.keys())
            csv_writer.writeheader()
        csv_writer.writerow(row)
        csv_file.flush()

        cer_str = f"CER={doc_cer:.3f}  WER={doc_wer:.3f}" if gt_full else "no GT"
        print(cer_str)

    csv_file.close()

    # ── SUMMARY ────────────────────────────────────────────────────────────
    sep = "=" * 62
    print(f"\n{sep}")
    print("OCR METRICS  (full raw OCR text vs full reconstructed GT text)")
    print(sep)
    mean_cer = _safe_mean(ocr_cer_list)
    mean_wer = _safe_mean(ocr_wer_list)
    if ocr_cer_list:
        print(f"  Docs evaluated : {len(ocr_cer_list)}")
        print(f"  Mean full CER  : {mean_cer:.4f}  ({1-mean_cer:.1%} char accuracy)")
        print(f"  Mean full WER  : {mean_wer:.4f}  ({1-mean_wer:.1%} word accuracy)")
    else:
        print("  No data")

    print(f"\n{sep}")
    print("KIE METRICS  (per field)")
    print(sep)
    print(f"  {'Field':<14} {'N':>4} {'ExactMatch':>11} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("  " + "-" * 58)
    macro_f1_vals = []
    for field in KIE_FIELDS:
        n  = len([v for v in kie_em[field] if v == v])
        em = _safe_mean(kie_em[field])
        p  = _safe_mean(kie_prec[field])
        r  = _safe_mean(kie_rec[field])
        f  = _safe_mean(kie_f1[field])
        if f == f:
            macro_f1_vals.append(f)
        em_s = f"{em:.3f}" if em == em else "  n/a"
        p_s  = f"{p:.3f}"  if p  == p  else "  n/a"
        r_s  = f"{r:.3f}"  if r  == r  else "  n/a"
        f_s  = f"{f:.3f}"  if f  == f  else "  n/a"
        print(f"  {field:<14} {n:>4} {em_s:>11} {p_s:>10} {r_s:>8} {f_s:>8}")

    macro_f1 = _safe_mean(macro_f1_vals)
    print("  " + "-" * 58)
    mf_s = f"{macro_f1:.3f}" if macro_f1 == macro_f1 else "n/a"
    print(f"  {'Macro avg':<14} {'':>4} {'':>11} {'':>10} {'':>8} {mf_s:>8}")
    if no_tables_only:
        print(f"\nSkipped table docs : {skipped_table_docs}")
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDP Evaluation Script")
    parser.add_argument("--docs", type=int, default=None,
                        help="Number of documents to evaluate (default: all)")
    parser.add_argument("--no-tables", action="store_true",
                        help="Evaluate OCR metrics only on documents whose GT has_table = No")
    args = parser.parse_args()
    evaluate(max_docs=args.docs, no_tables_only=args.no_tables)
