"""
Full IDP Evaluator
==================
OCR metrics : CER, WER, Levenshtein distance, Text Accuracy, processing time
KIE metrics : Precision, Recall, F1, Exact Match, Field-level Accuracy,
                            IoU (bounding-box quality proxy), processing time

Evaluation scope
----------------
- OCR is evaluated on page 1 only so later-page tables do not pollute the score.
- KIE is evaluated on page 1 only.
- The KIE report compares Regex vs Heuristic vs RapidFuzzy and the
    ensemble winner selected per field.

Output
------
- Per-document console progress
- full_ocr_eval.csv            — one row per document, all OCR metrics
- full_kie_eval.csv            — one row per document × field for the ensemble
- full_kie_approaches.csv      — one row per document × field × approach
- evaluation_reports/          — summary tables and charts
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
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import seaborn as sns
from PIL import Image

ROOT     = Path(__file__).parent
KIE_PATH = ROOT / "KEY_INFORMATION_EXTRACTION_MODULE"
OCR_PATH = ROOT / "OCR_MODULE"
DOCS_DIR = ROOT / "documents"
GT_CSV   = DOCS_DIR / "generated_documents.csv"
REPORT_DIR = ROOT / "evaluation_reports"
TABLE_DIR = REPORT_DIR / "tables"
FIG_DIR = REPORT_DIR / "figures"

sys.path.insert(0, str(KIE_PATH))
sys.path.insert(0, str(OCR_PATH))

# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------

def _load_pipeline():
    spec = importlib.util.spec_from_file_location("ocr_main", OCR_PATH / "main.py")
    ocr_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ocr_mod)

    spec2 = importlib.util.spec_from_file_location("kie_field_extractor", KIE_PATH / "kie_field_extractor.py")
    kie_mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(kie_mod)

    return ocr_mod, kie_mod


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


def _ensure_report_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _mean_std_summary(values: list[float]) -> dict[str, float]:
    clean = [v for v in values if v == v]
    if not clean:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan"), "std": float("nan")}
    mean_v = sum(clean) / len(clean)
    var_v = sum((v - mean_v) ** 2 for v in clean) / len(clean)
    return {
        "mean": mean_v,
        "min": min(clean),
        "max": max(clean),
        "std": var_v ** 0.5,
    }


def _display_value(v, precision: int = 4):
    if v != v:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{precision}f}"
    return v


def _display_df(df: pd.DataFrame, precision: int = 4) -> pd.DataFrame:
    return df.applymap(lambda value: _display_value(value, precision=precision))


def _print_markdown_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(df.to_markdown(index=False, tablefmt="fancy_grid"))


def _save_table_artifacts(df: pd.DataFrame, stem: str) -> None:
    csv_path = TABLE_DIR / f"{stem}.csv"
    tex_path = TABLE_DIR / f"{stem}.tex"
    md_path = TABLE_DIR / f"{stem}.md"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        df.to_latex(tex_path, index=False, float_format="%.4f")
    except Exception:
        pass
    try:
        md_path.write_text(df.to_markdown(index=False, tablefmt="fancy_grid"), encoding="utf-8")
    except Exception:
        pass


def _save_table_png(df: pd.DataFrame, stem: str, title: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(max(10, 1.6 * len(df.columns)), max(2.5, 0.45 * (len(df) + 2))))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    ax.set_title(title, pad=18, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_bar_chart(df: pd.DataFrame, x: str, y: str, stem: str, title: str, ylabel: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(df)), 5))
    sns.barplot(data=df, x=x, y=y, ax=ax, color="#3b82f6")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_heatmap(df: pd.DataFrame, stem: str, title: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(df.columns)), max(4, 0.5 * len(df) + 2)))
    sns.heatmap(df, annot=True, fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=1.0, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _page1_ocr(ocr_mod, pdf_path: Path) -> tuple[str, float, float]:
    t0 = time.perf_counter()
    raw_pages = ocr_mod.load_document(str(pdf_path))
    if not raw_pages:
        raise ValueError(f"No pages found in {pdf_path}")
    first_page = raw_pages[0]
    preprocessed = ocr_mod.preprocess(first_page)
    pil_image = Image.fromarray(cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB))
    ocr_result = ocr_mod._run_ocr(pil_image)
    page_text, page_confidence = ocr_mod._build_text(ocr_result)
    ocr_time = time.perf_counter() - t0
    return page_text, page_confidence, ocr_time


def _field_metric_pack(pred: str, gold: str, field: str) -> dict[str, float]:
    em = calc_exact_match(pred, gold, field)
    prec, rec, f1 = calc_token_prf(pred, gold)
    facc = calc_field_accuracy(pred, gold)
    iou = calc_iou_text(pred, gold)
    return {
        "Exact_Match": em,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "Field_Accuracy": facc,
        "IoU": iou,
    }


def _extract_kie_page1(kie_mod, raw_text: str, page_confidence: float) -> tuple[dict, dict, list[dict], float]:
    t0 = time.perf_counter()
    clean_text = raw_text
    seq_sender, seq_receiver = kie_mod._crf_like_header_decode(clean_text)

    field_rows: dict[str, dict] = {}
    fields: dict[str, Optional[str]] = {}
    approach_rows: list[dict] = []

    helper_specs = [
        ("sender", "Source", "_extract_sender_detailed"),
        ("receiver", "Destination", "_extract_receiver_detailed"),
        ("date", "Date", "_extract_date_detailed"),
        ("ref_header", "Ref", "_extract_ref_header_detailed"),
        ("objet", "Objet", "_extract_objet_detailed"),
        ("pj", "Pj", "_extract_pj_detailed"),
        ("body", "Content", "_extract_body_detailed"),
    ]

    for field_name, gt_col, helper_name in helper_specs:
        helper = getattr(kie_mod, helper_name)
        field_start = time.perf_counter()
        selected, candidates = helper(clean_text, page_confidence)
        field_time = time.perf_counter() - field_start

        if field_name == "sender" and not selected:
            selected = seq_sender
        if field_name == "receiver" and not selected:
            selected = seq_receiver

        selected_strategy = "n/a"
        selected_score = float("nan")
        for candidate in candidates:
            if candidate["value"] == selected:
                selected_strategy = candidate["strategy"]
                selected_score = candidate["score"]
                break
        if field_name in {"sender", "receiver"} and selected in {seq_sender, seq_receiver} and selected_strategy == "n/a":
            selected_strategy = "sequence_fallback"

        fields[field_name] = selected
        field_rows[field_name] = {
            "field": field_name,
            "field_label": gt_col,
            "selected": selected,
            "selected_strategy": selected_strategy,
            "selected_score": selected_score,
            "field_time_s": field_time,
        }

        for candidate in candidates:
            approach_rows.append({
                "field": field_name,
                "field_label": gt_col,
                "strategy": candidate["strategy"],
                "value": candidate["value"],
                "strategy_score": candidate["strategy_score"],
                "page_confidence": candidate["page_confidence"],
                "ensemble_score": candidate["score"],
                "reason": candidate["reason"],
                "field_time_s": field_time,
            })

    total_time = time.perf_counter() - t0
    return fields, field_rows, approach_rows, total_time


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
    ocr_mod, kie_mod = _load_pipeline()
    gt_data = _load_gt()
    _ensure_report_dirs()

    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    if max_docs:
        pdf_files = pdf_files[:max_docs]

    total = len(pdf_files)
    print(f"Found {total} document(s) to evaluate.\n")

    # ── accumulators ────────────────────────────────────────────────────────
    ocr_rows = []
    kie_rows = []
    approach_rows = []

    ocr_acc = defaultdict(list)
    kie_acc = defaultdict(lambda: defaultdict(list))
    approach_acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    approach_overall = defaultdict(lambda: defaultdict(list))
    kie_time_all = []

    strategy_labels = {
        "regex": "Regex",
        "heuristic": "Heuristic",
        "rapid_fuzzy": "RapidFuzzy",
    }
    metric_labels = {
        "Exact_Match": "Exact Match",
        "Field_Accuracy": "Field Accuracy",
    }
    field_labels = [gt_col for gt_col in KIE_FIELDS.values()]

    def _best_strategy(scores: dict[str, float]) -> str:
        valid = {k: v for k, v in scores.items() if v == v}
        if not valid:
            return "n/a"
        return max(valid, key=valid.get)

    def _metric_table(metric: str) -> pd.DataFrame:
        rows = []
        for field_label in field_labels:
            scores = {
                strategy_labels[strategy]: _safe_mean(approach_acc[field_label][strategy][metric])
                for strategy in strategy_labels
            }
            best_key = _best_strategy({strategy: score for strategy, score in scores.items()})
            row = {"Field": field_label}
            row.update(scores)
            row["Best"] = best_key
            rows.append(row)

        overall_scores = {
            strategy_labels[strategy]: _safe_mean(approach_overall[strategy][metric])
            for strategy in strategy_labels
        }
        overall_best = _best_strategy({strategy: score for strategy, score in overall_scores.items()})
        overall_row = {"Field": "Overall"}
        overall_row.update(overall_scores)
        overall_row["Best"] = overall_best
        rows.append(overall_row)
        return pd.DataFrame(rows)

    # ── per-document loop ───────────────────────────────────────────────────
    for idx, pdf_path in enumerate(pdf_files, 1):
        doc_name = pdf_path.name
        m        = re.search(r"doc_(\d+)", doc_name)
        doc_num  = m.group(1) if m else "???"
        gt_row   = gt_data.get(doc_num, {})
        gt_full  = _gt_full_text(gt_row) if gt_row else ""

        print(f"[{idx:3d}/{total}] {doc_name}", end=" … ", flush=True)

        # ── OCR ─────────────────────────────────────────────────────────────
        try:
            ocr_text, ocr_conf, ocr_time = _page1_ocr(ocr_mod, pdf_path)
        except Exception as exc:
            print(f"OCR ERROR: {exc}")
            continue

        if gt_full:
            doc_cer  = calc_cer(ocr_text, gt_full)
            doc_wer  = calc_wer(ocr_text, gt_full)
            doc_lev  = calc_levenshtein(ocr_text, gt_full)
            doc_tacc = calc_text_accuracy(ocr_text, gt_full)
        else:
            doc_cer = doc_wer = doc_lev = doc_tacc = float("nan")

        ocr_row = {
            "Document": doc_name,
            "CER": doc_cer,
            "WER": doc_wer,
            "Edit Distance": doc_lev,
            "OCR Accuracy": doc_tacc,
            "OCR Time_s": round(ocr_time, 3),
        }
        ocr_rows.append(ocr_row)

        if gt_full:
            ocr_acc["CER"].append(doc_cer)
            ocr_acc["WER"].append(doc_wer)
            ocr_acc["Edit Distance"].append(doc_lev)
            ocr_acc["OCR Accuracy"].append(doc_tacc)
        ocr_acc["OCR Time_s"].append(ocr_time)

        # ── KIE ─────────────────────────────────────────────────────────────
        try:
            fields, field_rows, approach_candidates, kie_time = _extract_kie_page1(kie_mod, ocr_text, ocr_conf)
        except Exception as exc:
            print(f"KIE ERROR: {exc}")
            fields, field_rows, approach_candidates, kie_time = {}, {}, [], float("nan")

        for kie_field, gt_col in KIE_FIELDS.items():
            pred = fields.get(kie_field) or ""
            gold = gt_row.get(gt_col, "") or ""

            metrics = _field_metric_pack(pred, gold, kie_field)
            field_meta = field_rows.get(kie_field, {})
            field_time = field_meta.get("field_time_s", float("nan"))

            kie_row = {
                "Document": doc_name,
                "Field": gt_col,
                "Predicted": pred[:120],
                "Ground Truth": gold[:120],
                "Selected Strategy": field_meta.get("selected_strategy", "n/a"),
                "Selected Score": field_meta.get("selected_score", float("nan")),
                "KIE Time_s": round(field_time, 4) if field_time == field_time else float("nan"),
                "Exact Match": metrics["Exact_Match"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1": metrics["F1"],
                "Field Accuracy": metrics["Field_Accuracy"],
                "IoU": metrics["IoU"],
            }
            kie_rows.append(kie_row)

            if gold:
                for metric_name, metric_value in metrics.items():
                    kie_acc[gt_col][metric_labels.get(metric_name, metric_name)].append(metric_value)
                if field_time == field_time:
                    kie_acc[gt_col]["KIE Time_s"].append(field_time)
                    kie_time_all.append(field_time)

        for cand in approach_candidates:
            gt_col = cand["field_label"]
            gold = gt_row.get(gt_col, "") or ""
            metrics = _field_metric_pack(cand["value"] or "", gold, cand["field"])

            approach_row = {
                "Document": doc_name,
                "Field": gt_col,
                "Strategy": strategy_labels.get(cand["strategy"], cand["strategy"]),
                "Value": (cand["value"] or "")[:120],
                "Strategy Score": cand["strategy_score"],
                "Page Confidence": cand["page_confidence"],
                "Ensemble Score": cand["ensemble_score"],
                "Reason": cand["reason"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1": metrics["F1"],
                "Exact Match": metrics["Exact_Match"],
                "Field Accuracy": metrics["Field_Accuracy"],
                "IoU": metrics["IoU"],
            }
            approach_rows.append(approach_row)

            if gold:
                for metric_name, metric_value in metrics.items():
                    label = metric_labels.get(metric_name, metric_name)
                    approach_acc[gt_col][cand["strategy"]][label].append(metric_value)
                    approach_overall[cand["strategy"]][label].append(metric_value)

        # ── per-doc console line ─────────────────────────────────────────────
        cer_s  = f"CER={doc_cer:.3f}" if doc_cer == doc_cer else "CER=n/a"
        wer_s  = f"WER={doc_wer:.3f}" if doc_wer == doc_wer else "WER=n/a"
        tacc_s = f"Acc={doc_tacc:.1%}" if doc_tacc == doc_tacc else "Acc=n/a"
        avg_f1 = _safe_mean([
            _safe_mean(kie_acc[KIE_FIELDS[f]]["F1"]) for f in KIE_FIELDS
            if kie_acc[KIE_FIELDS[f]]["F1"]
        ])
        f1_s = f"KIE-F1={avg_f1:.3f}" if avg_f1 == avg_f1 else "KIE-F1=n/a"
        print(f"{cer_s}  {wer_s}  {tacc_s}  {f1_s}  OCR={ocr_time:.1f}s  KIE={kie_time:.2f}s")

    # ── save CSVs ────────────────────────────────────────────────────────────
    ocr_csv = ROOT / "full_ocr_eval.csv"
    kie_csv = ROOT / "full_kie_eval.csv"
    approach_csv = ROOT / "full_kie_approaches.csv"

    ocr_df = pd.DataFrame(ocr_rows)
    kie_df = pd.DataFrame(kie_rows)
    approach_df = pd.DataFrame(approach_rows)

    if not ocr_df.empty:
        ocr_df = ocr_df[["Document", "CER", "WER", "Edit Distance", "OCR Accuracy", "OCR Time_s"]]
    if not kie_df.empty:
        kie_df = kie_df[["Document", "Field", "Predicted", "Ground Truth", "Selected Strategy", "Selected Score", "KIE Time_s", "Exact Match", "Precision", "Recall", "F1", "Field Accuracy", "IoU"]]
    if not approach_df.empty:
        approach_df = approach_df[["Document", "Field", "Strategy", "Value", "Strategy Score", "Page Confidence", "Ensemble Score", "Reason", "Precision", "Recall", "F1", "Exact Match", "Field Accuracy", "IoU"]]

    with open(ocr_csv, "w", newline="", encoding="utf-8-sig") as f:
        if ocr_rows:
            w = csv.DictWriter(f, fieldnames=ocr_rows[0].keys())
            w.writeheader(); w.writerows(ocr_rows)

    with open(kie_csv, "w", newline="", encoding="utf-8-sig") as f:
        if kie_rows:
            w = csv.DictWriter(f, fieldnames=kie_rows[0].keys())
            w.writeheader(); w.writerows(kie_rows)

    with open(approach_csv, "w", newline="", encoding="utf-8-sig") as f:
        if approach_rows:
            w = csv.DictWriter(f, fieldnames=approach_rows[0].keys())
            w.writeheader(); w.writerows(approach_rows)

    # ── report tables ───────────────────────────────────────────────────────
    ocr_summary_df = pd.DataFrame([
        {"Metric": metric, **_mean_std_summary(ocr_acc[metric])}
        for metric in ["CER", "WER", "Edit Distance", "OCR Accuracy", "OCR Time_s"]
        if ocr_acc[metric]
    ])
    if not ocr_summary_df.empty:
        ocr_summary_df = ocr_summary_df[["Metric", "mean", "min", "max", "std"]]

    kie_field_summary_rows = []
    for field_key, gt_col in KIE_FIELDS.items():
        stats = kie_acc[gt_col]
        kie_field_summary_rows.append({
            "Field": gt_col,
            "Precision": _safe_mean(stats["Precision"]),
            "Recall": _safe_mean(stats["Recall"]),
            "F1": _safe_mean(stats["F1"]),
            "Exact Match": _safe_mean(stats["Exact Match"]),
            "Field Accuracy": _safe_mean(stats["Field Accuracy"]),
            "KIE Time_s": _safe_mean(stats["KIE Time_s"]),
        })
    kie_field_summary_df = pd.DataFrame(kie_field_summary_rows)

    kie_overall_summary_df = pd.DataFrame([
        {"Metric": metric, **_mean_std_summary(sum((kie_acc[gt_col][metric] for gt_col in kie_acc), []))}
        for metric in ["Precision", "Recall", "F1", "Exact Match", "Field Accuracy", "KIE Time_s"]
    ])
    if not kie_overall_summary_df.empty:
        kie_overall_summary_df = kie_overall_summary_df[["Metric", "mean", "min", "max", "std"]]

    comparison_tables = {metric: _metric_table(metric) for metric in ["Precision", "Recall", "F1", "Exact Match", "Field Accuracy"]}

    for name, df in [
        ("ocr_table_1_p1_per_document", ocr_df),
        ("ocr_table_1_p2_overall", ocr_summary_df),
        ("kie_table_2_p1_per_field", kie_field_summary_df),
        ("kie_table_2_p2_overall", kie_overall_summary_df),
    ]:
        if not df.empty:
            _save_table_artifacts(df, name)

    for metric_name, df in comparison_tables.items():
        safe_name = metric_name.lower().replace(" ", "_")
        _save_table_artifacts(df, f"approach_table_3_{safe_name}")
        _save_table_png(df, f"approach_table_3_{safe_name}", f"Table 3 - {metric_name}")

    # ── visualisations ─────────────────────────────────────────────────────
    if not ocr_summary_df.empty:
        ocr_plot_df = ocr_summary_df[ocr_summary_df["Metric"].isin(["CER", "WER", "OCR Accuracy", "OCR Time_s"])].copy()
        _save_bar_chart(ocr_plot_df, "Metric", "mean", "ocr_overall_mean_metrics", "OCR Evaluation - Overall Mean Metrics", "Mean value")
        _save_table_png(ocr_summary_df, "ocr_table_1_p2_overall", "OCR Evaluation - Overall Summary")

    if not kie_field_summary_df.empty:
        _save_bar_chart(kie_field_summary_df, "Field", "F1", "kie_ensemble_f1_by_field", "Ensemble KIE - F1 by Field", "F1")
        _save_table_png(kie_field_summary_df, "kie_table_2_p1_per_field", "Ensemble KIE - Per Field")

    f1_heatmap = pd.DataFrame(index=["Source", "Destination", "Date", "Ref", "Objet", "Pj", "Content"])
    em_heatmap = pd.DataFrame(index=["Source", "Destination", "Date", "Ref", "Objet", "Pj", "Content"])
    for strategy_key, strategy_label in strategy_labels.items():
        f1_heatmap[strategy_label] = [
            _safe_mean(approach_acc[field_label][strategy_key]["F1"]) for field_label in field_labels
        ]
        em_heatmap[strategy_label] = [
            _safe_mean(approach_acc[field_label][strategy_key]["Exact Match"]) for field_label in field_labels
        ]
    _save_heatmap(f1_heatmap, "approach_comparison_f1_heatmap", "Approach Comparison - F1 by Field")
    _save_heatmap(em_heatmap, "approach_comparison_em_heatmap", "Approach Comparison - Exact Match by Field")

    # ── SUMMARY TABLES ───────────────────────────────────────────────────────
    sep = "=" * 70

    print(f"\n{sep}")
    print("  TABLE 1-P1 — OCR Metrics (per document, page 1 only)")
    print(sep)
    if not ocr_df.empty:
        print(_display_df(ocr_df).to_markdown(index=False, tablefmt="fancy_grid"))
    print(f"  Saved → {ocr_csv}")

    print(f"\n{sep}")
    print("  TABLE 1-P2 — OCR Metrics (overall, page 1 only)")
    print(sep)
    if not ocr_summary_df.empty:
        print(_display_df(ocr_summary_df).to_markdown(index=False, tablefmt="fancy_grid"))
    print(f"  Saved → {TABLE_DIR / 'ocr_table_1_p2_overall.csv'}")

    print(f"\n{sep}")
    print("  TABLE 2-P1 — Ensemble KIE Metrics (Approach D, per field averaged across documents)")
    print(sep)
    if not kie_field_summary_df.empty:
        print(_display_df(kie_field_summary_df).to_markdown(index=False, tablefmt="fancy_grid"))
    print(f"  Saved → {TABLE_DIR / 'kie_table_2_p1_per_field.csv'}")

    print(f"\n{sep}")
    print("  TABLE 2-P2 — Ensemble KIE Metrics (Approach D, overall)")
    print(sep)
    if not kie_overall_summary_df.empty:
        print(_display_df(kie_overall_summary_df).to_markdown(index=False, tablefmt="fancy_grid"))
    print(f"  Saved → {TABLE_DIR / 'kie_table_2_p2_overall.csv'}")

    for metric_name in ["F1", "Exact Match", "Precision", "Recall"]:
        df = comparison_tables[metric_name]
        print(f"\n{sep}")
        print(f"  TABLE 3 — Approach Comparison ({metric_name})")
        print(f"  Regex vs Heuristic vs RapidFuzzy")
        print(sep)
        print(_display_df(df).to_markdown(index=False, tablefmt="fancy_grid"))
        print(f"  Saved → {TABLE_DIR / f'approach_table_3_{metric_name.lower().replace(' ', '_')}.csv'}")

    print(f"\n  Saved → {ocr_csv}")
    print(f"  Saved → {kie_csv}")
    print(f"  Saved → {approach_csv}")
    print(f"  Reports → {REPORT_DIR}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Full IDP Evaluator (OCR + KIE)")
    parser.add_argument("--docs", type=int, default=None,
                        help="Limit number of documents (default: all)")
    args = parser.parse_args()
    evaluate(max_docs=args.docs)
