# c:\Users\Dell\Desktop\idp_system\IDP-V1\eval_viewer.py
import sys, re
from pathlib import Path
import pandas as pd
import streamlit as st
import pypdfium2 as pdfium
import numpy as np
from PIL import Image

st.set_page_config(page_title="IDP Eval Viewer", page_icon="🔬", layout="wide")

ROOT     = Path(__file__).parent
KIE_PATH = ROOT / "KEY_INFORMATION_EXTRACTION_MODULE"
OCR_PATH = ROOT / "OCR_MODULE"
DOCS_DIR = ROOT / "documents"
GT_CSV   = DOCS_DIR / "generated_documents.csv"

sys.path.insert(0, str(KIE_PATH))
sys.path.insert(0, str(OCR_PATH))

import importlib.util

# ── models ────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models… (first time only)")
def load_pipeline():
    spec = importlib.util.spec_from_file_location("ocr_main", OCR_PATH / "main.py")
    ocr_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ocr_mod)
    from layout import get_predictors
    get_predictors()
    spec2 = importlib.util.spec_from_file_location("kie_extractor", KIE_PATH / "extractor.py")
    kie_mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(kie_mod)
    return ocr_mod.process_document, kie_mod.extract

@st.cache_data(show_spinner=False)
def load_gt() -> dict:
    df = pd.read_csv(GT_CSV)
    out = {}
    for _, row in df.iterrows():
        m = re.search(r"doc_(\d+)", str(row["filename"]))
        if m:
            out[m.group(1)] = row.to_dict()
    return out

@st.cache_data(show_spinner="Running pipeline…")
def run_pipeline(doc_path: str):
    process_document, extract = load_pipeline()
    ocr_out = process_document(doc_path)
    kie_out = extract(ocr_out, doc_id=Path(doc_path).stem)
    return ocr_out, kie_out

@st.cache_data(show_spinner=False)
def render_pdf_pages(doc_path: str) -> list:
    """Render each PDF page to a PIL Image at 150 DPI for display."""
    pdf = pdfium.PdfDocument(doc_path)
    pages = []
    for i in range(len(pdf)):
        bm = pdf[i].render(scale=150/72)
        pages.append(bm.to_pil())
    return pages

# ── comparison helpers ────────────────────────────────────────────────────────

def _norm(s) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", str(s).lower().replace("\n", " ")).strip()

def _ref_norm(s) -> str:
    return re.sub(r"[-\s]+", "/", _norm(s)).strip("/")

def _compare(extracted, gt, field: str) -> str:
    """Returns 'exact' | 'partial' | 'miss' | 'no_gt'."""
    e, g = _norm(extracted), _norm(gt)
    if not g:
        return "no_gt"
    if field in ("ref_header", "Ref"):
        e, g = _ref_norm(extracted), _ref_norm(gt)
    if e == g:
        return "exact"
    if g and (g in e or e in g):
        return "partial"
    return "miss"

_STATUS = {
    "exact":   ("✅", "#1a7a1a", "#0d3d0d"),
    "partial": ("🟡", "#7a6000", "#3d3000"),
    "miss":    ("❌", "#7a0000", "#3d0000"),
    "no_gt":   ("⚪", "#555",    "#222"),
}

def _row(label: str, extracted, gt, field: str):
    status = _compare(extracted, gt, field)
    icon, fg, bg = _STATUS[status]
    c1, c2, c3 = st.columns([1.3, 2.8, 2.8])
    c1.markdown(f"**{label}** {icon}")
    for col, val in ((c2, extracted), (c3, gt)):
        col.markdown(
            f"<div style='background:{bg};padding:6px 10px;border-radius:6px;"
            f"color:{fg};font-size:0.85em;white-space:pre-wrap'>"
            f"{str(val or '—')}</div>",
            unsafe_allow_html=True,
        )

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🔬 IDP Evaluation Viewer")
st.caption("Extracted  vs  Ground Truth — side by side")

load_pipeline()          # warm up models
gt_data = load_gt()

pdf_files    = sorted(DOCS_DIR.glob("*.pdf"))
doc_names    = [f.name for f in pdf_files]

with st.sidebar:
    st.header("Document")
    selected = st.selectbox("Select", doc_names)
    process  = st.button("▶ Process", type="primary", use_container_width=True)
    clear    = st.button("🗑️ Clear cache & reprocess", use_container_width=True)
    st.divider()

    m       = re.search(r"doc_(\d+)", selected)
    doc_num = m.group(1) if m else None
    gt_row  = gt_data.get(doc_num, {})

    if gt_row:
        st.success(f"GT: doc_{doc_num}")
        st.caption(f"Template : {gt_row.get('template','?')}")
        st.caption(f"Has table: {gt_row.get('has_table','?')}")
    else:
        st.warning("No ground truth for this doc")

    st.caption(f"{len(pdf_files)} docs  |  {len(gt_data)} GT entries")

if clear:
    st.session_state.clear()
    run_pipeline.clear()
    render_pdf_pages.clear()
    st.rerun()

if process:
    ocr_out, kie_out = run_pipeline(str(DOCS_DIR / selected))
    st.session_state.update(ocr=ocr_out, kie=kie_out, doc=selected, gt=gt_row)

if "ocr" not in st.session_state:
    st.info("👈 Select a document and click ▶ Process")
    st.stop()

ocr_out  = st.session_state["ocr"]
kie_out  = st.session_state["kie"]
gt       = st.session_state["gt"]
doc_name = st.session_state["doc"]
doc_path = str(DOCS_DIR / doc_name)

conf  = ocr_out[0]["confidence"]
color = "green" if conf > 0.9 else "orange" if conf > 0.7 else "red"
st.markdown(
    f"**{doc_name}** — OCR confidence: <span style='color:{color}'>{conf:.0%}</span>",
    unsafe_allow_html=True,
)

tab_cmp, tab_pages, tab_raw, tab_eval, tab_json = st.tabs(
    ["📊 Comparison", "🖼 Document Pages", "📝 Raw OCR", "📈 Metrics", "🔧 JSON"]
)

# ── TAB 1 : Comparison ────────────────────────────────────────────────────────
with tab_cmp:
    for pg in kie_out["pages"]:
        f = pg["fields"]
        if len(kie_out["pages"]) > 1:
            st.subheader(f"Page {pg['page_number']}")

        # header
        h1, h2, h3 = st.columns([1.3, 2.8, 2.8])
        h1.markdown("**Field**")
        h2.markdown("**Extracted**")
        h3.markdown("**Ground Truth**")
        st.divider()

        _row("Ref",      f.get("ref_header"), gt.get("Ref"),         "ref_header")
        _row("Date",     f.get("date"),        gt.get("Date"),        "date")
        _row("Sender",   f.get("sender"),      gt.get("Source"),      "sender")
        _row("Receiver", f.get("receiver"),    gt.get("Destination"), "receiver")
        _row("Objet",    f.get("objet"),       gt.get("Objet"),       "objet")
        _row("P.J",      f.get("pj"),          gt.get("Pj"),          "pj")
        _row("Ref Body", f.get("ref_body"),    None,                  "ref_body")

        st.divider()
        bc1, bc2 = st.columns(2)
        bc1.markdown("**Body — Extracted**")
        bc1.text_area("", f.get("body") or "—", height=220,
                      label_visibility="collapsed", key=f"be_{pg['page_number']}")
        bc2.markdown("**Body — Ground Truth**")
        bc2.text_area("", gt.get("Content") or "—", height=220,
                      label_visibility="collapsed", key=f"bg_{pg['page_number']}")

        # table check
        if pg.get("tables"):
            st.divider()
            st.markdown(f"**Tables detected: {len(pg['tables'])}**  "
                        f"(GT has_table: **{gt.get('has_table','?')}**)")
            for tbl in pg["tables"]:
                data = tbl["data"]
                if data:
                    try:
                        if len(data) > 1:
                            seen, cols = {}, []
                            for c in data[0]:
                                c = str(c) if c else ""
                                seen[c] = seen.get(c, 0) + 1
                                cols.append(f"{c}_{seen[c]}" if seen[c] > 1 else c)
                            df = pd.DataFrame(data[1:], columns=cols)
                        else:
                            df = pd.DataFrame(data)
                        st.dataframe(df, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Table display error: {e}")
                        st.code(str(data[:3]))
        elif gt.get("has_table") == "Yes":
            st.warning("⚠️ GT says table present — none detected")

# ── TAB 2 : Document Pages ────────────────────────────────────────────────────
with tab_pages:
    pages_img = render_pdf_pages(doc_path)
    for i, img in enumerate(pages_img, 1):
        st.markdown(f"**Page {i}**")
        st.image(img, use_column_width=True)

# ── TAB 3 : Raw OCR ───────────────────────────────────────────────────────────
with tab_raw:
    for pg in kie_out["pages"]:
        raw = next((p["raw_text"] for p in ocr_out if p["page"] == pg["page_number"]), "")
        st.subheader(f"Page {pg['page_number']}")
        st.text_area("", raw, height=500,
                     label_visibility="collapsed", key=f"raw_{pg['page_number']}")

# ── TAB 4 : METRICS ───────────────────────────────────────────────────────────
with tab_eval:
    import difflib

    def _edit_distance(a, b):
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n + 1):
                temp = dp[j]
                dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
                prev = temp
        return dp[n]

    def _norm(s):
        return re.sub(r"\s+", " ", str(s or "").lower()).strip()

    def _cer(hyp, ref):
        h, r = list(_norm(hyp)), list(_norm(ref))
        return min(_edit_distance(h, r) / len(r), 1.0) if r else (0.0 if not h else 1.0)

    def _wer(hyp, ref):
        h, r = _norm(hyp).split(), _norm(ref).split()
        return min(_edit_distance(h, r) / len(r), 1.0) if r else (0.0 if not h else 1.0)

    def _token_f1(pred, gold):
        from collections import Counter
        p_toks = _norm(pred).split()
        g_toks = _norm(gold).split()
        if not g_toks: return 1.0, 1.0, 1.0
        if not p_toks: return 0.0, 0.0, 0.0
        common = sum((Counter(p_toks) & Counter(g_toks)).values())
        prec = common / len(p_toks)
        rec  = common / len(g_toks)
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        return prec, rec, f1

    KIE_FIELDS = {
        "ref_header": "Ref", "date": "Date", "sender": "Source",
        "receiver": "Destination", "objet": "Objet", "pj": "Pj", "body": "Content"
    }

    # ── OCR metrics ──────────────────────────────────────────────────────────
    st.subheader("OCR Metrics")
    st.caption("Computed on page 1 raw OCR text vs ground truth body (Content field)")
    raw_p1  = next((p["raw_text"] for p in ocr_out if p["page"] == 1), "")
    gt_body = gt.get("Content", "") or ""
    if gt_body:
        doc_cer = _cer(raw_p1, gt_body)
        doc_wer = _wer(raw_p1, gt_body)
        c1, c2, c3 = st.columns(3)
        c1.metric("CER", f"{doc_cer:.3f}", help="Character Error Rate — lower is better")
        c2.metric("WER", f"{doc_wer:.3f}", help="Word Error Rate — lower is better")
        c3.metric("Char Accuracy", f"{(1-doc_cer):.1%}")
    else:
        st.info("No ground truth body text available for OCR evaluation.")

    st.divider()

    # ── KIE metrics ──────────────────────────────────────────────────────────
    st.subheader("KIE Metrics")
    st.caption("Per-field exact match and token F1 for this document")

    fields = kie_out["pages"][0]["fields"] if kie_out.get("pages") else {}
    rows = []
    f1_vals = []
    for kie_field, gt_col in KIE_FIELDS.items():
        pred = fields.get(kie_field) or ""
        gold = gt.get(gt_col, "") or ""
        em   = int(_norm(pred) == _norm(gold)) if gold else None
        prec, rec, f1 = _token_f1(pred, gold)
        if gold:
            f1_vals.append(f1)
        rows.append({
            "Field":       kie_field,
            "Exact Match": "✅" if em == 1 else ("❌" if em == 0 else "⚪"),
            "Precision":   f"{prec:.3f}" if gold else "—",
            "Recall":      f"{rec:.3f}"  if gold else "—",
            "F1":          f"{f1:.3f}"   if gold else "—",
        })

    macro_f1 = sum(f1_vals) / len(f1_vals) if f1_vals else 0.0
    df_metrics = pd.DataFrame(rows)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
    st.metric("Macro F1", f"{macro_f1:.3f}",
              help="Average token F1 across all fields with ground truth")

# ── TAB 5 : JSON ──────────────────────────────────────────────────────────────
with tab_json:
    st.json({"ocr": ocr_out, "kie": kie_out})
