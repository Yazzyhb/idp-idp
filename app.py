import sys
import json
import csv
import io
import tempfile
import os
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="IDP System", page_icon="📄", layout="wide")

ROOT = Path(__file__).parent
KIE_PATH = ROOT / "KEY_INFORMATION_EXTRACTION_MODULE"
OCR_PATH = ROOT / "OCR_MODULE"

sys.path.insert(0, str(KIE_PATH))
sys.path.insert(0, str(OCR_PATH))

import importlib.util


@st.cache_resource(show_spinner="Loading models... (first time only)")
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


process_document, extract = load_pipeline()


def _build_full_csv(ocr_output: list, kie_output: dict) -> str:
    buf = io.StringIO()
    cols = ["doc_id", "doc_type", "doc_subtype", "page", "ocr_confidence",
            "sender", "receiver", "date", "ref_header", "ref_body",
            "objet", "pj", "body", "signature_detected", "handwritten_ref"]
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for page in kie_output["pages"]:
        f   = page.get("fields", {})
        sig = page.get("signature", {}) or {}
        w.writerow({
            "doc_id":             kie_output.get("document_id", ""),
            "doc_type":           kie_output.get("doc_type", ""),
            "doc_subtype":        kie_output.get("doc_subtype", ""),
            "page":               page["page_number"],
            "ocr_confidence":     f"{page.get('ocr_confidence', 0):.2%}",
            "sender":             (f.get("sender")   or "").replace("\n", " | "),
            "receiver":           (f.get("receiver") or "").replace("\n", " | "),
            "date":               f.get("date")       or "",
            "ref_header":         f.get("ref_header") or "",
            "ref_body":           f.get("ref_body")   or "",
            "objet":              f.get("objet")       or "",
            "pj":                 f.get("pj")          or "",
            "body":               (f.get("body") or "").replace("\n", " "),
            "signature_detected": "yes" if sig.get("detected") else "no",
            "handwritten_ref":    page.get("handwritten_ref_number") or "",
        })
    return buf.getvalue()


def _build_summary_csv(kie_output: dict) -> str:
    buf = io.StringIO()
    w   = csv.DictWriter(buf, fieldnames=["doc_id", "page", "objet", "body"], lineterminator="\n")
    w.writeheader()
    for page in kie_output["pages"]:
        f = page.get("fields", {})
        w.writerow({
            "doc_id": kie_output.get("document_id", ""),
            "page":   page["page_number"],
            "objet":  f.get("objet") or "",
            "body":   (f.get("body") or "").replace("\n", " "),
        })
    return buf.getvalue()


# =============================================================================
# UI
# =============================================================================

st.title("📄 IDP System — Intelligent Document Processing")
st.caption("Algérie Poste | French Administrative Documents")

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("PDF, JPG or PNG", type=["pdf", "jpg", "jpeg", "png"])
    process_btn   = st.button("Process Document", type="primary", use_container_width=True)
    st.divider()
    st.caption("Modules active: OCR + KIE")
    st.caption("Classification & Topic Modeling: coming soon")

# =============================================================================
# PROCESS — store results in session_state so they survive re-runs
# =============================================================================

if uploaded_file and process_btn:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        with st.spinner("Running OCR pipeline..."):
            ocr_output = process_document(tmp_path)
        with st.spinner("Extracting key information..."):
            doc_id     = Path(uploaded_file.name).stem
            kie_output = extract(ocr_output, doc_id=doc_id)
        st.session_state["ocr_output"] = ocr_output
        st.session_state["kie_output"] = kie_output
        st.session_state["doc_stem"]   = doc_id
    finally:
        os.unlink(tmp_path)

# =============================================================================
# RENDER — reads from session_state, persists across all button clicks
# =============================================================================

if "ocr_output" in st.session_state and "kie_output" in st.session_state:
    ocr_output = st.session_state["ocr_output"]
    kie_output = st.session_state["kie_output"]
    doc_stem   = st.session_state["doc_stem"]

    st.success(
        f"✅ Processed {len(ocr_output)} page(s) — "
        f"confidence: {ocr_output[0]['confidence']:.0%}"
    )

    _dl1, _dl2, _dl3 = st.columns(3)
    _dl1.download_button("⬇️ Full JSON",
        data=json.dumps({"ocr": ocr_output, "kie": kie_output}, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"{doc_stem}_idp.json", mime="application/json", use_container_width=True)
    _dl2.download_button("⬇️ Full CSV",
        data=_build_full_csv(ocr_output, kie_output).encode("utf-8-sig"),
        file_name=f"{doc_stem}_full.csv", mime="text/csv", use_container_width=True)
    _dl3.download_button("⬇️ Summary CSV",
        data=_build_summary_csv(kie_output).encode("utf-8-sig"),
        file_name=f"{doc_stem}_summary.csv", mime="text/csv", use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Extracted Fields", "📊 Tables", "📝 Raw Text", "🔧 Full JSON"
    ])

    for page_data in kie_output["pages"]:
        page_num = page_data["page_number"]
        fields   = page_data["fields"]
        tables   = page_data["tables"]
        raw_text = next((p["raw_text"] for p in ocr_output if p["page"] == page_num), "")

        with tab1:
            st.subheader(f"Page {page_num}")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Document Type**")
                st.info(f"{kie_output['doc_type']} — {kie_output['doc_subtype']}")
                st.markdown("**Sender**");   st.info(fields.get("sender")   or "—")
                st.markdown("**Receiver**"); st.info(fields.get("receiver") or "—")
                st.markdown("**Date**");     st.info(fields.get("date")     or "*(blank/handwritten)*")

            with col2:
                st.markdown("**Reference (Header)**"); st.info(fields.get("ref_header") or "—")
                st.markdown("**Reference (Body)**");   st.info(fields.get("ref_body")   or "—")
                st.markdown("**P.J**");                st.info(fields.get("pj")          or "—")
                conf  = page_data["ocr_confidence"]
                color = "green" if conf > 0.9 else "orange" if conf > 0.7 else "red"
                st.markdown("**OCR Confidence**")
                st.markdown(
                    f"<span style='color:{color}; font-size:1.2em'>**{conf:.0%}**</span>",
                    unsafe_allow_html=True
                )

            st.markdown("**Object**")
            if fields.get("objet"):
                st.success(fields["objet"])
            else:
                st.warning("Not detected")

            st.markdown("**Body**")
            if fields.get("body"):
                st.text_area("body", value=fields["body"], height=200,
                             label_visibility="collapsed", key=f"body_{page_num}")
            else:
                st.warning("Not detected")

        with tab2:
            st.subheader(f"Page {page_num}")
            if not tables:
                st.info("No tables detected on this page.")
            else:
                for table in tables:
                    import pandas as pd
                    st.markdown(
                        f"**Table {table['table_index']}** — "
                        f"{table['shape'][0]} rows × {table['shape'][1]} cols — "
                        f"confidence: {table['confidence']:.0%} — "
                        f"method: `{table['method']}`"
                    )
                    data = table["data"]
                    if data:
                        df = pd.DataFrame(
                            data[1:], columns=data[0]
                        ) if len(data) > 1 else pd.DataFrame(data)
                        st.dataframe(df, use_container_width=True)
                    with st.expander("View Markdown"):
                        st.code(table.get("markdown", ""), language="markdown")
                    st.divider()

        with tab3:
            st.subheader(f"Page {page_num} — Raw OCR Text")
            st.text_area("raw", value=raw_text, height=400,
                         label_visibility="collapsed", key=f"raw_{page_num}")

        with tab4:
            st.subheader("Full Pipeline Output (JSON)")
            st.json({"ocr": ocr_output, "kie": kie_output})

elif not uploaded_file:
    st.info("👈 Upload a document from the sidebar to get started.")
