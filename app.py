import sys
import json
import tempfile
import os
from pathlib import Path

import streamlit as st

# ── page config MUST be first ─────────────────────────────────────────────────
st.set_page_config(
    page_title="IDP System",
    page_icon="📄",
    layout="wide"
)

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
KIE_PATH = ROOT / "KEY_INFORMATION_EXTRACTION_MODULE"
OCR_PATH = ROOT / "OCR_MODULE"

sys.path.insert(0, str(KIE_PATH))
sys.path.insert(0, str(OCR_PATH))

import importlib.util


@st.cache_resource(show_spinner="Loading models... (first time only)")
def load_pipeline():
    """Load all models once and cache them."""
    spec = importlib.util.spec_from_file_location("ocr_main", OCR_PATH / "main.py")
    ocr_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ocr_mod)

    # pre-warm surya models
    from layout import get_predictors
    get_predictors()

    spec2 = importlib.util.spec_from_file_location("kie_extractor", KIE_PATH / "extractor.py")
    kie_mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(kie_mod)

    return ocr_mod.process_document, kie_mod.extract


# load pipeline (cached after first run)
process_document, extract = load_pipeline()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📄 IDP System — Intelligent Document Processing")
st.caption("Algérie Poste | French Administrative Documents")

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "PDF, JPG or PNG",
        type=["pdf", "jpg", "jpeg", "png"]
    )
    process_btn = st.button(
        "Process Document", type="primary", use_container_width=True
    )
    st.divider()
    st.caption("Modules active: OCR + KIE")
    st.caption("Classification & Topic Modeling: coming soon")

if uploaded_file and process_btn:
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=Path(uploaded_file.name).suffix
    ) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        with st.spinner("Running OCR pipeline..."):
            ocr_output = process_document(tmp_path)

        with st.spinner("Extracting key information..."):
            doc_id = Path(uploaded_file.name).stem
            kie_output = extract(ocr_output, doc_id=doc_id)

        st.success(
            f"✅ Processed {len(ocr_output)} page(s) — "
            f"confidence: {ocr_output[0]['confidence']:.0%}"
        )

        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Extracted Fields",
            "📊 Tables",
            "📝 Raw Text",
            "🔧 Full JSON"
        ])

        for page_data in kie_output["pages"]:
            page_num = page_data["page_number"]
            fields = page_data["fields"]
            tables = page_data["tables"]
            raw_text = next(
                (p["raw_text"] for p in ocr_output if p["page"] == page_num), ""
            )

            with tab1:
                st.subheader(f"Page {page_num}")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Document Type**")
                    st.info(f"{kie_output['doc_type']} — {kie_output['doc_subtype']}")
                    st.markdown("**Sender**")
                    st.info(fields.get("sender") or "—")
                    st.markdown("**Receiver**")
                    st.info(fields.get("receiver") or "—")
                    st.markdown("**Date**")
                    st.info(fields.get("date") or "*(blank/handwritten)*")

                with col2:
                    st.markdown("**Reference (Header)**")
                    st.info(fields.get("ref_header") or "—")
                    st.markdown("**Reference (Body)**")
                    st.info(fields.get("ref_body") or "—")
                    st.markdown("**P.J**")
                    st.info(fields.get("pj") or "—")
                    conf = page_data["ocr_confidence"]
                    color = "green" if conf > 0.9 else "orange" if conf > 0.7 else "red"
                    st.markdown("**OCR Confidence**")
                    st.markdown(
                        f"<span style='color:{color}; font-size:1.2em'>"
                        f"**{conf:.0%}**</span>",
                        unsafe_allow_html=True
                    )

                st.markdown("**Object**")
                if fields.get("objet"):
                    st.success(fields["objet"])
                else:
                    st.warning("Not detected")

                st.markdown("**Body**")
                if fields.get("body"):
                    st.text_area(
                        "body",
                        value=fields["body"],
                        height=200,
                        label_visibility="collapsed"
                    )
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
                            st.code(
                                table.get("markdown", ""), language="markdown"
                            )
                        st.divider()

            with tab3:
                st.subheader(f"Page {page_num} — Raw OCR Text")
                st.text_area(
                    "raw", value=raw_text, height=400,
                    label_visibility="collapsed"
                )

            with tab4:
                st.subheader("Full Pipeline Output (JSON)")
                st.json({"ocr": ocr_output, "kie": kie_output})

    finally:
        os.unlink(tmp_path)

elif not uploaded_file:
    st.info("👈 Upload a document from the sidebar to get started.")
