from kie_doc_type import detect_doc_type
from kie_field_extractor import extract_fields
from kie_output_builder import build_document_output
from pathlib import Path


def extract(ocr_output: list[dict], doc_id: str = None) -> dict:
    if not doc_id:
        doc_id = f"doc_{id(ocr_output)}"

    full_text = "\n".join(p.get("raw_text", "") for p in ocr_output)
    doc_type = detect_doc_type(full_text)

    pages_fields = []
    for page in ocr_output:
        raw_text = page.get("raw_text", "")
        fields = extract_fields(raw_text)
        pages_fields.append(fields)

    return build_document_output(doc_id, ocr_output, pages_fields, doc_type)
