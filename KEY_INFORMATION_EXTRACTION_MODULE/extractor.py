from kie_doc_type import detect_doc_type
from kie_field_extractor import extract_fields_detailed
from kie_output_builder import build_document_output
from pathlib import Path


def extract(ocr_output: list[dict], doc_id: str = None) -> dict:
    if not doc_id:
        doc_id = f"doc_{id(ocr_output)}"

    full_text = "\n".join(p.get("raw_text", "") for p in ocr_output)
    doc_type = detect_doc_type(full_text)

    pages_fields = []
    pages_field_details = []
    for page in ocr_output:
        raw_text = page.get("raw_text", "")
        page_confidence = page.get("confidence", 0.0)
        detailed = extract_fields_detailed(raw_text, page_confidence=page_confidence)
        fields = detailed["fields"]
        pages_fields.append(fields)
        pages_field_details.append(detailed)

    return build_document_output(doc_id, ocr_output, pages_fields, doc_type, pages_field_details)
