from pathlib import Path


def build_document_output(
    doc_id: str,
    pages_ocr: list[dict],
    pages_fields: list[dict],
    doc_type: dict
) -> dict:
    pages = []
    for ocr, fields in zip(pages_ocr, pages_fields):
        pages.append({
            "page_number": ocr.get("page", 1),
            "ocr_confidence": ocr.get("confidence", 0.0),
            "fields": fields,
            "tables": ocr.get("tables", [])
        })

    return {
        "document_id": doc_id,
        "total_pages": len(pages),
        "doc_type": doc_type.get("doc_type"),
        "doc_subtype": doc_type.get("doc_subtype"),
        "pages": pages
    }
