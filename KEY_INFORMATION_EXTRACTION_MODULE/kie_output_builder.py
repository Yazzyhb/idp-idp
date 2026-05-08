from pathlib import Path


def build_document_output(
    doc_id: str,
    pages_ocr: list[dict],
    pages_fields: list[dict],
    doc_type: dict,
    pages_field_details: list[dict] | None = None,
) -> dict:
    pages = []
    for index, (ocr, fields) in enumerate(zip(pages_ocr, pages_fields)):
        page_entry = {
            "page_number": ocr.get("page", 1),
            "ocr_confidence": ocr.get("confidence", 0.0),
            "fields": fields,
            "tables": ocr.get("tables", []),
        }
        if pages_field_details is not None and index < len(pages_field_details):
            page_entry["field_details"] = pages_field_details[index]
        pages.append(page_entry)

    return {
        "document_id": doc_id,
        "total_pages": len(pages),
        "doc_type": doc_type.get("doc_type"),
        "doc_subtype": doc_type.get("doc_subtype"),
        "pages": pages
    }
