"""
=============================================================================
  TABLE EXTRACTION MODULE
  img2table (primary) + TATR (fallback)

  Fixes applied:
    1. Word-level OCR adapter  — splits Surya text_lines into pseudo-words
    2. Column detection fix    — gentle denoising only, no binarization
    3. Table region masking    — returns bboxes to mask body text
    4. TATR model caching      — models loaded once, reused across calls
    5. TATR detection first    — finds table regions before structure
=============================================================================
"""

import os
import tempfile

import cv2
import numpy as np
import polars as pl
from PIL import Image

from img2table.document import Image as Img2TableImage
from img2table.ocr.base import OCRInstance
from img2table.ocr.data import OCRDataframe
from img2table.document.base import Document


# =============================================================================
# TATR MODEL CACHE — loaded once, reused across all calls
# =============================================================================

_tatr_det_model = None
_tatr_det_processor = None
_tatr_str_model = None
_tatr_str_processor = None


def _load_tatr_models():
    global _tatr_det_model, _tatr_det_processor
    global _tatr_str_model, _tatr_str_processor

    if _tatr_det_model is not None:
        return True

    try:
        from transformers import (
            AutoImageProcessor,
            TableTransformerForObjectDetection
        )
    except ImportError:
        return False

    DETECTION_MODEL = "microsoft/table-transformer-detection"
    STRUCTURE_MODEL = "microsoft/table-transformer-structure-recognition-v1.1-all"

    _tatr_det_processor = AutoImageProcessor.from_pretrained(DETECTION_MODEL)
    _tatr_det_model = TableTransformerForObjectDetection.from_pretrained(
        DETECTION_MODEL
    )
    _tatr_str_processor = AutoImageProcessor.from_pretrained(
        STRUCTURE_MODEL, size={"height": 800, "width": 800}
    )
    _tatr_str_model = TableTransformerForObjectDetection.from_pretrained(
        STRUCTURE_MODEL
    )
    return True


# =============================================================================
# WORD-LEVEL OCR ADAPTER
# =============================================================================

class PrecomputedOCRAdapter(OCRInstance):
    """
    Bridges Surya OCR output to img2table's expected word-level format.

    Surya returns full text lines with one bounding polygon per line.
    img2table needs individual words with tight bboxes to correctly
    assign text to table cells. This adapter splits each line into
    words and distributes the bbox proportionally by character count.
    """

    def __init__(self, ocr_result, page_index: int = 0):
        self.ocr_result = ocr_result
        self.page_index = page_index

    def content(self, document: Document) -> list:
        return [self.ocr_result]

    def to_ocr_dataframe(self, content: list) -> OCRDataframe:
        list_elements = []

        for page_id, ocr_result in enumerate(content):
            actual_page = self.page_index

            for line_idx, text_line in enumerate(ocr_result.text_lines):
                if not text_line.text or not text_line.text.strip():
                    continue

                xs = [p[0] for p in text_line.polygon]
                ys = [p[1] for p in text_line.polygon]
                line_x1 = int(min(xs))
                line_y1 = int(min(ys))
                line_x2 = int(max(xs))
                line_y2 = int(max(ys))

                line_conf = round(100 * (text_line.confidence or 0.9))
                line_width = max(1, line_x2 - line_x1)

                words = text_line.text.split()
                if not words:
                    continue

                total_chars = sum(len(w) for w in words)
                if total_chars == 0:
                    continue

                n_spaces = len(words) - 1
                total_units = total_chars + (n_spaces * 0.5)
                char_width = line_width / total_units

                current_x = line_x1
                for w_idx, word in enumerate(words):
                    word_pixel_width = int(len(word) * char_width)
                    word_x1 = current_x
                    word_x2 = current_x + word_pixel_width
                    if w_idx == len(words) - 1:
                        word_x2 = line_x2

                    list_elements.append({
                        "page": actual_page,
                        "class": "ocrx_word",
                        "id": f"word_{actual_page + 1}_{line_idx + 1}_{w_idx}",
                        "parent": f"line_{actual_page + 1}_{line_idx + 1}",
                        "value": word,
                        "confidence": line_conf,
                        "x1": word_x1,
                        "y1": line_y1,
                        "x2": word_x2,
                        "y2": line_y2,
                    })

                    current_x = word_x2 + int(0.5 * char_width)

        if not list_elements:
            return None

        return OCRDataframe(df=pl.DataFrame(list_elements))


# =============================================================================
# PREPROCESSING — gentle denoising only, preserves table border lines
# =============================================================================

def _preprocess_for_table_detection(img: np.ndarray) -> np.ndarray:
    """
    Gentle denoising without binarization.
    img2table needs a grayscale/color image — binarization destroys
    faint table borders making them undetectable.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(
        gray, h=5, templateWindowSize=7, searchWindowSize=15
    )
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)


# =============================================================================
# IMG2TABLE EXTRACTION (PRIMARY)
# =============================================================================

def _estimate_table_bbox(table) -> tuple:
    """Estimate table bbox from its cells when top-level bbox is missing."""
    if not hasattr(table, 'content') or not table.content:
        return (0, 0, 0, 0)
    all_x1, all_y1, all_x2, all_y2 = [], [], [], []
    for row_cells in table.content.values():
        for cell in row_cells:
            if hasattr(cell, 'bbox') and cell.bbox:
                all_x1.append(cell.bbox.x1)
                all_y1.append(cell.bbox.y1)
                all_x2.append(cell.bbox.x2)
                all_y2.append(cell.bbox.y2)
    if not all_x1:
        return (0, 0, 0, 0)
    return (min(all_x1), min(all_y1), max(all_x2), max(all_y2))

def _extract_with_img2table(img, ocr_adapter, page_index=0):
    preprocessed = _preprocess_for_table_detection(img)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, preprocessed)

    try:
        doc = Img2TableImage(src=tmp_path, detect_rotation=False)
        tables = doc.extract_tables(
            ocr=ocr_adapter,
            implicit_rows=True,
            implicit_columns=False,
            borderless_tables=False,
            min_confidence=40
        )
        # removed second attempt — saves ~30s on CPU
        if not tables:
            return [], []

        bboxes = []
        for table in tables:
            if hasattr(table, 'bbox') and table.bbox is not None:
                bboxes.append((
                    table.bbox.x1, table.bbox.y1,
                    table.bbox.x2, table.bbox.y2
                ))
            else:
                bboxes.append(_estimate_table_bbox(table))

        return tables, bboxes
    finally:
        os.unlink(tmp_path)


def _convert_img2table_results(tables: list, bboxes: list) -> list[dict]:
    results = []
    for i, (table, bbox) in enumerate(zip(tables, bboxes)):
        df = table.df.fillna("")
        rows = df.values.tolist()
        total = df.size
        filled = sum(1 for row in rows for cell in row if str(cell).strip())
        confidence = round(filled / total, 2) if total > 0 else 0.0
        n_rows = len(rows)
        n_cols = len(rows[0]) if rows else 0
        results.append({
            "table_index": i,
            "data": rows,
            "bbox": [int(v) for v in bbox],
            "confidence": confidence,
            "method": "img2table",
            "shape": [n_rows, n_cols]
        })
    return results



# =============================================================================
# TATR EXTRACTION (FALLBACK)
# =============================================================================

def _extract_with_tatr(
    raw_img: np.ndarray,
    ocr_result
) -> tuple[list, list]:
    """TATR deep learning fallback. Returns (tables, bboxes)."""
    if not _load_tatr_models():
        return [], []

    import torch

    STRUCTURE_LABELS = {
        0: "table", 1: "table column", 2: "table row",
        3: "table column header", 4: "table projected row header",
        5: "table spanning cell", 6: "no object"
    }

    pil_image = Image.fromarray(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))

    det_inputs = _tatr_det_processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        det_outputs = _tatr_det_model(**det_inputs)

    target_sizes = torch.tensor([pil_image.size[::-1]])
    det_results = _tatr_det_processor.post_process_object_detection(
        det_outputs, threshold=0.7, target_sizes=target_sizes
    )[0]

    if len(det_results["boxes"]) == 0:
        return [], []

    all_tables = []
    all_bboxes = []

    for table_box in det_results["boxes"]:
        tx1, ty1, tx2, ty2 = [int(v) for v in table_box.tolist()]
        pad = 10
        tx1 = max(0, tx1 - pad)
        ty1 = max(0, ty1 - pad)
        tx2 = min(pil_image.width, tx2 + pad)
        ty2 = min(pil_image.height, ty2 + pad)

        table_crop = pil_image.crop((tx1, ty1, tx2, ty2))

        str_inputs = _tatr_str_processor(images=table_crop, return_tensors="pt")
        with torch.no_grad():
            str_outputs = _tatr_str_model(**str_inputs)

        crop_target = torch.tensor([table_crop.size[::-1]])
        str_results = _tatr_str_processor.post_process_object_detection(
            str_outputs, threshold=0.6, target_sizes=crop_target
        )[0]

        rows, cols = [], []
        for box, label in zip(str_results["boxes"], str_results["labels"]):
            label_name = STRUCTURE_LABELS.get(label.item(), "")
            bbox = [int(v) for v in box.tolist()]
            if "row" in label_name and "header" not in label_name:
                rows.append(bbox)
            elif "column" in label_name and "header" not in label_name:
                cols.append(bbox)

        if not rows or not cols:
            continue

        rows.sort(key=lambda b: b[1])
        cols.sort(key=lambda b: b[0])

        num_rows, num_cols = len(rows), len(cols)
        grid = [[""] * num_cols for _ in range(num_rows)]

        for text_line in ocr_result.text_lines:
            xs = [p[0] for p in text_line.polygon]
            ys = [p[1] for p in text_line.polygon]
            cx = (min(xs) + max(xs)) / 2 - tx1
            cy = (min(ys) + max(ys)) / 2 - ty1
            row_idx = next(
                (i for i, b in enumerate(rows) if b[1] <= cy <= b[3]), None
            )
            col_idx = next(
                (i for i, b in enumerate(cols) if b[0] <= cx <= b[2]), None
            )
            if row_idx is not None and col_idx is not None:
                existing = grid[row_idx][col_idx]
                grid[row_idx][col_idx] = (
                    (existing + " " + text_line.text).strip()
                    if existing else text_line.text
                )

        filled = sum(1 for row in grid for cell in row if str(cell).strip())
        total = num_rows * num_cols
        if total == 0 or filled / total < 0.3:
            continue

        all_tables.append({
            "data": grid,
            "confidence": round(filled / total, 2),
            "method": "tatr",
            "shape": (num_rows, num_cols)
        })
        all_bboxes.append((tx1, ty1, tx2, ty2))

    return all_tables, all_bboxes


# =============================================================================
# BODY TEXT MASKING
# =============================================================================

def mask_table_regions_from_text(
    ocr_result,
    table_bboxes: list[tuple],
    padding: int = 8
) -> list:
    """
    Filter OCR text lines to exclude anything inside a detected table.
    Returns filtered list of text_lines with no table content.
    """
    if not table_bboxes:
        return ocr_result.text_lines

    def line_centroid(text_line):
        xs = [p[0] for p in text_line.polygon]
        ys = [p[1] for p in text_line.polygon]
        return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2

    def in_any_table(cx, cy):
        for x1, y1, x2, y2 in table_bboxes:
            if (x1 - padding <= cx <= x2 + padding and
                    y1 - padding <= cy <= y2 + padding):
                return True
        return False

    return [
        line for line in ocr_result.text_lines
        if not in_any_table(*line_centroid(line))
    ]


def rebuild_body_text(filtered_lines: list) -> str:
    """
    Reconstruct clean body text from non-table OCR lines
    sorted in reading order.
    """
    if not filtered_lines:
        return ""

    sorted_lines = sorted(
        filtered_lines,
        key=lambda l: (
            min(p[1] for p in l.polygon),
            min(p[0] for p in l.polygon)
        )
    )
    return "\n".join(
        line.text for line in sorted_lines if line.text and line.text.strip()
    )


# =============================================================================
# PUBLIC API
# =============================================================================

def extract_tables(
    raw_img: np.ndarray,
    ocr_result=None,
    page_index: int = 0
) -> tuple[list[dict], list[tuple]]:
    """
    Extract tables from a document page.

    Args:
        raw_img:    BGR numpy array of the raw (non-preprocessed) page
        ocr_result: Surya OCRResult for this page
        page_index: 0-based page number

    Returns:
        (tables, bboxes)
        - tables: list of dicts with keys:
                  table_index, data, bbox, confidence, method, shape
        - bboxes: list of (x1,y1,x2,y2) page coordinates

    Usage:
        tables, bboxes = extract_tables(raw_img, ocr_result, page_index=0)
        clean_lines = mask_table_regions_from_text(ocr_result, bboxes)
        body_text = rebuild_body_text(clean_lines)
    """
    ocr_adapter = (
        PrecomputedOCRAdapter(ocr_result, page_index=page_index)
        if ocr_result else None
    )

    # primary: img2table
    if ocr_adapter:
        raw_tables, bboxes = _extract_with_img2table(
            raw_img, ocr_adapter, page_index=page_index
        )
        if raw_tables:
            return _convert_img2table_results(raw_tables, bboxes), bboxes

    # fallback: TATR
    if ocr_result:
        tatr_tables, bboxes = _extract_with_tatr(raw_img, ocr_result)
        if tatr_tables:
            for i, t in enumerate(tatr_tables):
                t["table_index"] = i
            return tatr_tables, bboxes

    return [], []
