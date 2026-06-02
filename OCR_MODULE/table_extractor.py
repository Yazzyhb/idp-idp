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


def _line_text(text_line) -> str:
    if hasattr(text_line, "text"):
        return str(getattr(text_line, "text") or "")
    if isinstance(text_line, (tuple, list)) and text_line:
        first = text_line[0]
        return str(first) if first is not None else ""
    return ""


def _line_confidence(text_line, default: float = 0.9) -> float:
    if hasattr(text_line, "confidence"):
        try:
            return float(getattr(text_line, "confidence") or default)
        except Exception:
            return default
    if isinstance(text_line, (tuple, list)):
        for item in text_line[1:]:
            if isinstance(item, (int, float)):
                return float(item)
    return default


def _line_polygon(text_line):
    if hasattr(text_line, "polygon"):
        return getattr(text_line, "polygon") or []
    if isinstance(text_line, (tuple, list)):
        for item in text_line[1:]:
            if isinstance(item, (list, tuple)) and item and isinstance(item[0], (list, tuple)):
                return item
    return []


# =============================================================================
# TATR MODEL CACHE
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
        from transformers import AutoImageProcessor, TableTransformerForObjectDetection
    except ImportError:
        return False
    DETECTION_MODEL = "microsoft/table-transformer-detection"
    STRUCTURE_MODEL = "microsoft/table-transformer-structure-recognition-v1.1-all"
    _tatr_det_processor = AutoImageProcessor.from_pretrained(DETECTION_MODEL)
    _tatr_det_model = TableTransformerForObjectDetection.from_pretrained(DETECTION_MODEL)
    _tatr_str_processor = AutoImageProcessor.from_pretrained(
        STRUCTURE_MODEL, size={"height": 800, "width": 800}
    )
    _tatr_str_model = TableTransformerForObjectDetection.from_pretrained(STRUCTURE_MODEL)
    return True


# =============================================================================
# WORD-LEVEL OCR ADAPTER
# =============================================================================

class PrecomputedOCRAdapter(OCRInstance):
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
                line_text = _line_text(text_line).strip()
                if not line_text:
                    continue
                polygon = _line_polygon(text_line)
                if not polygon:
                    continue
                xs = [p[0] for p in polygon]
                ys = [p[1] for p in polygon]
                line_x1, line_y1 = int(min(xs)), int(min(ys))
                line_x2, line_y2 = int(max(xs)), int(max(ys))
                line_conf = round(100 * _line_confidence(text_line, default=0.9))
                line_width = max(1, line_x2 - line_x1)
                words = line_text.split()
                if not words:
                    continue
                total_chars = sum(len(w) for w in words)
                if total_chars == 0:
                    continue
                total_units = total_chars + (len(words) - 1) * 0.5
                char_width = line_width / total_units
                current_x = line_x1
                for w_idx, word in enumerate(words):
                    word_x1 = current_x
                    word_x2 = current_x + int(len(word) * char_width)
                    if w_idx == len(words) - 1:
                        word_x2 = line_x2
                    list_elements.append({
                        "page": actual_page,
                        "class": "ocrx_word",
                        "id": f"word_{actual_page+1}_{line_idx+1}_{w_idx}",
                        "parent": f"line_{actual_page+1}_{line_idx+1}",
                        "value": word,
                        "confidence": line_conf,
                        "x1": word_x1, "y1": line_y1,
                        "x2": word_x2, "y2": line_y2,
                    })
                    current_x = word_x2 + int(0.5 * char_width)
        if not list_elements:
            return None
        return OCRDataframe(df=pl.DataFrame(list_elements))


# =============================================================================
# PREPROCESSING FOR TABLE DETECTION
# =============================================================================

def _preprocess_for_table_detection(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=5, templateWindowSize=7, searchWindowSize=15)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)


# =============================================================================
# IMG2TABLE EXTRACTION (PRIMARY)
# =============================================================================

def _estimate_table_bbox(table) -> tuple:
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


def _extract_with_img2table(img: np.ndarray, ocr_adapter, page_index: int = 0):
    preprocessed = _preprocess_for_table_detection(img)
    page_h, page_w = img.shape[:2]
    page_area = page_h * page_w

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
        if not tables:
            return [], []

        filtered_tables, bboxes = [], []
        for table in tables:
            if hasattr(table, 'bbox') and table.bbox is not None:
                bbox = (table.bbox.x1, table.bbox.y1, table.bbox.x2, table.bbox.y2)
            else:
                bbox = _estimate_table_bbox(table)
            x1, y1, x2, y2 = bbox
            table_area = max(0, x2 - x1) * max(0, y2 - y1)
            # reject false positives: anything covering >30% of page is document body
            if table_area / page_area > 0.30:
                continue
            filtered_tables.append(table)
            bboxes.append(bbox)

        return filtered_tables, bboxes
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

def _extract_with_tatr(raw_img: np.ndarray, ocr_result) -> tuple[list, list]:
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

    all_tables, all_bboxes = [], []
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
            polygon = _line_polygon(text_line)
            if not polygon:
                continue
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            cx = (min(xs) + max(xs)) / 2 - tx1
            cy = (min(ys) + max(ys)) / 2 - ty1
            row_idx = next((i for i, b in enumerate(rows) if b[1] <= cy <= b[3]), None)
            col_idx = next((i for i, b in enumerate(cols) if b[0] <= cx <= b[2]), None)
            if row_idx is not None and col_idx is not None:
                existing = grid[row_idx][col_idx]
                line_text = _line_text(text_line).strip()
                if not line_text:
                    continue
                grid[row_idx][col_idx] = (
                    (existing + " " + line_text).strip() if existing else line_text
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

def mask_table_regions_from_text(ocr_result, table_bboxes: list[tuple], padding: int = 8) -> list:
    if not table_bboxes:
        return ocr_result.text_lines

    def line_centroid(text_line):
        polygon = _line_polygon(text_line)
        if not polygon:
            return -1, -1
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2

    def in_any_table(cx, cy):
        for x1, y1, x2, y2 in table_bboxes:
            if x1 - padding <= cx <= x2 + padding and y1 - padding <= cy <= y2 + padding:
                return True
        return False

    return [line for line in ocr_result.text_lines if not in_any_table(*line_centroid(line))]


def rebuild_body_text(filtered_lines: list) -> str:
    if not filtered_lines:
        return ""
    sorted_lines = sorted(
        filtered_lines,
        key=lambda l: (
            min(p[1] for p in (_line_polygon(l) or [(0, 0)])),
            min(p[0] for p in (_line_polygon(l) or [(0, 0)])),
        )
    )
    texts = [_line_text(line).strip() for line in sorted_lines]
    return "\n".join(t for t in texts if t)


# =============================================================================
# PUBLIC API
# =============================================================================

def extract_tables(
    raw_img: np.ndarray,
    ocr_result=None,
    page_index: int = 0
) -> tuple[list[dict], list[tuple]]:
    ocr_adapter = (
        PrecomputedOCRAdapter(ocr_result, page_index=page_index)
        if ocr_result else None
    )

    # primary: img2table
    if ocr_adapter:
        raw_tables, bboxes = _extract_with_img2table(raw_img, ocr_adapter, page_index=page_index)
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
