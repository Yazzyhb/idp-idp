"""
================================================================================
COMPLETE IDP PIPELINE
================================================================================
Single-file integration of:
  - OCR Module (preprocessor, layout, ocr_engine, table_extractor, output_builder)
  - KIE Module (field_extractor, doc_type, output_builder)
  - Full Evaluator (metrics, evaluation harness)

Organized with clear sections. No external dependencies beyond standard libraries
and ML models (surya, transformers, img2table, pypdfium2, etc.).

Use: python complete_pipeline.py --docs N
  or import and call process_document(), extract(), evaluate() from notebooks
================================================================================
"""

import multiprocessing
multiprocessing.freeze_support()

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import csv
import json
import re
import sys
import time
import tempfile
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "documents"
GT_CSV = DOCS_DIR / "generated_documents.csv"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: OCR MODULE - PREPROCESSOR FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def load_document(file_path: str) -> list[np.ndarray]:
    """Load document (PDF or image) and return list of page images."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise ImportError("pypdfium2 required. Install: pip install pypdfium2")
    
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pdf = pdfium.PdfDocument(str(path))
        pages = []
        for i in range(len(pdf)):
            bitmap = pdf[i].render(scale=300/72)
            pil_image = bitmap.to_pil()
            pages.append(cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))
        return pages
    elif suffix in [".jpg", ".jpeg", ".png"]:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not read image: {file_path}")
        return [img]
    else:
        raise ValueError(f"Unsupported format: {suffix}")


def _is_digital(image: np.ndarray) -> bool:
    """True if the image is a clean digital render (not a degraded paper scan)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    # digital PDFs: mostly white background (mean > 200), low noise (std < 60)
    return mean > 200 and std < 60


def _sharpen(image: np.ndarray) -> np.ndarray:
    """Mild unsharp-mask — safe for both digital and scanned docs."""
    blurred = cv2.GaussianBlur(image, (0, 0), 1.5)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


def deskew(image: np.ndarray) -> np.ndarray:
    """Rotate image to correct skew."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(gray > 0))
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    if abs(angle) < 0.5:
        return image
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def denoise(image: np.ndarray) -> np.ndarray:
    """Remove noise from image."""
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


def binarize(image: np.ndarray) -> np.ndarray:
    """Convert to binary (black and white) for better OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def preprocess(image: np.ndarray) -> np.ndarray:
    """Preprocess image for OCR: adapt strategy based on image type."""
    if _is_digital(image):
        # clean digital render — heavy preprocessing hurts quality, just sharpen
        return _sharpen(image)
    # degraded paper scan — full pipeline
    image = deskew(image)
    image = denoise(image)
    image = binarize(image)
    return image


def preprocess_document(file_path: str) -> list[np.ndarray]:
    """Load and preprocess all pages of a document."""
    pages = load_document(file_path)
    return [preprocess(page) for page in pages]


def detect_and_correct_rotated_regions(image: np.ndarray) -> np.ndarray:
    """Detect and correct locally rotated regions in the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = image.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500 or area > image.shape[0] * image.shape[1] * 0.1:
            continue
        rect = cv2.minAreaRect(contour)
        angle = rect[-1]
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        if abs(angle) > 5:
            x, y, w, h = cv2.boundingRect(contour)
            pad = 10
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
            region = image[y1:y2, x1:x2]
            if region.size == 0:
                continue
            M = cv2.getRotationMatrix2D((region.shape[1]//2, region.shape[0]//2), angle, 1.0)
            result[y1:y2, x1:x2] = cv2.warpAffine(
                region, M, (region.shape[1], region.shape[0]),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
    return result


def detect_circular_stamps(image: np.ndarray) -> list[tuple]:
    """Detect circular stamps/watermarks in the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
        param1=50, param2=30, minRadius=30, maxRadius=200
    )
    if circles is None:
        return []
    circles = np.round(circles[0, :]).astype("int")
    return [(x, y, r) for x, y, r in circles]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: OCR MODULE - LAYOUT DETECTION
# ═════════════════════════════════════════════════════════════════════════════

_layout_predictors = None

def _load_layout_predictors():
    """Lazy-load layout detection models."""
    global _layout_predictors
    if _layout_predictors is None:
        try:
            from surya.detection import DetectionPredictor
            from surya.layout import LayoutPredictor
            from surya.recognition import RecognitionPredictor
            from surya.table_rec import TableRecPredictor
            from surya.foundation import FoundationPredictor
            from surya.settings import settings
            
            _layout_predictors = {
                "detection": DetectionPredictor(),
                "recognition": RecognitionPredictor(
                    FoundationPredictor(checkpoint=settings.RECOGNITION_MODEL_CHECKPOINT)
                ),
                "layout": LayoutPredictor(
                    FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
                ),
                "table_rec": TableRecPredictor(),
            }
        except ImportError:
            raise ImportError("surya required. Install: pip install surya-ocr")


def get_predictors():
    """Get cached OCR/layout predictor instances."""
    _load_layout_predictors()
    return _layout_predictors


def detect_layout(pil_image: Image.Image) -> dict:
    """Detect layout regions (text, tables) in the image."""
    predictors = get_predictors()
    results = predictors["layout"]([pil_image])
    layout = results[0]

    regions = {"text": [], "table": []}
    for box in layout.bboxes:
        poly = box.polygon
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
        label = box.label.lower()
        if "table" in label:
            regions["table"].append(bbox)
        else:
            regions["text"].append(bbox)

    return regions


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: OCR MODULE - OCR ENGINE
# ═════════════════════════════════════════════════════════════════════════════

_trocr_processor = None
_trocr_model = None


def _ocr_line_text(line) -> str:
    if hasattr(line, "text"):
        return str(getattr(line, "text") or "")
    if isinstance(line, (tuple, list)) and line:
        return str(line[0] or "")
    return str(line or "")


def _ocr_line_confidence(line, default: float = 0.0) -> float:
    if hasattr(line, "confidence"):
        try:
            return float(getattr(line, "confidence") or default)
        except Exception:
            return default
    if isinstance(line, (tuple, list)):
        for item in line[1:]:
            if isinstance(item, (int, float)):
                return float(item)
    return default


def _ocr_line_polygon(line):
    if hasattr(line, "polygon"):
        return getattr(line, "polygon") or []
    if isinstance(line, (tuple, list)):
        for item in line[1:]:
            if isinstance(item, (list, tuple)) and item and isinstance(item[0], (list, tuple)):
                return item
    return []


def _load_trocr():
    """Lazy-load TrOCR model for handwritten text (disabled for CPU speed)."""
    global _trocr_processor, _trocr_model
    if _trocr_model is None:
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            _trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            _trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        except ImportError:
            raise ImportError("transformers required. Install: pip install transformers")


def is_handwritten(region_img: np.ndarray) -> bool:
    """Detect if region contains handwritten text (currently disabled for speed)."""
    return False  # CPU too slow for TrOCR in demo


def ocr_full_page(pil_image: Image.Image) -> tuple[str, float]:
    """Run full-page OCR with Surya."""
    predictors = get_predictors()
    results = predictors["recognition"](
        [pil_image],
        det_predictor=predictors["detection"],
        task_names=None,
        sort_lines=True
    )
    result = results[0]
    if not result.text_lines:
        return "", 0.0
    texts = [_ocr_line_text(line).strip() for line in result.text_lines]
    texts = [t for t in texts if t]
    if not texts:
        return "", 0.0
    text = "\n".join(texts)
    avg_conf = float(np.mean([_ocr_line_confidence(line) for line in result.text_lines]))
    return text, avg_conf


def ocr_region_printed(pil_image: Image.Image) -> tuple[str, float]:
    """Run OCR on a printed region."""
    predictors = get_predictors()
    results = predictors["recognition"](
        [pil_image],
        det_predictor=predictors["detection"],
        task_names=None
    )
    result = results[0]
    if not result.text_lines:
        return "", 0.0
    texts = [_ocr_line_text(line).strip() for line in result.text_lines]
    texts = [t for t in texts if t]
    if not texts:
        return "", 0.0
    text = "\n".join(texts)
    avg_conf = float(np.mean([_ocr_line_confidence(line) for line in result.text_lines]))
    return text, avg_conf


def ocr_region_handwritten(region_img: np.ndarray) -> tuple[str, float]:
    """Run OCR on handwritten region using TrOCR."""
    import torch
    _load_trocr()
    pil_image = Image.fromarray(cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB)).convert("RGB")
    pixel_values = _trocr_processor(pil_image, return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = _trocr_model.generate(pixel_values)
    text = _trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text, 0.70


def ocr_region(region_img: np.ndarray) -> tuple[str, float]:
    """Detect handwriting vs. printed and run appropriate OCR."""
    pil_image = Image.fromarray(cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB))
    if is_handwritten(region_img):
        return ocr_region_handwritten(region_img)
    return ocr_region_printed(pil_image)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: OCR MODULE - TABLE EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

_tatr_det_model = None
_tatr_det_processor = None
_tatr_str_model = None
_tatr_str_processor = None


def _load_tatr_models():
    """Lazy-load Table Transformer models."""
    global _tatr_det_model, _tatr_det_processor, _tatr_str_model, _tatr_str_processor
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


def _preprocess_for_table_detection(img: np.ndarray) -> np.ndarray:
    """Preprocess image for robust table detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=5, templateWindowSize=7, searchWindowSize=15)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)


def _estimate_table_bbox(table) -> tuple:
    """Estimate bounding box of table from cells."""
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


def _extract_with_img2table(img: np.ndarray, page_index: int = 0):
    """Extract tables using img2table library."""
    try:
        from img2table.document import Image as Img2TableImage
        from img2table.ocr.base import OCRInstance
        from img2table.ocr.data import OCRDataframe
        from img2table.document.base import Document
        import polars as pl
    except ImportError:
        return [], []

    preprocessed = _preprocess_for_table_detection(img)
    page_h, page_w = img.shape[:2]
    page_area = page_h * page_w

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, preprocessed)

    try:
        doc = Img2TableImage(src=tmp_path, detect_rotation=False)
        tables = doc.extract_tables(
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
    """Convert img2table results to standardized format."""
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


def extract_tables(img: np.ndarray, ocr_result=None, page_index: int = 0) -> tuple[list, list]:
    """Extract tables from image using img2table."""
    tables, bboxes = _extract_with_img2table(img, page_index)
    results = _convert_img2table_results(tables, bboxes)
    return results, bboxes


def mask_table_regions_from_text(ocr_result, table_bboxes):
    """Mask out table regions from OCR text lines."""
    if not hasattr(ocr_result, 'text_lines'):
        return ocr_result
    filtered_lines = []
    for line in ocr_result.text_lines:
        polygon = _ocr_line_polygon(line)
        if not polygon:
            filtered_lines.append(line)
            continue
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        line_x1, line_y1 = min(xs), min(ys)
        line_x2, line_y2 = max(xs), max(ys)
        in_table = False
        for bbox in table_bboxes:
            t_x1, t_y1, t_x2, t_y2 = bbox
            if not (line_x2 < t_x1 or line_x1 > t_x2 or line_y2 < t_y1 or line_y1 > t_y2):
                in_table = True
                break
        if not in_table:
            filtered_lines.append(line)
    # Return modified OCR result
    ocr_result.text_lines = filtered_lines
    return ocr_result


def rebuild_body_text(filtered_lines) -> str:
    """Rebuild body text from filtered OCR lines."""
    if not filtered_lines:
        return ""
    return "\n".join(
        text for text in (_ocr_line_text(line).strip() for line in filtered_lines)
        if text
    )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: OCR MODULE - OUTPUT BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def _strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r'<[^>]+>', '', text)


def build_page_output(page_number: int, full_text: str, text_confidence: float, tables: list[dict]) -> dict:
    """Build structured output for a single OCR page."""
    return {
        "page": page_number,
        "raw_text": _strip_html(full_text).strip(),
        "confidence": round(text_confidence, 2),
        "tables": tables
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: KIE MODULE - FIELD EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def _norm_ocr_label(line: str) -> str:
    """Normalize OCR labels (fix common confusion patterns)."""
    s = (line or "").replace("'", "'").strip()
    s = re.sub(r"[|¦]+", ":", s)
    s = re.sub(r"^\s*R[EF]{1,2}\s*[:\-]?\s*", "REF: ", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*R[ée]f\s*[:\-]?\s*", "Réf : ", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*Obje[tl1]\s*[:\-]?\s*", "Objet : ", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*P[\.\s]*[Jj1][\.\s]*[:\-]?\s*", "P.J : ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s{2,}", " ", s)
    return s


def _normalized_lines(raw_text: str) -> list[str]:
    """Split and normalize text lines."""
    return [_norm_ocr_label(l) for l in raw_text.split("\n")]


# Regex patterns for field extraction
_REF_CODE = (
    r"[A-Z]{2,}(?:[/-][A-Z0-9]{2,})*"
    r"(?:"
    r"[/-]N[°º]\s*\d*[/-]\d{4}"
    r"|[/-]N[°º]\s*\d+"
    r"|[/-]\d{4}[/-]\d+"
    r"|[/-]\d{4}"
    r")"
)

_REF_BODY_RE = re.compile(
    r"(?:Réf|REF)\s*:\s*(" + _REF_CODE + r"(?:\s+du\s+[\d/]+)?)",
    re.IGNORECASE
)

_OBJET_RE = re.compile(r"Objet\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_PJ_RE = re.compile(
    r"(?:P\.?\s*J\.?\s*:?\s*|Pi[eè]ces?\s+jointes?\s*:?\s*)(.+?)(?:\n|$)",
    re.IGNORECASE
)

_BODY_START_RE = re.compile(
    r"(?:^|\n)(?:Monsieur\s*[;:,]|"
    r"Faisant\s+suite|Dans\s+le\s+cadre|Suite\s+\u00e0|"
    r"J['\u2019]ai\s+l['\u2019]honneur|Je\s+vous\s+prie|"
    r"Nous\s+avons\s+l['\u2019]honneur|Conform[eé]ment|"
    r"Suite\s+aux|Suite\s+à|"
    r"Nous\s+avons\s+constat[eé]|Nous\s+vous\s+sollicitons|"
    r"Nous\s+vous\s+informons|Nous\s+vous\s+transmettons|"
    r"Le\s+D[eé]partement|Nous\s+avons\s+proc[eé]d[eé]|"
    r"Suite\s+aux\s+r[eé]centes|Nous\s+vous\s+adressons)\s*\n?",
    re.IGNORECASE | re.MULTILINE
)

_BODY_END_RE = re.compile(
    r"(?:Veuillez\s+agréer|Veuillez\s+recevoir|Veuillez\s+trouver|"
    r"Recevez|Je\s+vous\s+prie\s+d['\u2019]agréer|"
    r"Dans\s+l.attente|Comptant\s+sur|En\s+vous\s+remerciant|"
    r"Copie\s+[aà]\s*:|Direction\s+G[eé]n[eé]rale\s*,\s*Service|"
    r"Direction\s+des\s+Op[eé]rations\s*$|"
    r"D[eé]partement\s+Risques\s*,)",
    re.IGNORECASE
)

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_URL_RE = re.compile(r"(?:www\.|http|\.dz|\.com|\.org)", re.IGNORECASE)
_NOISE_RE = re.compile(r"^[^A-Za-zÀ-ÿ]{0,3}[^A-Za-zÀ-ÿ\s]{3,}")
_FOOTER_RE = re.compile(r"(?:Quartier|Tél\s*:|Bab\s+Ezzouar|BP\s*\d)", re.IGNORECASE)


def _is_noise(line: str) -> bool:
    """Detect OCR artifacts and noise."""
    if re.search(r"(?:\+{1,}|%{1,}|N8X|oloX|米)", line):
        return True
    if _ARABIC_RE.search(line):
        return True
    if _URL_RE.search(line):
        return True
    if _NOISE_RE.search(line):
        return True
    if _FOOTER_RE.search(line):
        return True
    if sum(c.isalpha() for c in line) / max(len(line), 1) < 0.4:
        return True
    if len(line) < 3:
        return True
    return False


def _is_ref_line(line: str) -> bool:
    """Check if line is a reference code."""
    return bool(re.search(_REF_CODE, line)) or bool(re.match(r'^/?REF\s*:?$', line, re.IGNORECASE))


def _is_keyword_line(line: str) -> bool:
    """Check if line contains structural keywords."""
    return bool(re.match(
        r"(?:Alger|REF|Réf|Objet|P\.J|Copie|www\.|http)",
        line, re.IGNORECASE
    ))


def _extract_sender(raw_text: str) -> Optional[str]:
    """Extract sender field from document."""
    lines = [l.strip() for l in _normalized_lines(raw_text) if l.strip()]

    # pattern 1: explicit "De: <sender>" label
    for line in lines[:12]:
        m = re.match(r"De\s*:\s*(.+)", line, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if not _is_noise(val) and not _is_ref_line(val):
                return val

    # pattern 2: first clean non-noise line before ref/date block
    for line in lines[:8]:
        if _is_ref_line(line):
            break
        if _is_keyword_line(line):
            break
        if _is_noise(line):
            continue
        return line

    # pattern 3: explicit department/service/direction
    for line in lines[:14]:
        if _is_noise(line) or _is_ref_line(line):
            continue
        if re.search(r"\b(Service|Département|Direction)\b", line, re.IGNORECASE):
            return line

    return None


def _extract_ref_header(raw_text: str) -> Optional[str]:
    """Extract reference number from document header."""
    lines = _normalized_lines(raw_text)
    first_lines = [l.strip() for l in lines[:25]]

    # pass 1: "REF:" alone on line, value on next
    for i, line in enumerate(first_lines[:-1]):
        if re.match(r"^REF\s*:?\s*$", line, re.IGNORECASE):
            m = re.search(_REF_CODE, first_lines[i + 1])
            if m:
                return " ".join(m.group(0).split())

    # pass 2: "REF: value" on same line
    for line in first_lines:
        m = re.search(r"REF\s*:\s*(" + _REF_CODE + r")", line, re.IGNORECASE)
        if m:
            return " ".join(m.group(1).split())

    # pass 3: ref code standalone
    for line in first_lines:
        if re.match(r"^Réf\s*:", line, re.IGNORECASE):
            continue
        m = re.search(_REF_CODE, line)
        if m:
            return " ".join(m.group(0).split())

    # pass 4: split across two lines
    for i in range(len(first_lines) - 1):
        for combo in [first_lines[i] + first_lines[i+1],
                      first_lines[i+1] + first_lines[i]]:
            if re.match(r"^Réf\s*:", combo, re.IGNORECASE):
                continue
            m = re.search(_REF_CODE, combo)
            if m:
                return " ".join(m.group(0).split())

    return None


def _extract_ref_body(raw_text: str) -> Optional[str]:
    """Extract reference from document body."""
    norm_text = "\n".join(_normalized_lines(raw_text))
    match = _REF_BODY_RE.search(norm_text)
    return match.group(1).strip() if match else None


def _extract_date(raw_text: str) -> Optional[str]:
    """Extract date field."""
    norm_text = "\n".join(_normalized_lines(raw_text))
    # pattern 1: "Alger:\n DD/MM/YYYY"
    m = re.search(
        r"Alger\s*:?\s*,?\s*(?:le\s*)?\n\s*(\d{2}[/\-]\d{2}[/\-]\d{4})",
        norm_text, re.IGNORECASE
    )
    if m:
        return m.group(1)

    # pattern 2: DD/MM/YYYY in first 25 lines
    for line in norm_text.split("\n")[:25]:
        m = re.search(r"\b(\d{2}[/\-]\d{2}[/\-]\d{4})\b", line)
        if m:
            return m.group(1)

    return None


def _extract_receiver(raw_text: str) -> Optional[str]:
    """Extract receiver/recipient field."""
    lines = [l.strip() for l in _normalized_lines(raw_text)]

    def _clean_receiver(s: str) -> str:
        s = re.sub(r"^[\W_]+", "", (s or "").strip())
        s = re.sub(r"^\bA\s*:\s*", "À: ", s, flags=re.IGNORECASE)
        return s.strip()

    def _valid_receiver_candidate(s: str) -> bool:
        s = (s or "").strip()
        if not s:
            return False
        if _is_noise(s) or _is_ref_line(s) or _is_keyword_line(s):
            return False
        if _URL_RE.search(s) or _FOOTER_RE.search(s):
            return False
        return len(s) > 5

    # Find the "De:" sender line
    de_idx = None
    for i, line in enumerate(lines[:15]):
        if re.match(r"De\s*:", line, re.IGNORECASE):
            de_idx = i
            break

    if de_idx is not None:
        for j in range(de_idx + 1, min(de_idx + 6, len(lines))):
            candidate = lines[j]
            if _valid_receiver_candidate(candidate):
                if sum(c.isalpha() for c in candidate) / len(candidate) > 0.5:
                    return _clean_receiver(candidate)

    # pattern 2: explicit "À l'attention ..." / "À: ..."
    norm_text = "\n".join(lines)
    m = re.search(
        r"(?im)^\s*[AÀ]\s*(?:l['\u2019]attention\s+(?:de|du|des)|:)\s*(.+?)\s*$",
        norm_text,
    )
    if m:
        receiver = _clean_receiver(m.group(1))
        if not _valid_receiver_candidate(receiver):
            receiver = ""
        for line in norm_text[m.end():].split("\n")[:3]:
            line = line.strip()
            if not line:
                break
            if re.match(r"(?:Objet|Réf|P\.J|Alger|REF)", line, re.IGNORECASE):
                break
            if _valid_receiver_candidate(line):
                receiver += "\n" + _clean_receiver(line)
        return receiver.strip() or None

    return None


def _extract_objet(raw_text: str) -> Optional[str]:
    """Extract subject/object field."""
    norm_text = "\n".join(_normalized_lines(raw_text))
    lines = [l.strip() for l in norm_text.split("\n")]

    obj_idx = None
    first_line = ""
    for i, line in enumerate(lines):
        m = re.match(r"^\s*Objet\s*:?\s*(.*)$", line, re.IGNORECASE)
        if m:
            obj_idx = i
            first_line = (m.group(1) or "").strip()
            break
    
    if obj_idx is None:
        return None

    if not first_line:
        for line in lines[obj_idx + 1 : obj_idx + 4]:
            line = line.strip()
            if line:
                first_line = line
                break

    extra = []
    for line in lines[obj_idx + 1 : obj_idx + 5]:
        line = line.strip()
        if not line:
            break
        if re.match(
            r"(?:P\.J|Réf|REF|Alger|Monsieur|Messieurs?|Mesdames?|"
            r"Dans\s+le\s+cadre|Faisant\s+suite|Suite\s+[aà]|"
            r"J['\u2019]ai\s+l['\u2019]honneur|Nous\s+vous|Le\s+D[eé]partement|"
            r"Nous\s+avons|Conform[eé]ment|Suite\s+aux)",
            line, re.IGNORECASE
        ):
            break
        extra.append(line)

    merged = (first_line + (" " + " ".join(extra) if extra else "")).strip()
    merged = re.sub(r"\s+", " ", merged).strip()
    merged = re.sub(r"^\s*Réf\s*:\s*", "", merged, flags=re.IGNORECASE).strip()

    return merged if merged else None


def _extract_pj(raw_text: str) -> Optional[str]:
    """Extract attached pieces (Pièces Jointes) field."""
    norm_text = "\n".join(_normalized_lines(raw_text))
    lines = [l.strip() for l in norm_text.split("\n")]

    for i, line in enumerate(lines):
        m = re.match(r"^\s*P\.?\s*J\.?\s*:?\s*(.*)$", line, re.IGNORECASE)
        if m:
            first_line = (m.group(1) or "").strip()
            extra = []
            for nxt_line in lines[i + 1 : i + 4]:
                nxt_line = nxt_line.strip()
                if not nxt_line:
                    break
                if re.match(r"(?:Objet|Réf|REF|Alger|Monsieur|Copie\s+[aà]\s*:)", nxt_line, re.IGNORECASE):
                    break
                extra.append(nxt_line)
            merged = (first_line + (" " + " ".join(extra) if extra else "")).strip()
            return merged if merged else None

    return None


def _extract_body(raw_text: str) -> Optional[str]:
    """Extract document body text."""
    norm_text = "\n".join(_normalized_lines(raw_text))
    
    start_m = _BODY_START_RE.search(norm_text)
    if not start_m:
        return None
    
    start_idx = start_m.start()
    body_start = norm_text[start_idx:]
    
    end_m = _BODY_END_RE.search(body_start)
    if end_m:
        body_text = body_start[:end_m.start()].strip()
    else:
        body_text = body_start.strip()
    
    return body_text if body_text else None


def extract_fields(raw_text: str) -> dict:
    """Extract all KIE fields from OCR text."""
    return {
        "sender": _extract_sender(raw_text),
        "receiver": _extract_receiver(raw_text),
        "date": _extract_date(raw_text),
        "ref_header": _extract_ref_header(raw_text),
        "objet": _extract_objet(raw_text),
        "pj": _extract_pj(raw_text),
        "body": _extract_body(raw_text),
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: KIE MODULE - DOCUMENT TYPE DETECTION
# ═════════════════════════════════════════════════════════════════════════════

_SUBTYPE_PATTERNS = [
    ("demande", [
        r"\bdemande\b", r"\bdemander\b", r"\bsollicite\b", r"\bprie\b",
        r"\bveuille[zs]\b", r"\bprière\b"
    ]),
    ("transmission", [
        r"\btransmet[s]?\b", r"\bci[- ]joint\b", r"\bci[- ]après\b",
        r"\bvous\s+adresse\b", r"\bfaire\s+parvenir\b"
    ]),
    ("information", [
        r"\bporte[r]?\s+à\s+(votre\s+)?connaissance\b",
        r"\binform\w+\b", r"\bnotif\w+\b", r"\bsignale[r]?\b"
    ]),
]


def detect_doc_type(raw_text: str) -> dict:
    """Detect document type and subtype."""
    text_lower = raw_text.lower()
    subtype = "autre"

    for name, patterns in _SUBTYPE_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, text_lower):
                subtype = name
                break
        if subtype != "autre":
            break

    return {
        "doc_type": "lettre_administrative",
        "doc_subtype": subtype
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: KIE MODULE - OUTPUT BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def build_document_output(
    doc_id: str,
    pages_ocr: list[dict],
    pages_fields: list[dict],
    doc_type: dict
) -> dict:
    """Build structured KIE output."""
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


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9: OCR ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

class _JSONEncoder(json.JSONEncoder):
    """JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _run_ocr(pil_image: Image.Image):
    """Execute OCR on a PIL image."""
    predictors = get_predictors()
    results = predictors["recognition"](
        [pil_image],
        det_predictor=predictors["detection"],
        task_names=None,
        sort_lines=True,
    )
    return results[0]


def _build_text(ocr_result) -> tuple[str, float]:
    """Extract text and confidence from OCR result."""
    lines = getattr(ocr_result, "text_lines", [])
    if not lines:
        return "", 0.0
    text = "\n".join(l.text for l in lines if l.text and l.text.strip())
    conf = float(np.mean([l.confidence for l in lines]))
    return text, conf


def process_document(file_path: str, include_debug: bool = False):
    """
    OCR entry point: process a document (PDF or image) and extract structured data.
    
    Args:
        file_path: Path to document file
        include_debug: If True, return (results, debug_pages); else just results
    
    Returns:
        List of page outputs with OCR text, tables, confidence scores
    """
    raw_pages = load_document(file_path)
    results = []
    debug_pages = []

    for page_num, raw_img in enumerate(raw_pages, start=1):
        preprocessed_img = preprocess(raw_img)
        pil_image = Image.fromarray(preprocessed_img[:, :, ::-1])
        ocr_result = _run_ocr(pil_image)
        full_text, avg_confidence = _build_text(ocr_result)

        tables, table_bboxes = extract_tables(preprocessed_img, ocr_result=ocr_result, page_index=page_num - 1)
        filtered_lines = mask_table_regions_from_text(ocr_result, table_bboxes)
        body_text = rebuild_body_text(filtered_lines)

        stamps = detect_circular_stamps(raw_img)
        page_output = build_page_output(page_num, full_text, avg_confidence, tables=tables)
        page_output["body_text_no_tables"] = body_text
        page_output["table_bboxes"] = [list(map(int, bbox)) for bbox in table_bboxes]

        if stamps:
            page_output["stamp_regions"] = [
                {"x": int(x), "y": int(y), "radius": int(r)}
                for x, y, r in stamps
            ]

        results.append(page_output)

        if include_debug:
            debug_pages.append({
                "page": page_num,
                "raw_image": raw_img,
                "preprocessed_image": preprocessed_img,
                "table_bboxes": [list(map(int, bbox)) for bbox in table_bboxes],
                "table_count": len(tables),
            })

    if include_debug:
        return results, debug_pages
    return results


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10: KIE ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def extract(ocr_output: list[dict], doc_id: str = None) -> dict:
    """
    KIE entry point: extract key information from OCR output.
    
    Args:
        ocr_output: List of page dicts from process_document()
        doc_id: Optional document identifier
    
    Returns:
        Structured document with extracted fields, doc type, etc.
    """
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


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 11: EVALUATION METRICS
# ═════════════════════════════════════════════════════════════════════════════

def _norm(s: str) -> str:
    """Normalize string for comparison."""
    return re.sub(r"\s+", " ", str(s or "").lower()).strip()


def _ref_norm(s: str) -> str:
    """Normalize reference codes."""
    return re.sub(r"[-\s]+", "/", _norm(s)).strip("/")


def _edit_distance(a: list, b: list) -> int:
    """Compute Levenshtein distance."""
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
    """Calculate character error rate."""
    h, r = list(_norm(hyp)), list(_norm(ref))
    if not r:
        return 0.0 if not h else 1.0
    return min(_edit_distance(h, r) / len(r), 1.0)


def calc_wer(hyp: str, ref: str) -> float:
    """Calculate word error rate."""
    h, r = _norm(hyp).split(), _norm(ref).split()
    if not r:
        return 0.0 if not h else 1.0
    return min(_edit_distance(h, r) / len(r), 1.0)


def calc_levenshtein(hyp: str, ref: str) -> int:
    """Calculate raw Levenshtein distance (character-level, unnormalized)."""
    return _edit_distance(list(_norm(hyp)), list(_norm(ref)))


def calc_text_accuracy(hyp: str, ref: str) -> float:
    """Calculate text accuracy: 1 - CER, clamped to [0, 1]."""
    return max(0.0, 1.0 - calc_cer(hyp, ref))


def calc_exact_match(pred: str, gold: str, field: str = "") -> float:
    """Calculate exact match for a field."""
    if not gold:
        return float("nan")
    p = _ref_norm(pred) if "ref" in field else _norm(pred)
    g = _ref_norm(gold) if "ref" in field else _norm(gold)
    return 1.0 if p == g else 0.0


def calc_token_prf(pred: str, gold: str) -> tuple[float, float, float]:
    """Calculate token-level precision, recall, F1."""
    if not gold:
        return float("nan"), float("nan"), float("nan")
    p_toks = _norm(pred).split()
    g_toks = _norm(gold).split()
    if not p_toks and not g_toks:
        return 1.0, 1.0, 1.0
    if not p_toks or not g_toks:
        return 0.0, 0.0, 0.0
    common = sum((Counter(p_toks) & Counter(g_toks)).values())
    prec = common / len(p_toks)
    rec = common / len(g_toks)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def calc_field_accuracy(pred: str, gold: str, threshold: float = 0.5) -> float:
    """Calculate field-level accuracy."""
    if not gold:
        return float("nan")
    _, _, f1 = calc_token_prf(pred, gold)
    return 1.0 if (f1 == f1 and f1 >= threshold) else 0.0


def calc_iou_text(pred: str, gold: str) -> float:
    """Calculate text-based IoU as bounding-box quality proxy."""
    if not gold:
        return float("nan")
    p_set = set(_norm(pred).split())
    g_set = set(_norm(gold).split())
    if not p_set and not g_set:
        return 1.0
    inter = len(p_set & g_set)
    union = len(p_set | g_set)
    return inter / union if union > 0 else 0.0


def _safe_mean(vals: list) -> float:
    """Compute mean, ignoring NaN values."""
    clean = [v for v in vals if v == v]
    return sum(clean) / len(clean) if clean else float("nan")


def _fmt(v) -> str:
    """Format value for display."""
    if v != v:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 12: EVALUATION HARNESS
# ═════════════════════════════════════════════════════════════════════════════

def _load_gt() -> dict:
    """Load ground truth data from CSV."""
    gt = {}
    with open(GT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = re.search(r"doc_(\d+)", str(row.get("filename", "")))
            if m:
                gt[m.group(1)] = row
    return gt


def _gt_full_text(row: dict) -> str:
    """Concatenate all GT fields into full text."""
    parts = [
        row.get("Source", ""),
        row.get("Destination", ""),
        row.get("Date", ""),
        row.get("Ref", ""),
        row.get("Objet", ""),
        row.get("Pj", ""),
        row.get("Content", ""),
        row.get("Copie", ""),
        row.get("Table", ""),
    ]
    return " ".join(p.strip() for p in parts if p and str(p).strip())


KIE_FIELDS = {
    "sender":     "Source",
    "receiver":   "Destination",
    "date":       "Date",
    "ref_header": "Ref",
    "objet":      "Objet",
    "pj":         "Pj",
    "body":       "Content",
}


def evaluate(max_docs: int | None = None):
    """
    Main evaluation function: run OCR and KIE on documents and compute metrics.
    
    Args:
        max_docs: Limit number of documents to evaluate (for testing)
    
    Outputs:
        - full_ocr_eval.csv: Per-document OCR metrics
        - full_kie_eval.csv: Per-document × field KIE metrics
        - Summary tables printed to console
    """
    print("Loading pipeline…")
    gt_data = _load_gt()

    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    if max_docs:
        pdf_files = pdf_files[:max_docs]

    total = len(pdf_files)
    print(f"Found {total} document(s) to evaluate.\n")

    ocr_rows = []
    kie_rows = []

    ocr_acc = defaultdict(list)
    kie_acc = defaultdict(lambda: defaultdict(list))
    # per-approach aggregation: approach -> field -> metric -> list
    approach_acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for idx, pdf_path in enumerate(pdf_files, 1):
        doc_name = pdf_path.name
        m = re.search(r"doc_(\d+)", doc_name)
        doc_num = m.group(1) if m else "???"
        gt_row = gt_data.get(doc_num, {})
        gt_full = _gt_full_text(gt_row) if gt_row else ""

        print(f"[{idx:3d}/{total}] {doc_name}", end=" … ", flush=True)

        t0 = time.perf_counter()
        try:
            ocr_out = process_document(str(pdf_path))
        except Exception as exc:
            print(f"OCR ERROR: {exc}")
            continue
        ocr_time = time.perf_counter() - t0

        ocr_text = " ".join(p.get("raw_text", "") for p in ocr_out)
        ocr_conf = sum(p.get("confidence", 0.0) for p in ocr_out) / max(len(ocr_out), 1)

        if gt_full:
            doc_cer = calc_cer(ocr_text, gt_full)
            doc_wer = calc_wer(ocr_text, gt_full)
            doc_lev = calc_levenshtein(ocr_text, gt_full)
            doc_tacc = calc_text_accuracy(ocr_text, gt_full)
        else:
            doc_cer = doc_wer = doc_lev = doc_tacc = float("nan")

        ocr_row = {
            "doc": doc_name,
            "doc_num": doc_num,
            "ocr_conf": round(ocr_conf, 4),
            "ocr_time_s": round(ocr_time, 3),
            "CER": _fmt(doc_cer),
            "WER": _fmt(doc_wer),
            "Levenshtein": _fmt(doc_lev),
            "Text_Accuracy": _fmt(doc_tacc),
        }
        ocr_rows.append(ocr_row)

        if gt_full:
            ocr_acc["CER"].append(doc_cer)
            ocr_acc["WER"].append(doc_wer)
            ocr_acc["Levenshtein"].append(doc_lev)
            ocr_acc["Text_Accuracy"].append(doc_tacc)
            ocr_acc["ocr_time_s"].append(ocr_time)

        t1 = time.perf_counter()
        try:
            kie_out = extract(ocr_out, doc_id=pdf_path.stem)
        except Exception as exc:
            print(f"KIE ERROR: {exc}")
            kie_out = {"pages": []}
        kie_time = time.perf_counter() - t1

        fields = kie_out["pages"][0]["fields"] if kie_out.get("pages") else {}

        # If available, detailed candidate info for each field
        field_details = None
        if kie_out.get("pages") and kie_out["pages"][0].get("field_details"):
            field_details = kie_out["pages"][0]["field_details"]

        for kie_field, gt_col in KIE_FIELDS.items():
            pred = fields.get(kie_field) or ""
            gold = gt_row.get(gt_col, "") or ""

            em = calc_exact_match(pred, gold, kie_field)
            prec, rec, f1 = calc_token_prf(pred, gold)
            facc = calc_field_accuracy(pred, gold)
            iou = calc_iou_text(pred, gold)
            kie_cer = calc_cer(pred, gold)

            def _text_sim(a, b):
                a = (a or "").strip().lower(); b = (b or "").strip().lower()
                if not a or not b: return 0.0
                return SequenceMatcher(None, a, b).ratio()

            fuzzy_score = _text_sim(pred, gold)

            # Record ensemble (selected) prediction
            kie_row = {
                "doc": doc_name,
                "doc_num": doc_num,
                "field": kie_field,
                "approach": "ensemble",
                "predicted": pred[:120],
                "ground_truth": gold[:120],
                "kie_time_s": round(kie_time, 3),
                "Exact_Match": _fmt(em),
                "Precision": _fmt(prec),
                "Recall": _fmt(rec),
                "F1": _fmt(f1),
                "Field_Accuracy": _fmt(facc),
                "IoU": _fmt(iou),
                "FuzzyScore": _fmt(fuzzy_score),
                "KIE_CER": _fmt(kie_cer),
            }
            kie_rows.append(kie_row)

            if gold:
                kie_acc[kie_field]["Exact_Match"].append(em)
                kie_acc[kie_field]["Precision"].append(prec)
                kie_acc[kie_field]["Recall"].append(rec)
                kie_acc[kie_field]["F1"].append(f1)
                kie_acc[kie_field]["Field_Accuracy"].append(facc)
                kie_acc[kie_field]["IoU"].append(iou)
                # per-approach (ensemble)
                approach_acc["ensemble"][kie_field]["Exact_Match"].append(em)
                approach_acc["ensemble"][kie_field]["Precision"].append(prec)
                approach_acc["ensemble"][kie_field]["Recall"].append(rec)
                approach_acc["ensemble"][kie_field]["F1"].append(f1)
                approach_acc["ensemble"][kie_field]["Field_Accuracy"].append(facc)
                approach_acc["ensemble"][kie_field]["IoU"].append(iou)

            # Also evaluate individual candidate strategies (regex, heuristic, fuzzy, ...)
            if field_details and field_details.get(kie_field):
                candidates = field_details[kie_field].get("candidates", [])
                for cand in candidates:
                    cval = cand.get("value") or ""
                    strategy = cand.get("strategy") or "unknown"
                    em_c = calc_exact_match(cval, gold, kie_field)
                    prec_c, rec_c, f1_c = calc_token_prf(cval, gold)
                    facc_c = calc_field_accuracy(cval, gold)
                    iou_c = calc_iou_text(cval, gold)
                    kie_cer_c = calc_cer(cval, gold)
                    fuzzy_c = cand.get("strategy_score") if cand.get("strategy_score") is not None else _text_sim(cval, gold)
                    cand_row = {
                        "doc": doc_name,
                        "doc_num": doc_num,
                        "field": kie_field,
                        "approach": strategy,
                        "approach_score": cand.get("score"),
                        "predicted": (cval or "")[:120],
                        "ground_truth": gold[:120],
                        "kie_time_s": round(kie_time, 3),
                        "Exact_Match": _fmt(em_c),
                        "Precision": _fmt(prec_c),
                        "Recall": _fmt(rec_c),
                        "F1": _fmt(f1_c),
                        "Field_Accuracy": _fmt(facc_c),
                        "IoU": _fmt(iou_c),
                        "FuzzyScore": _fmt(fuzzy_c),
                        "KIE_CER": _fmt(kie_cer_c),
                    }
                    kie_rows.append(cand_row)
                    if gold:
                        kie_acc[kie_field]["Exact_Match"].append(em_c)
                        kie_acc[kie_field]["Precision"].append(prec_c)
                        kie_acc[kie_field]["Recall"].append(rec_c)
                        kie_acc[kie_field]["F1"].append(f1_c)
                        kie_acc[kie_field]["Field_Accuracy"].append(facc_c)
                        kie_acc[kie_field]["IoU"].append(iou_c)
                        # per-approach aggregation
                        approach_acc[strategy][kie_field]["Exact_Match"].append(em_c)
                        approach_acc[strategy][kie_field]["Precision"].append(prec_c)
                        approach_acc[strategy][kie_field]["Recall"].append(rec_c)
                        approach_acc[strategy][kie_field]["F1"].append(f1_c)
                        approach_acc[strategy][kie_field]["Field_Accuracy"].append(facc_c)
                        approach_acc[strategy][kie_field]["IoU"].append(iou_c)

        cer_s = f"CER={doc_cer:.3f}" if doc_cer == doc_cer else "CER=n/a"
        wer_s = f"WER={doc_wer:.3f}" if doc_wer == doc_wer else "WER=n/a"
        tacc_s = f"Acc={doc_tacc:.1%}" if doc_tacc == doc_tacc else "Acc=n/a"
        avg_f1 = _safe_mean([
            _safe_mean(kie_acc[f]["F1"]) for f in KIE_FIELDS
            if kie_acc[f]["F1"]
        ])
        f1_s = f"KIE-F1={avg_f1:.3f}" if avg_f1 == avg_f1 else "KIE-F1=n/a"
        print(f"{cer_s}  {wer_s}  {tacc_s}  {f1_s}  OCR={ocr_time:.1f}s  KIE={kie_time:.2f}s")

    ocr_csv = ROOT / "full_ocr_eval.csv"
    kie_csv = ROOT / "full_kie_eval.csv"

    with open(ocr_csv, "w", newline="", encoding="utf-8-sig") as f:
        if ocr_rows:
            w = csv.DictWriter(f, fieldnames=ocr_rows[0].keys())
            w.writeheader(); w.writerows(ocr_rows)

    with open(kie_csv, "w", newline="", encoding="utf-8-sig") as f:
        if kie_rows:
            w = csv.DictWriter(f, fieldnames=kie_rows[0].keys())
            w.writeheader(); w.writerows(kie_rows)

    sep = "=" * 70

    print(f"\n{sep}")
    print("  OCR EVALUATION SUMMARY")
    print(sep)
    print(f"  {'Metric':<20} {'Mean':>10} {'Min':>10} {'Max':>10}")
    print("  " + "-" * 54)
    for metric in ["CER", "WER", "Levenshtein", "Text_Accuracy", "ocr_time_s"]:
        vals = ocr_acc[metric]
        if not vals:
            continue
        clean = [v for v in vals if v == v]
        if not clean:
            continue
        mean_v = sum(clean) / len(clean)
        print(f"  {metric:<20} {mean_v:>10.4f} {min(clean):>10.4f} {max(clean):>10.4f}")
    print(f"  Docs evaluated: {len(ocr_rows)}")
    print(f"  Saved → {ocr_csv}")

    print(f"\n{sep}")
    print("  KIE EVALUATION SUMMARY  (mean over all documents)")
    print(sep)
    header = f"  {'Field':<14} {'N':>4} {'ExactMatch':>11} {'Precision':>10} {'Recall':>8} {'F1':>8} {'FieldAcc':>9} {'IoU':>8}"
    print(header)
    print("  " + "-" * 76)

    macro = defaultdict(list)
    for field in KIE_FIELDS:
        fa = kie_acc[field]
        n = len([v for v in fa["F1"] if v == v])
        em = _safe_mean(fa["Exact_Match"])
        pr = _safe_mean(fa["Precision"])
        rc = _safe_mean(fa["Recall"])
        f1 = _safe_mean(fa["F1"])
        ac = _safe_mean(fa["Field_Accuracy"])
        iu = _safe_mean(fa["IoU"])
        for k, v in [("EM", em), ("P", pr), ("R", rc), ("F1", f1), ("Acc", ac), ("IoU", iu)]:
            if v == v:
                macro[k].append(v)
        print(
            f"  {field:<14} {n:>4} "
            f"{_fmt(em):>11} {_fmt(pr):>10} {_fmt(rc):>8} "
            f"{_fmt(f1):>8} {_fmt(ac):>9} {_fmt(iu):>8}"
        )

    print("  " + "-" * 76)
    print(
        f"  {'Macro avg':<14} {'':>4} "
        f"{_fmt(_safe_mean(macro['EM'])):>11} "
        f"{_fmt(_safe_mean(macro['P'])):>10} "
        f"{_fmt(_safe_mean(macro['R'])):>8} "
        f"{_fmt(_safe_mean(macro['F1'])):>8} "
        f"{_fmt(_safe_mean(macro['Acc'])):>9} "
        f"{_fmt(_safe_mean(macro['IoU'])):>8}"
    )
    # Per-approach summary (mean over fields)
    if approach_acc:
        print(f"\n{sep}")
        print("  KIE EVALUATION BY APPROACH  (mean over fields)")
        print(sep)
        header2 = f"  {'Approach':<12} {'Fields':>6} {'ExactMatch':>11} {'Precision':>10} {'Recall':>8} {'F1':>8} {'FieldAcc':>9} {'IoU':>8}"
        print(header2)
        print("  " + "-" * 76)
        for approach in sorted(approach_acc.keys()):
            mac = defaultdict(list)
            n_fields = 0
            for field in KIE_FIELDS:
                fa = approach_acc[approach].get(field, {})
                if not fa:
                    continue
                f1 = _safe_mean(fa.get("F1", []))
                if f1 != f1:
                    continue
                n_fields += 1
                em = _safe_mean(fa.get("Exact_Match", []))
                pr = _safe_mean(fa.get("Precision", []))
                rc = _safe_mean(fa.get("Recall", []))
                ac = _safe_mean(fa.get("Field_Accuracy", []))
                iu = _safe_mean(fa.get("IoU", []))
                for k, v in [("EM", em), ("P", pr), ("R", rc), ("F1", f1), ("Acc", ac), ("IoU", iu)]:
                    if v == v:
                        mac[k].append(v)
            print(
                f"  {approach:<12} {n_fields:>6} "
                f"{_fmt(_safe_mean(mac['EM'])):>11} {_fmt(_safe_mean(mac['P'])):>10} {_fmt(_safe_mean(mac['R'])):>8} "
                f"{_fmt(_safe_mean(mac['F1'])):>8} {_fmt(_safe_mean(mac['Acc'])):>9} {_fmt(_safe_mean(mac['IoU'])):>8}"
            )
        print("  " + "-" * 76)
    kie_time_all = [r['kie_time_s'] for r in kie_rows if r['field'] == 'body']
    if kie_time_all:
        print(f"\n  Mean KIE time/doc : {sum(kie_time_all)/len(kie_time_all):.3f}s")
    print(f"  Saved → {kie_csv}\n")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 13: MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Complete IDP Pipeline (OCR + KIE + Evaluation)")
    parser.add_argument("--docs", type=int, default=None,
                        help="Limit number of documents to evaluate (default: all)")
    args = parser.parse_args()
    evaluate(max_docs=args.docs)
