import cv2
import numpy as np
from PIL import Image
from surya.common.surya.schema import TaskNames
from layout import get_predictors

_trocr_processor = None
_trocr_model = None


def _load_trocr():
    global _trocr_processor, _trocr_model
    if _trocr_model is None:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        _trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        _trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")


#def is_handwritten(region_img: np.ndarray) -> bool:
    h, w = region_img.shape[:2]
    # stamps and cachets are wide and short — skip them
    if w > h * 3:
        return False
    # too small to be a meaningful handwritten region
    if h < 30 or w < 30:
        return False

    gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels < 5:
        return False
    areas = stats[1:, cv2.CC_STAT_AREA]
    # normalize variance by region size to avoid large regions always triggering
    normalized_var = float(np.var(areas)) / (h * w)
    return normalized_var > 0.5
def is_handwritten(region_img: np.ndarray) -> bool:
    return False  # CPU too slow for TrOCR in demo


def ocr_full_page(pil_image: Image.Image) -> tuple[str, float]:
    predictors = get_predictors()
    results = predictors["recognition"](
        [pil_image],
        det_predictor=predictors["detection"],
        task_names=[TaskNames.ocr_with_boxes],
        sort_lines=True
    )
    result = results[0]
    if not result.text_lines:
        return "", 0.0
    text = "\n".join([line.text for line in result.text_lines])
    avg_conf = float(np.mean([line.confidence for line in result.text_lines]))
    return text, avg_conf


def ocr_region_printed(pil_image: Image.Image) -> tuple[str, float]:
    predictors = get_predictors()
    results = predictors["recognition"](
        [pil_image],
        det_predictor=predictors["detection"],
        task_names=[TaskNames.ocr_with_boxes]
    )
    result = results[0]
    if not result.text_lines:
        return "", 0.0
    text = "\n".join([line.text for line in result.text_lines])
    avg_conf = float(np.mean([line.confidence for line in result.text_lines]))
    return text, avg_conf


def ocr_region_handwritten(region_img: np.ndarray) -> tuple[str, float]:
    import torch
    _load_trocr()
    pil_image = Image.fromarray(cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB)).convert("RGB")
    pixel_values = _trocr_processor(pil_image, return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = _trocr_model.generate(pixel_values)
    text = _trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text, 0.70


def ocr_region(region_img: np.ndarray) -> tuple[str, float]:
    pil_image = Image.fromarray(cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB))
    if is_handwritten(region_img):
        return ocr_region_handwritten(region_img)
    return ocr_region_printed(pil_image)
