import numpy as np
from PIL import Image
from surya.detection import DetectionPredictor
from surya.layout import LayoutPredictor
from surya.recognition import RecognitionPredictor
from surya.table_rec import TableRecPredictor
from surya.foundation import FoundationPredictor
from surya.settings import settings

_predictors = None


def _load():
    global _predictors
    if _predictors is None:
        _predictors = {
            "detection": DetectionPredictor(),
            "recognition": RecognitionPredictor(
                FoundationPredictor(checkpoint=settings.RECOGNITION_MODEL_CHECKPOINT)
            ),
            "layout": LayoutPredictor(
                FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
            ),
            "table_rec": TableRecPredictor(),
        }


def get_predictors():
    _load()
    return _predictors


def detect_layout(pil_image: Image.Image) -> dict:
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
