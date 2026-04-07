import json
import sys
import cv2
import numpy as np
from PIL import Image
from preprocessor import preprocess_document, load_document, detect_circular_stamps
from layout import get_predictors
from ocr_engine import is_handwritten, ocr_region_handwritten
from surya.common.surya.schema import TaskNames
from table_extractor import (
    extract_tables,
    mask_table_regions_from_text,
    rebuild_body_text
)
from output_builder import build_page_output


class _JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _run_ocr(pil_image: Image.Image):
    predictors = get_predictors()
    results = predictors["recognition"](
        [pil_image],
        det_predictor=predictors["detection"],
        task_names=[TaskNames.ocr_with_boxes],
        sort_lines=True
    )
    return results[0]


def process_document(file_path: str) -> list[dict]:
    preprocessed_pages = preprocess_document(file_path)
    raw_pages = load_document(file_path)
    results = []

    for page_num, (page_img, raw_img) in enumerate(
        zip(preprocessed_pages, raw_pages), start=1
    ):
        pil_image = Image.fromarray(cv2.cvtColor(page_img, cv2.COLOR_BGR2RGB))

        # run surya OCR once — result reused for tables and text
        ocr_result = _run_ocr(pil_image)

        # extract tables — returns tables + bboxes for masking
        tables, table_bboxes = extract_tables(
            raw_img, ocr_result=ocr_result, page_index=page_num - 1
        )

        # build clean text excluding table regions
        non_table_lines = mask_table_regions_from_text(ocr_result, table_bboxes)
        full_text = rebuild_body_text(non_table_lines)
        avg_confidence = float(np.mean(
            [l.confidence for l in non_table_lines]
        )) if non_table_lines else 0.0

        # detect circular stamps on raw image
        stamps = detect_circular_stamps(raw_img)

        # check fixed regions for handwriting
        handwritten_extras = []
        h, w = page_img.shape[:2]
        check_regions = [
            (w // 2, 0, w, h // 4),       # top-right: date/ref area
            (0, int(h * 0.85), w, h),      # bottom: signature area
        ]
        for (rx1, ry1, rx2, ry2) in check_regions:
            crop = page_img[ry1:ry2, rx1:rx2]
            if crop.size > 0 and is_handwritten(crop):
                hw_text, _ = ocr_region_handwritten(crop)
                if hw_text.strip():
                    handwritten_extras.append({
                        "bbox": [rx1, ry1, rx2, ry2],
                        "text": hw_text
                    })

        page_output = build_page_output(
            page_num, full_text, avg_confidence, tables
        )

        if handwritten_extras:
            page_output["handwritten_regions"] = handwritten_extras

        if stamps:
            page_output["stamp_regions"] = [
                {"x": int(x), "y": int(y), "radius": int(r)}
                for x, y, r in stamps
            ]
            page_output["notes"] = (
                "Circular stamp(s) detected — stamp text is not reliably "
                "extractable due to radial text orientation"
            )

        results.append(page_output)

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_document>")
        sys.exit(1)

    output = process_document(sys.argv[1])
    print(json.dumps(output, ensure_ascii=False, indent=2, cls=_JSONEncoder))
