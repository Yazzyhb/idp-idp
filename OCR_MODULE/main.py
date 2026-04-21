import json
import sys
import cv2
import numpy as np
from PIL import Image
from preprocessor import load_document, preprocess, detect_circular_stamps
from layout import get_predictors
from surya.common.surya.schema import TaskNames
from table_extractor import extract_tables, mask_table_regions_from_text, rebuild_body_text
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


def _build_text(ocr_result) -> tuple[str, float]:
    lines = ocr_result.text_lines
    if not lines:
        return "", 0.0
    text = "\n".join(l.text for l in lines if l.text and l.text.strip())
    conf = float(np.mean([l.confidence for l in lines]))
    return text, conf


def process_document(file_path: str, include_debug: bool = False):
    raw_pages = load_document(file_path)
    results = []
    debug_pages = []

    for page_num, raw_img in enumerate(raw_pages, start=1):
        preprocessed_img = preprocess(raw_img)
        pil_image = Image.fromarray(cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2RGB))
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_document>")
        sys.exit(1)

    output = process_document(sys.argv[1])
    print(json.dumps(output, ensure_ascii=False, indent=2, cls=_JSONEncoder))
