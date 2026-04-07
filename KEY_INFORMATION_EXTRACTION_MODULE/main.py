import json
import sys
from pathlib import Path
import numpy as np

ocr_module_path = str(Path(__file__).parent.parent / "OCR_MODULE")
sys.path.insert(0, ocr_module_path)
 
import importlib.util
spec = importlib.util.spec_from_file_location(
    "ocr_main",
    Path(ocr_module_path) / "main.py"
)
ocr_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ocr_main)
process_document = ocr_main.process_document

sys.path.insert(0, str(Path(__file__).parent))

from extractor import extract


def run_pipeline(file_path: str, doc_id: str = None) -> dict:
    ocr_output = process_document(file_path)
    doc_id = doc_id or Path(file_path).stem
    return extract(ocr_output, doc_id=doc_id)
class _JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_document>")
        sys.exit(1)


    result = run_pipeline(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False, indent=2, cls=_JSONEncoder))

