"""
Table and Signature Viewer
===========================
Extracts and visualizes all detected tables and signatures from documents.
Generates an HTML report and PNG images for inspection.

Usage:
    python table_signature_viewer.py [--docs N] [--output DIR]
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import csv
import time
import importlib.util
import re
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
KIE_PATH = ROOT / "KEY_INFORMATION_EXTRACTION_MODULE"
OCR_PATH = ROOT / "OCR_MODULE"
DOCS_DIR = ROOT / "documents"
GT_CSV = DOCS_DIR / "generated_documents.csv"
OUTPUT_DIR = ROOT / "table_signature_reports"
IMG_DIR = OUTPUT_DIR / "images"

sys.path.insert(0, str(KIE_PATH))
sys.path.insert(0, str(OCR_PATH))


def _load_pipeline():
    """Load OCR and KIE modules."""
    spec = importlib.util.spec_from_file_location("ocr_main", OCR_PATH / "main.py")
    ocr_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ocr_mod)

    spec2 = importlib.util.spec_from_file_location("kie_field_extractor", KIE_PATH / "kie_field_extractor.py")
    kie_mod = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(kie_mod)

    return ocr_mod, kie_mod


def _ensure_dirs():
    """Create output directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)


def _draw_bboxes_on_image(image: Image.Image, bboxes: list, label: str = "") -> Image.Image:
    """
    Draw bounding boxes on an image.
    
    Args:
        image: PIL Image
        bboxes: list of [x1, y1, x2, y2] coordinates
        label: label to display with the boxes
    
    Returns:
        Image with bboxes drawn
    """
    if not bboxes:
        return image
    
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]
    
    for idx, bbox in enumerate(bboxes):
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            color = colors[idx % len(colors)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            if label:
                draw.text((x1, y1 - 10), f"{label}_{idx}", fill=color)
        except Exception:
            pass
    
    return img_copy


def _extract_tables_from_doc(ocr_mod, pdf_path: Path, doc_name: str) -> list[dict]:
    """
    Extract all tables from a document.
    
    Returns list of dict with:
        - page_num: page number (1-indexed)
        - table_content: extracted table data (CSV string)
        - bbox: bounding box [x1, y1, x2, y2]
    """
    tables_data = []
    
    try:
        raw_pages = ocr_mod.load_document(str(pdf_path))
        if not raw_pages:
            return tables_data
        
        for page_idx, page_img in enumerate(raw_pages):
            # Extract tables from this page
            if hasattr(ocr_mod, 'extract_tables_from_image'):
                tables = ocr_mod.extract_tables_from_image(page_img)
            else:
                # Fallback: try table_extractor module
                try:
                    spec = importlib.util.spec_from_file_location("table_extractor", OCR_PATH / "table_extractor.py")
                    te_mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(te_mod)
                    tables = te_mod.extract_tables(page_img) if hasattr(te_mod, 'extract_tables') else []
                except Exception:
                    tables = []
            
            for table_idx, table in enumerate(tables):
                table_content = ""
                bbox = [0, 0, 100, 100]
                
                # Try to extract content and bbox
                if isinstance(table, dict):
                    table_content = table.get("content", "")
                    bbox = table.get("bbox", bbox)
                else:
                    table_content = str(table)
                
                tables_data.append({
                    "page_num": page_idx + 1,
                    "table_idx": table_idx + 1,
                    "table_content": table_content,
                    "bbox": bbox,
                })
    
    except Exception as e:
        print(f"  Table extraction error: {e}")
    
    return tables_data


def _extract_signatures_from_doc(kie_mod, raw_text: str, page_confidence: float) -> list[dict]:
    """
    Extract signature-related fields from KIE.
    
    Returns list of detected fields that might be signatures.
    """
    signatures = []
    
    try:
        # Try to call a signature extraction helper if available
        if hasattr(kie_mod, '_extract_signature_detailed'):
            selected, candidates = kie_mod._extract_signature_detailed(raw_text, page_confidence)
            if selected or candidates:
                signatures.append({
                    "value": selected,
                    "candidates": candidates,
                })
        
        # Also check for "Copie" field which might relate to signatures
        if hasattr(kie_mod, '_extract_copie_detailed'):
            selected, candidates = kie_mod._extract_copie_detailed(raw_text, page_confidence)
            if selected or candidates:
                signatures.append({
                    "field": "Copie",
                    "value": selected,
                    "candidates": candidates,
                })
    
    except Exception as e:
        pass  # Signature extraction is optional
    
    return signatures


def _generate_html_report(report_data: list[dict]) -> str:
    """Generate an HTML report of all tables and signatures."""
    html_lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>Table & Signature Report</title>",
        "<style>",
        "  body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }",
        "  .document { background: white; margin: 20px 0; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        "  .doc-title { font-size: 20px; font-weight: bold; color: #333; margin-bottom: 15px; }",
        "  .section { margin: 20px 0; }",
        "  .section-title { font-size: 16px; font-weight: bold; color: #0066cc; border-bottom: 2px solid #0066cc; padding-bottom: 5px; }",
        "  .table-item { background: #f9f9f9; margin: 10px 0; padding: 10px; border-left: 4px solid #0066cc; }",
        "  .sig-item { background: #fff9e6; margin: 10px 0; padding: 10px; border-left: 4px solid #ff9900; }",
        "  .table-content { background: white; border: 1px solid #ddd; padding: 10px; overflow-x: auto; margin-top: 5px; }",
        "  table { border-collapse: collapse; width: 100%; }",
        "  th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "  th { background: #f0f0f0; }",
        "  img { max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ddd; }",
        "  .image-container { text-align: center; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>📊 Table & Signature Detection Report</h1>",
        f"<p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>",
    ]
    
    for doc_data in report_data:
        doc_name = doc_data.get("document", "Unknown")
        tables = doc_data.get("tables", [])
        signatures = doc_data.get("signatures", [])
        page1_img = doc_data.get("page1_img", "")
        
        html_lines.extend([
            f"<div class='document'>",
            f"<div class='doc-title'>📄 {doc_name}</div>",
        ])
        
        # Page 1 preview
        if page1_img and Path(page1_img).exists():
            rel_path = Path(page1_img).relative_to(OUTPUT_DIR)
            html_lines.extend([
                "<div class='section'>",
                "<div class='section-title'>📸 Page 1 Preview</div>",
                "<div class='image-container'>",
                f"<img src='{rel_path}' alt='Page 1'>",
                "</div>",
                "</div>",
            ])
        
        # Tables section
        if tables:
            html_lines.extend([
                "<div class='section'>",
                f"<div class='section-title'>📋 Tables Detected ({len(tables)})</div>",
            ])
            
            for table_idx, table in enumerate(tables):
                page_num = table.get("page_num", "?")
                table_content = table.get("table_content", "")
                
                html_lines.append(f"<div class='table-item'>")
                html_lines.append(f"<strong>Table {table_idx + 1}</strong> (Page {page_num})")
                
                if table_content:
                    html_lines.append("<div class='table-content'>")
                    if isinstance(table_content, str) and "," in table_content:
                        # Try to render as HTML table
                        try:
                            lines = table_content.strip().split("\n")
                            html_lines.append("<table>")
                            for line in lines:
                                cells = line.split(",")
                                html_lines.append("<tr>")
                                for cell in cells:
                                    html_lines.append(f"<td>{cell.strip()}</td>")
                                html_lines.append("</tr>")
                            html_lines.append("</table>")
                        except Exception:
                            html_lines.append(f"<pre>{table_content}</pre>")
                    else:
                        html_lines.append(f"<pre>{table_content}</pre>")
                    html_lines.append("</div>")
                
                html_lines.append("</div>")
            
            html_lines.append("</div>")
        else:
            html_lines.append("<div class='section'><div class='section-title'>📋 Tables</div><p><em>No tables detected</em></p></div>")
        
        # Signatures section
        if signatures:
            html_lines.extend([
                "<div class='section'>",
                f"<div class='section-title'>✍️ Signatures Detected ({len(signatures)})</div>",
            ])
            
            for sig_idx, sig in enumerate(signatures):
                field = sig.get("field", "Signature")
                value = sig.get("value", "")
                candidates = sig.get("candidates", [])
                
                html_lines.append(f"<div class='sig-item'>")
                html_lines.append(f"<strong>{field}</strong>")
                if value:
                    html_lines.append(f"<div><strong>Detected:</strong> {value[:200]}</div>")
                if candidates:
                    html_lines.append(f"<div><strong>Candidates:</strong> {len(candidates)} option(s)</div>")
                html_lines.append("</div>")
            
            html_lines.append("</div>")
        
        html_lines.append("</div>")
    
    html_lines.extend([
        "</body>",
        "</html>",
    ])
    
    return "\n".join(html_lines)


def main(max_docs: int | None = None):
    """Main entry point."""
    print("Table & Signature Viewer")
    print("=" * 70)
    
    _ensure_dirs()
    print(f"Loading pipeline…")
    ocr_mod, kie_mod = _load_pipeline()
    
    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    if max_docs:
        pdf_files = pdf_files[:max_docs]
    
    total = len(pdf_files)
    print(f"Found {total} document(s).\n")
    
    report_data = []
    
    for idx, pdf_path in enumerate(pdf_files, 1):
        doc_name = pdf_path.name
        m = re.search(r"doc_(\d+)", doc_name)
        doc_num = m.group(1) if m else "???"
        
        print(f"[{idx:3d}/{total}] {doc_name}", end=" … ", flush=True)
        start_time = time.perf_counter()
        
        # Extract page 1 and OCR
        try:
            raw_pages = ocr_mod.load_document(str(pdf_path))
            if not raw_pages:
                print("No pages found")
                continue
            
            first_page = raw_pages[0]
            preprocessed = ocr_mod.preprocess(first_page)
            pil_page1 = Image.fromarray(cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB))
            
            # Save page 1 preview
            page1_img_path = IMG_DIR / f"{doc_num}_page1.png"
            pil_page1.save(page1_img_path)
            
            ocr_result = ocr_mod._run_ocr(pil_page1)
            page_text, page_confidence = ocr_mod._build_text(ocr_result)
        except Exception as exc:
            print(f"OCR ERROR: {exc}")
            continue
        
        # Extract tables
        tables = _extract_tables_from_doc(ocr_mod, pdf_path, doc_name)
        
        # Extract signatures
        signatures = _extract_signatures_from_doc(kie_mod, page_text, page_confidence)
        
        elapsed = time.perf_counter() - start_time
        
        report_data.append({
            "document": doc_name,
            "tables": tables,
            "signatures": signatures,
            "page1_img": str(page1_img_path),
        })
        
        print(f"{len(tables)} table(s), {len(signatures)} signature(s) — {elapsed:.1f}s")
    
    # Generate HTML report
    html_content = _generate_html_report(report_data)
    report_path = OUTPUT_DIR / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    # Generate CSV summary
    csv_rows = []
    for doc_data in report_data:
        csv_rows.append({
            "Document": doc_data["document"],
            "Tables Detected": len(doc_data["tables"]),
            "Signatures Detected": len(doc_data["signatures"]),
        })
    
    csv_path = OUTPUT_DIR / "summary.csv"
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["Document", "Tables Detected", "Signatures Detected"])
            w.writeheader()
            w.writerows(csv_rows)
    
    print(f"\n{'=' * 70}")
    print(f"Report saved → {report_path}")
    print(f"Summary CSV → {csv_path}")
    print(f"Images → {IMG_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract and view all tables and signatures")
    parser.add_argument("--docs", type=int, default=None, help="Limit number of documents")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()
    main(max_docs=args.docs)
