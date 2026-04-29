# Intelligent Document Processing (IDP)

OCR + Key Information Extraction for French administrative letters.

This repository contains:
- A Streamlit app for end-to-end document processing
- An OCR module for page rendering, preprocessing, OCR, and table extraction
- A KIE module for extracting structured fields from OCR text
- Evaluation scripts for OCR and KIE quality analysis

## What the Pipeline Does

Given a PDF or image document, the system:
1. Loads and renders pages (PDF supported)
2. Applies adaptive preprocessing
3. Runs OCR with Surya predictors
4. Extracts tables (img2table primary, TATR fallback)
5. Builds per-page OCR output
6. Extracts structured KIE fields
7. Returns document-level JSON + optional CSV exports

## Extracted KIE Fields

For each page, the KIE module extracts:
- sender
- receiver
- date
- ref_header
- ref_body
- objet
- pj
- body

Document-level type output:
- doc_type: lettre_administrative
- doc_subtype: demande | transmission | information | autre

## Installation

Python 3.10+ is recommended.

1. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies

```bash
pip install -r OCR_MODULE/requirements.txt
pip install -r KEY_INFORMATION_EXTRACTION_MODULE/requirements.txt
pip install streamlit pypdfium2 polars
```

Optional API dependencies:

```bash
pip install fastapi uvicorn
```

## Run the Streamlit App

```bash
streamlit run app.py
```

Open http://localhost:8501.

Notes:
- First run is slower because model weights are downloaded and cached.
- Supported upload formats: PDF, JPG, JPEG, PNG.

## Direct OCR Module Usage

Run OCR on one document directly from CLI:

```bash
python OCR_MODULE/main.py <path_to_document>
```

The command prints OCR JSON to stdout.

## Evaluation Workflows

### 1) OCR Text Evaluation

Compares OCR text with ground-truth text reconstructed from documents/generated_documents.csv.

```bash
python ocr_text_evaluator.py
python ocr_text_evaluator.py --docs 5
python ocr_text_evaluator.py --all-pages
python ocr_text_evaluator.py --with-tables
```

Output file:
- ocr_text_evaluation_results.csv

### 2) End-to-End OCR + KIE Evaluation

Runs full OCR pipeline and KIE extraction, then computes OCR and per-field KIE metrics.

```bash
python evaluator.py
python evaluator.py --docs 10
python evaluator.py --no-tables
python evaluator.py --with-tables
```

Output file:
- evaluation_results.csv

### 3) KIE Evaluation from Existing OCR CSV

Uses OCR text already stored in ocr_text_evaluation_results.csv, then evaluates only KIE extraction quality.

```bash
python kie_evaluator_from_ocr_csv.py
python kie_evaluator_from_ocr_csv.py --docs 20
```

Output file:
- kie_evaluation_results.csv

### 4) KIE Error Analysis Utilities

Generate error-focused reports from kie_evaluation_results.csv:

```bash
python kie_error_analysis.py
python kie_error_analysis.py --top 30
python kie_error_pattern_summary.py
```

Output files:
- kie_error_analysis_top20.txt
- kie_error_pattern_summary.txt

### 5) Evaluation Viewer (Streamlit)

Interactive side-by-side comparison of extracted fields vs ground truth:

```bash
streamlit run eval_viewer.py
```

## Production Warmup and API Startup Example

Preload heavy models once:

```bash
python preload_models.py
```

Run the FastAPI startup example:

```bash
uvicorn fastapi_startup_example:app --host 0.0.0.0 --port 8000
```

Health endpoint:
- GET /health

## Input and Output Overview

Input sources:
- App uploads (PDF/JPG/JPEG/PNG)
- Evaluation corpus in documents/
- Ground truth CSV in documents/generated_documents.csv

Main generated artifacts:
- evaluation_results.csv
- ocr_text_evaluation_results.csv
- kie_evaluation_results.csv
- kie_error_analysis_top20.txt
- kie_error_pattern_summary.txt

App exports per processed document:
- Full JSON
- Full CSV (all extracted fields)
- Summary CSV (doc_id, page, objet, body)

## Project Structure

```text
.
|- app.py
|- eval_viewer.py
|- evaluator.py
|- ocr_text_evaluator.py
|- kie_evaluator_from_ocr_csv.py
|- kie_error_analysis.py
|- kie_error_pattern_summary.py
|- preload_models.py
|- fastapi_startup_example.py
|- documents/
|  |- generated_documents.csv
|- OCR_MODULE/
|  |- main.py
|  |- preprocessor.py
|  |- layout.py
|  |- ocr_engine.py
|  |- table_extractor.py
|  |- output_builder.py
|  |- requirements.txt
|- KEY_INFORMATION_EXTRACTION_MODULE/
|  |- extractor.py
|  |- kie_doc_type.py
|  |- kie_field_extractor.py
|  |- kie_output_builder.py
|  |- requirements.txt
```

## Known Limitations

- Handwritten-heavy content is difficult in CPU-only mode.
- Degraded scans and strong rotations can reduce OCR confidence.
- KIE rules are tuned for French administrative letter templates and may not generalize to unrelated layouts.
- First-run latency is expected due to model downloads.
