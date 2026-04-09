# Intelligent Document Processing System
### Algérie Poste — French Administrative Documents
**Version 1.0** | OCR + Key Information Extraction

---





## Installation

**Step 1 — Install dependencies**


```
pip install -r OCR_MODULE/requirements.txt
pip install -r KEY_INFORMATION_EXTRACTION_MODULE/requirements.txt
pip install streamlit pypdfium2 polars
```

**Step 2 — Launch the application**

```
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

> **Note:** The first launch takes several minutes as AI models are downloaded and cached automatically. Subsequent launches are fast.

---

## How to Use

### Step 1 — Upload a Document
In the left sidebar, click **Browse files** and select a document.

**Supported formats:** PDF, JPG, JPEG, PNG


---

### Step 2 — Process the Document
Click the **Process Document** button in the sidebar.

The system will:
1. Render each page at high resolution
2. Apply image preprocessing (deskew, denoise, binarize)
3. Run OCR to detect and recognize all text
4. Detect and extract any tables present
5. Extract key information fields using linguistic rules
6. Detect handwritten signatures

Processing time is approximately **2–4 minutes per page** on CPU.

---

### Step 3 — Review Results

Results are organized across **4 tabs:**

#### 📋 Extracted Fields
Displays all extracted structured information:

| Field | Description |
|---|---|
| Document Type | Automatically classified (lettre administrative + subtype) |
| Sender | Organization or person at the top of the document |
| Receiver | Addressee(s) of the letter |
| Date | Date of the document |
| Reference (Header) | Reference code in the document header |
| Reference (Body) | Reference cited within the body text |
| Object | Subject line of the letter |
| Body | Main body text of the letter |
| P.J | Attachments listed |
| OCR Confidence | Percentage confidence of the OCR engine |

#### 📊 Tables
Any tables detected in the document are displayed as interactive dataframes with row/column counts and extraction confidence.

#### 📝 Raw Text
The complete raw OCR output for each page, useful for verifying extraction accuracy.

#### 🔧 Full JSON
The complete structured output of the entire pipeline in JSON format, including all OCR metadata and extracted fields.

---

### Step 4 — Export Results

Three download buttons are available at the top of the results:

| Button | Format | Contents |
|---|---|---|
| ⬇️ Full JSON | `.json` | Complete pipeline output — all OCR data, tables, fields, confidence scores |
| ⬇️ Full CSV | `.csv` | One row per page — all extracted fields, suitable for database import |
| ⬇️ Summary CSV | `.csv` | One row per page — document ID, object, and body text only |

> CSV files are encoded in **UTF-8 with BOM** for correct display of French characters in Microsoft Excel.

---

## Extracted Fields — Technical Details

The KIE module extracts the following fields from `lettre_administrative` documents:

| Field | Extraction Method | Notes |
|---|---|---|
| `sender` | First 4 non-reference lines | Stops at first reference code or REF: label |
| `ref_header` | Regex on first 15 lines | Matches patterns like `DGAP/DSOCG/Nº 99512026` |
| `ref_body` | Regex on full text | Matches `Réf :` or `REF :` labels |
| `date` | Regex after `Alger, le` | Returns null if date field is blank |
| `receiver` | Regex on Messieurs/Mesdames/Monsieur lines | Collects up to 4 consecutive lines |
| `objet` | Regex after `Objet :` label | Single line |
| `pj` | Regex after `P.J :` label | Single line |
| `body` | Between opening and closing formula | Starts at Faisant suite / J'ai l'honneur / etc. |

**Document subtypes detected:** `demande`, `transmission`, `information`, `autre`

---



## Known Limitations

- **Handwritten dates** are not extracted — the date field will show *(blank/handwritten)*
- **Rotated or heavily degraded** scans may produce lower OCR confidence
- **Arabic text** in headers is intentionally excluded from the sender field
- Processing speed depends on document page count and CPU performance

---

## Project Structure

```
THE IDP SYSTEM/
├── app.py                              ← Main Streamlit application
├── OCR_MODULE/
│   ├── main.py                         ← OCR pipeline entry point
│   ├── preprocessor.py                 ← Image preprocessing (deskew, denoise, binarize)
│   ├── layout.py                       ← Surya model loader
│   ├── ocr_engine.py                   ← OCR engine (Surya + TrOCR)
│   ├── table_extractor.py              ← Table detection and extraction
│   └── output_builder.py               ← Page output formatter
├── KEY_INFORMATION_EXTRACTION_MODULE/
│   ├── extractor.py                    ← KIE pipeline entry point
│   ├── kie_field_extractor.py          ← Regex field extractors
│   ├── kie_doc_type.py                 ← Document type classifier
│   └── kie_output_builder.py           ← KIE output formatter
├── CLASSIFICATION MODULE/              ← In development
└── TOPIC MODELING MODULE/              ← In development
```



*Algérie Poste — Intelligent Document Processing System — v1.0*
