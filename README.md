# Manuscript Layout Region Detection

An end-to-end Python project that detects and labels manuscript page regions as
`header`, `footer`, `main_text`, `side_text`, or `filler`. It includes a batch CLI,
a FastAPI backend, a responsive browser interface, annotated output images, and
JSON metadata.

## Features

- Processes one image or a complete folder.
- Supports JPG, PNG, BMP, TIFF and WebP files with varied aspect ratios.
- Uses contrast enhancement and adaptive thresholding for uneven light, stains,
  faded ink and moderate bleed-through.
- Keeps original input images unchanged.
- Clips every bounding box to the page boundary.
- Writes annotated JPG images, per-image JSON, and a batch summary.
- Provides an upload UI and documented REST API.

## Project structure

```text
.
├── app/
│   ├── api.py          # FastAPI backend
│   ├── config.py       # paths and upload rules
│   ├── detector.py     # OpenCV detection pipeline
│   ├── schemas.py      # region data structures
│   └── service.py      # shared result-writing logic
├── data/test_images/   # sample manuscript pages
├── docs/               # design notes
├── static/             # HTML, CSS and JavaScript frontend
├── storage/            # runtime web uploads and results
├── tests/              # automated tests
├── inference.py        # batch command-line entry point
└── requirements.txt
```

## Setup on Windows

The repository already uses a local `.venv`. To recreate it from scratch:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

If PowerShell blocks activation, the environment can still be used directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Run the web application

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api:app --reload
```

Open <http://127.0.0.1:8000>. Interactive API documentation is available at
<http://127.0.0.1:8000/docs>.

### API endpoints

- `GET /health` — health check.
- `GET /api/classes` — supported classes and meanings.
- `POST /api/detect` — multipart upload using a field named `file`.
- `GET /results/{filename}` — generated annotated image or JSON metadata.

## Run batch CLI inference

The assignment command works with relative paths:

```powershell
.\.venv\Scripts\python.exe inference.py --input .\data\test_images --output .\results
```

Process one file:

```powershell
.\.venv\Scripts\python.exe inference.py --input .\page.png --output .\results
```

Search nested input folders by adding `--recursive`.

## Output format

For `page.png`, the CLI creates:

- `page_annotated.jpg`
- `page_predictions.json`
- `batch_summary.json`

Each detected region has a label, confidence, and original-image pixel bounds:

```json
{
  "label": "main_text",
  "confidence": 0.9,
  "bbox": {
    "x_min": 118,
    "y_min": 202,
    "x_max": 1470,
    "y_max": 932
  }
}
```

## Detection approach and rationale

The provided assignment does not include labeled training data or pretrained
weights, so this submission uses an explainable OpenCV baseline that runs
immediately and is easy for a junior developer to understand:

1. CLAHE improves local contrast.
2. Adaptive thresholding finds ink under uneven lighting.
3. Morphological operations join nearby characters into text bands.
4. Normalized position and shape rules assign the five layout classes.
5. Nearby bands are merged and boxes are clipped to image dimensions.

This is suitable as a reproducible baseline and application prototype. Semantic
distinctions such as English versus another script require labeled examples; for
production quality, fine-tune YOLO, Detectron2, or LayoutLMv3 on the five classes
and keep the existing API/service interface. More detail is in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The tests cover valid bounded predictions, input immutability, annotated output,
the health endpoint, and an end-to-end image upload.

## Reproducibility notes

- Dependency versions are pinned in `requirements.txt` and
  `requirements-dev.txt`.
- All paths in the CLI work relatively from the repository root.
- Input images are only read. The CLI writes to the selected output folder, and
  the web app stores a separate upload copy under `storage/uploads`.
- Generated runtime files are excluded by `.gitignore`.

## Known limitations

- The rules are a baseline, not a trained manuscript-specific model.
- Heavy overlap, severe page curvature and subtle pencil/ink differences may be
  classified incorrectly.
- Confidence values are heuristic quality indicators rather than calibrated
  model probabilities.

