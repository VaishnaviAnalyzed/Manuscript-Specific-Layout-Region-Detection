# Manuscript Layout Region Detection

A Python application that detects manuscript page regions and labels them as
`header`, `footer`, `main_text`, `side_text`, or `filler`.

This repository includes:

- A FastAPI backend with a static web UI.
- A CLI for batch inference over individual files or folders.
- OpenCV-based region detection and annotated result generation.
- JSON metadata output and served result files.

## Project structure

```text
.
├── app/
│   ├── api.py          # FastAPI app and static frontend entry point
│   ├── config.py       # runtime directories, upload limits, allowed file types
│   ├── detector.py     # OpenCV detection pipeline and annotation logic
│   ├── schemas.py      # region and bounding-box data structures
│   └── service.py      # shared processing logic for API and CLI
├── data/test_images/   # sample manuscript images used by tests
├── docs/               # architecture and design notes
├── static/             # frontend HTML/CSS/JavaScript
├── storage/            # runtime upload and result storage
├── tests/              # pytest test suite
├── inference.py        # batch CLI entrypoint
├── requirements.txt    # runtime dependencies
└── requirements-dev.txt# development dependencies
```

## Requirements

- Python 3.13+
- Windows or any OS with OpenCV support
- `uvicorn`, `fastapi`, `opencv-python-headless`, `numpy`, `Pillow`

## Setup

If you already have the provided `myenv` environment, activate it before running
commands. To create a fresh virtual environment in the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

If you prefer to use the existing `myenv` folder:

```powershell
.\myenv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## Run the web application

Start the FastAPI server from the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api:app --reload
```

Or, if using `myenv`:

```powershell
.\myenv\Scripts\python.exe -m uvicorn app.api:app --reload
```

Open the UI at:

- `http://127.0.0.1:8000`

Interactive API documentation is available at:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/redoc`

## API endpoints

The backend exposes these routes:

- `GET /health`
  - Returns service status.
  - Example response: `{ "status": "ok", "service": "manuscript-layout-detector" }`
- `GET /api/classes`
  - Returns supported layout labels and their meanings.
- `POST /api/detect`
  - Accepts `multipart/form-data` with a single file field named `file`.
  - Supported file types: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`.
  - Maximum upload size: 20 MB.
  - Returns: `job_id`, `source_filename`, `image_size`, `region_count`, `regions`, `annotated_image_url`, and `metadata_url`.
- `GET /results/{filename}`
  - Serves generated annotated images and JSON metadata created by the API.

## Run batch CLI inference

The CLI entrypoint is `inference.py`.

Process a folder or a single image:

```powershell
.\.venv\Scripts\python.exe inference.py --input .\data\test_images --output .\results
```

```powershell
.\.venv\Scripts\python.exe inference.py --input .\page.png --output .\results
```

Allow searching nested folders with:

```powershell
.\.venv\Scripts\python.exe inference.py --input .\data\test_images --output .\results --recursive
```

## Storage behavior

The web app creates runtime files under:

- `storage/uploads/` — uploaded source images
- `storage/results/` — annotated images and JSON outputs

The CLI writes results to the user-specified `--output` folder.

**Notes**

- The detector uses a rule-based OpenCV baseline, not a trained neural network.
- It is designed for quick local prototyping and explainable document layout detection.
- The API and CLI share the same `ProcessingService` implementation.
- If no regions are found, the detector returns a single `main_text` fallback box.
