"""FastAPI backend and static frontend entry point."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import (
    ALLOWED_EXTENSIONS,
    MAX_UPLOAD_BYTES,
    PROJECT_ROOT,
    RESULT_DIR,
    STATIC_DIR,
    UPLOAD_DIR,
    create_runtime_directories,
)
from app.service import ProcessingService


create_runtime_directories()
app = FastAPI(
    title="Manuscript Layout Region Detection",
    description="Detect layout regions in historical manuscript images.",
    version="1.0.0",
)
service = ProcessingService()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "manuscript-layout-detector"}


@app.post("/api/detect")
async def detect_image(file: UploadFile = File(...)) -> dict:
    """Process one uploaded manuscript image."""

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Use: {allowed}")

    file_content = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    if not file_content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(file_content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The image is larger than 20 MB.")

    job_id = uuid4().hex
    upload_path = UPLOAD_DIR / f"{job_id}{suffix}"
    upload_path.write_bytes(file_content)

    try:
        result = service.process(upload_path, RESULT_DIR, job_id)
    except ValueError as error:
        upload_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Image processing failed.") from error

    return {
        "job_id": job_id,
        "source_filename": file.filename,
        "image_size": result["image_size"],
        "region_count": result["region_count"],
        "regions": result["regions"],
        "annotated_image_url": f"/results/{job_id}_annotated.jpg",
        "metadata_url": f"/results/{job_id}_predictions.json",
    }


@app.get("/api/classes")
def classes() -> dict[str, list[dict[str, str]]]:
    return {
        "classes": [
            {"name": "header", "meaning": "Top-margin text or running title"},
            {"name": "footer", "meaning": "Bottom-margin text, catchword or page number"},
            {"name": "main_text", "meaning": "Main manuscript body"},
            {"name": "side_text", "meaning": "Marginal note or commentary"},
            {"name": "filler", "meaning": "Decoration, English text or pencil text"},
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=True)

