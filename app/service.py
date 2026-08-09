"""Shared file-processing service for both the web API and CLI."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2

from app.detector import LABELS, ManuscriptLayoutDetector


class ProcessingService:
    """Save annotated images and machine-readable prediction metadata."""

    def __init__(self, detector: ManuscriptLayoutDetector | None = None) -> None:
        self.detector = detector or ManuscriptLayoutDetector()

    def process(self, input_path: Path, output_directory: Path, result_name: str | None = None) -> dict:
        input_path = Path(input_path)
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        image, regions = self.detector.detect_file(input_path)
        annotated = self.detector.annotate(image, regions)
        safe_stem = result_name or input_path.stem
        annotated_path = output_directory / f"{safe_stem}_annotated.jpg"
        metadata_path = output_directory / f"{safe_stem}_predictions.json"

        success, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not success:
            raise RuntimeError("Could not encode the annotated result image.")
        encoded.tofile(str(annotated_path))

        height, width = image.shape[:2]
        metadata = {
            "source_image": input_path.name,
            "image_size": {"width": width, "height": height},
            "model": {
                "name": "opencv-layout-baseline",
                "version": "1.0.0",
                "classes": list(LABELS),
            },
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "region_count": len(regions),
            "regions": [region.to_dict() for region in regions],
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        metadata["files"] = {
            "annotated_image": str(annotated_path),
            "metadata": str(metadata_path),
        }
        return metadata

