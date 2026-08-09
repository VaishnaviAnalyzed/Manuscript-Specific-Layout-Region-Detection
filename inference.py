"""Command-line batch inference for manuscript images.

Example:
    python inference.py --input ./data/test_images --output ./results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.config import ALLOWED_EXTENSIONS
from app.service import ProcessingService


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect layout regions in one manuscript image or a folder of images."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input image or image folder")
    parser.add_argument("--output", required=True, type=Path, help="Folder for generated results")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also search subfolders when the input is a directory",
    )
    return parser.parse_args()


def find_images(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in ALLOWED_EXTENSIONS else []
    if not input_path.is_dir():
        return []

    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    return sorted(
        path for path in iterator if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    )


def main() -> int:
    args = parse_arguments()
    images = find_images(args.input, args.recursive)
    if not images:
        print(f"No supported images found at: {args.input}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    service = ProcessingService()
    completed: list[dict] = []
    failed: list[dict[str, str]] = []

    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] Processing {image_path.name}")
        try:
            result = service.process(image_path, args.output)
            completed.append(
                {
                    "source_image": image_path.name,
                    "region_count": result["region_count"],
                    "annotated_image": result["files"]["annotated_image"],
                    "metadata": result["files"]["metadata"],
                }
            )
        except Exception as error:
            failed.append({"source_image": image_path.name, "error": str(error)})
            print(f"  Failed: {error}", file=sys.stderr)

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "total": len(images),
        "completed": completed,
        "failed": failed,
    }
    summary_path = args.output / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Finished: {len(completed)} succeeded, {len(failed)} failed")
    print(f"Summary: {summary_path}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())

