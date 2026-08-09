# Architecture

The project uses one detection service from two entry points:

```text
Browser -> FastAPI API --\
                         -> ProcessingService -> OpenCV detector -> JPG + JSON
CLI --------------------/
```

## Modules

- `app/detector.py`: image preprocessing, candidate discovery, classification and annotation.
- `app/service.py`: saves annotated images and JSON metadata.
- `app/api.py`: validates uploads and exposes the browser/API endpoints.
- `inference.py`: discovers images and runs batch processing from the command line.
- `static/`: plain HTML, CSS and JavaScript frontend.
- `tests/`: unit and integration tests.

## Detection stages

1. Resize very large pages for predictable memory use.
2. Estimate the light page/substrate boundary.
3. Improve local contrast with CLAHE and create an adaptive binary ink mask.
4. Connect nearby ink components into text bands.
5. Classify bands using normalized page position, size and shape.
6. Merge nearby bands into regions, clip every box to the image, and scale it back to the original resolution.
7. Draw the boxes on a copy and save both annotated JPG and JSON metadata.

## Model limitations

This is a reproducible, no-training baseline. It can localize visual regions across
different page sizes, but script-level meaning such as English versus historical
script or pencil versus faded ink needs a labeled dataset and a trained detector.
For production accuracy, annotate the five assignment classes and replace
`ManuscriptLayoutDetector` with a fine-tuned object detector while preserving its
`detect_file`, `detect`, and `annotate` interface.

