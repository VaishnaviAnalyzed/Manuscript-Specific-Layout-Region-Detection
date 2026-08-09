"""Basic tests for detector output and assignment constraints."""

from pathlib import Path

import cv2
import numpy as np

from app.detector import LABELS, ManuscriptLayoutDetector


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def make_test_page() -> np.ndarray:
    image = np.full((800, 600, 3), 238, dtype=np.uint8)
    cv2.rectangle(image, (30, 30), (570, 770), (190, 190, 190), 2)
    cv2.putText(image, "HEADER", (210, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2)
    for y in range(210, 590, 55):
        cv2.line(image, (120, y), (490, y), (25, 25, 25), 8)
        cv2.line(image, (140, y + 18), (470, y + 18), (40, 40, 40), 5)
    cv2.putText(image, "12", (280, 735), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2)
    return image


def test_detector_returns_valid_regions() -> None:
    image = make_test_page()
    detector = ManuscriptLayoutDetector()
    regions = detector.detect(image)

    assert regions
    assert any(region.label == "main_text" for region in regions)
    for region in regions:
        assert region.label in LABELS
        assert 0 <= region.confidence <= 1
        assert 0 <= region.bbox.x_min <= region.bbox.x_max < image.shape[1]
        assert 0 <= region.bbox.y_min <= region.bbox.y_max < image.shape[0]


def test_annotation_does_not_change_input() -> None:
    image = make_test_page()
    original = image.copy()
    detector = ManuscriptLayoutDetector()
    annotated = detector.annotate(image, detector.detect(image))

    assert np.array_equal(image, original)
    assert not np.array_equal(annotated, original)


def test_palm_leaf_sample_is_not_split_into_false_footer_or_side_text() -> None:
    detector = ManuscriptLayoutDetector()
    _, regions = detector.detect_file(
        PROJECT_ROOT / "data" / "test_images" / "palm_leaf_sample.png"
    )

    assert sum(region.label == "main_text" for region in regions) == 1
    assert sum(region.label == "header" for region in regions) >= 1
    assert sum(region.label == "footer" for region in regions) >= 1
    assert all(region.label != "side_text" for region in regions)


def test_margin_credit_lines_are_header_and_footer_on_paper_sample() -> None:
    detector = ManuscriptLayoutDetector()
    _, regions = detector.detect_file(
        PROJECT_ROOT / "data" / "test_images" / "paper_sample.png"
    )
    labels = [region.label for region in regions]

    assert "main_text" in labels
    assert "header" in labels
    assert "footer" in labels
