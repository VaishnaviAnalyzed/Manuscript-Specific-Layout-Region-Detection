"""Explainable OpenCV baseline for manuscript layout-region detection.

This detector is deliberately simple: it finds ink components, groups them into
text bands, and classifies those bands from their location and shape.  A trained
detector can replace this class later without changing the API or CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.schemas import BoundingBox, Region


LABELS = ("header", "footer", "main_text", "side_text", "filler")
COLORS = {
    # OpenCV uses BGR channel order.
    "header": (246, 130, 49),
    "footer": (12, 88, 234),
    "main_text": (74, 163, 22),
    "side_text": (247, 85, 168),
    "filler": (8, 179, 234),
}


@dataclass
class DetectorSettings:
    """Tunable values for the rule-based detector."""

    max_processing_side: int = 1800
    minimum_component_area_ratio: float = 0.000006
    minimum_region_area_ratio: float = 0.0008


class ManuscriptLayoutDetector:
    """Detect header, footer, main text, side text and filler regions."""

    def __init__(self, settings: DetectorSettings | None = None) -> None:
        self.settings = settings or DetectorSettings()

    def detect_file(self, image_path: str | Path) -> tuple[np.ndarray, list[Region]]:
        """Load an image and return the original pixels plus detected regions."""

        image_path = Path(image_path)
        file_bytes = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return image, self.detect(image)

    def detect(self, image: np.ndarray) -> list[Region]:
        """Run the complete detection pipeline on a BGR image."""

        if image is None or image.size == 0:
            raise ValueError("The input image is empty.")

        original_height, original_width = image.shape[:2]
        working, scale = self._resize_for_processing(image)
        page_box = self._find_page_box(working)
        ink_mask = self._create_ink_mask(working, page_box)
        candidate_boxes = self._find_text_bands(ink_mask, page_box)
        regions = self._classify_and_merge(candidate_boxes, page_box, working.shape[:2])
        regions.extend(self._find_outside_margin_regions(working, page_box))

        if not regions:
            regions = [
                Region(
                    label="main_text",
                    confidence=0.35,
                    bbox=self._scale_box(page_box, 1.0 / scale, original_width, original_height),
                )
            ]
            return regions

        scaled_regions: list[Region] = []
        for region in regions:
            scaled_regions.append(
                Region(
                    label=region.label,
                    confidence=region.confidence,
                    bbox=self._scale_box(
                        region.bbox,
                        1.0 / scale,
                        original_width,
                        original_height,
                    ),
                )
            )
        return self._sort_regions(scaled_regions)

    def _find_outside_margin_regions(
        self, image: np.ndarray, page_box: BoundingBox
    ) -> list[Region]:
        """Find header/footer text above or below a detected substrate.

        Palm-leaf scans often place English collection credits on the blue scan
        background. They are outside the leaf, so they need a separate local-
        contrast pass instead of the normal page ink mask.
        """

        image_height, image_width = image.shape[:2]
        page_height = page_box.y_max - page_box.y_min + 1
        margin = max(4, int(page_height * 0.012))
        outside_ranges = [
            (0, max(0, page_box.y_min - margin)),
            (min(image_height, page_box.y_max + margin + 1), image_height),
        ]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel_size = max(9, int(min(image_height, image_width) * 0.023))
        if kernel_size % 2 == 0:
            kernel_size += 1
        blackhat_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )
        local_dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, blackhat_kernel)
        strong_ink = (local_dark >= 30).astype(np.uint8)

        margin_regions: list[Region] = []
        for range_index, (range_start, range_end) in enumerate(outside_ranges):
            if range_end - range_start < 4:
                continue
            band_mask = strong_ink[range_start:range_end, :]
            row_ink = np.count_nonzero(band_mask, axis=1)
            active_rows = np.flatnonzero(row_ink >= max(20, int(image_width * 0.05)))
            row_runs = self._group_indexes(active_rows, max_gap=2)

            for local_start, local_end in row_runs:
                run_height = local_end - local_start + 1
                if run_height < 3 or run_height > image_height * 0.10:
                    continue
                text_crop = band_mask[local_start : local_end + 1, :]
                column_ink = np.count_nonzero(text_crop, axis=0)
                strong_columns = np.flatnonzero(
                    column_ink >= max(2, int(run_height * 0.45))
                )
                if strong_columns.size == 0:
                    continue
                column_runs = self._group_indexes(
                    strong_columns,
                    max_gap=max(5, int(image_width * 0.012)),
                )
                x_min, x_max = max(
                    column_runs,
                    key=lambda run: run[1] - run[0],
                )
                strong_in_run = np.count_nonzero(
                    (strong_columns >= x_min) & (strong_columns <= x_max)
                )
                if strong_in_run / (x_max - x_min + 1) < 0.25:
                    continue
                width_ratio = (x_max - x_min + 1) / image_width
                center_x = (x_min + x_max) / (2 * image_width)
                if not (0.15 <= width_ratio <= 0.85 and 0.15 <= center_x <= 0.85):
                    continue
                label = "header" if range_index == 0 else "footer"
                margin_regions.append(
                    Region(
                        label,
                        0.82,
                        BoundingBox(
                        x_min,
                        range_start + local_start,
                        x_max,
                        range_start + local_end,
                        ),
                    )
                )
        return margin_regions

    def annotate(self, image: np.ndarray, regions: list[Region]) -> np.ndarray:
        """Draw labeled boxes on a copy, leaving the original image unchanged."""

        annotated = image.copy()
        image_height, image_width = annotated.shape[:2]
        line_width = max(2, int(min(image_width, image_height) / 450))
        font_scale = max(0.5, min(image_width, image_height) / 1200)

        for region in regions:
            box = region.bbox
            color = COLORS[region.label]
            cv2.rectangle(
                annotated,
                (box.x_min, box.y_min),
                (box.x_max, box.y_max),
                color,
                line_width,
            )
            text = f"{region.label} {region.confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_width
            )
            label_top = max(0, box.y_min - text_height - 10)
            label_right = min(image_width - 1, box.x_min + text_width + 10)
            cv2.rectangle(
                annotated,
                (box.x_min, label_top),
                (label_right, box.y_min),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                text,
                (box.x_min + 5, max(text_height, box.y_min - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                line_width,
                cv2.LINE_AA,
            )
        return annotated

    def _resize_for_processing(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = image.shape[:2]
        longest_side = max(height, width)
        if longest_side <= self.settings.max_processing_side:
            return image.copy(), 1.0
        scale = self.settings.max_processing_side / longest_side
        resized = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    def _find_page_box(self, image: np.ndarray) -> BoundingBox:
        """Estimate the manuscript substrate and fall back to the whole image."""

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        _, bright_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel_size = max(7, int(min(width, height) * 0.025))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        minimum_area = width * height * 0.30
        valid = [contour for contour in contours if cv2.contourArea(contour) >= minimum_area]
        if not valid:
            return BoundingBox(0, 0, width - 1, height - 1)

        contour = max(valid, key=cv2.contourArea)
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width < width * 0.55 or box_height < height * 0.35:
            return BoundingBox(0, 0, width - 1, height - 1)
        return BoundingBox(x, y, x + box_width - 1, y + box_height - 1)

    def _create_ink_mask(self, image: np.ndarray, page_box: BoundingBox) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
        block_size = max(15, int(min(image.shape[:2]) / 28))
        if block_size % 2 == 0:
            block_size += 1
        mask = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            11,
        )

        # Ignore the dark area outside a palm leaf or paper page. A filled contour
        # handles rounded and damaged edges better than a rectangular crop.
        _, bright_page = cv2.threshold(
            cv2.GaussianBlur(gray, (9, 9), 0),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        substrate_kernel_size = max(7, int(min(image.shape[:2]) * 0.025))
        substrate_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (substrate_kernel_size, substrate_kernel_size)
        )
        bright_page = cv2.morphologyEx(bright_page, cv2.MORPH_CLOSE, substrate_kernel)
        page_contours, _ = cv2.findContours(
            bright_page, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        page_mask = np.zeros_like(mask)
        if page_contours:
            largest_page = max(page_contours, key=cv2.contourArea)
            if cv2.contourArea(largest_page) >= mask.shape[0] * mask.shape[1] * 0.30:
                cv2.drawContours(page_mask, [largest_page], -1, 255, -1)
            else:
                cv2.rectangle(
                    page_mask,
                    (page_box.x_min, page_box.y_min),
                    (page_box.x_max, page_box.y_max),
                    255,
                    -1,
                )
        else:
            cv2.rectangle(
                page_mask,
                (page_box.x_min, page_box.y_min),
                (page_box.x_max, page_box.y_max),
                255,
                -1,
            )
        mask = cv2.bitwise_and(mask, page_mask)
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, clean_kernel)

        # Printed borders and scan edges should not be joined to nearby text.
        page_width = page_box.x_max - page_box.x_min + 1
        page_height = page_box.y_max - page_box.y_min + 1
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(15, int(page_height * 0.28)))
        )
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(15, int(page_width * 0.75)), 1)
        )
        vertical_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel)
        horizontal_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel)
        # Remove a small halo as adaptive thresholding makes old rules/borders thick.
        line_halo = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        vertical_lines = cv2.dilate(vertical_lines, line_halo, iterations=1)
        horizontal_lines = cv2.dilate(horizontal_lines, line_halo, iterations=1)
        page_lines = cv2.bitwise_or(vertical_lines, horizontal_lines)
        return cv2.subtract(mask, page_lines)

    def _find_text_bands(
        self, mask: np.ndarray, page_box: BoundingBox
    ) -> list[BoundingBox]:
        page_width = page_box.x_max - page_box.x_min + 1
        page_height = page_box.y_max - page_box.y_min + 1
        page_crop = mask[
            page_box.y_min : page_box.y_max + 1,
            page_box.x_min : page_box.x_max + 1,
        ]

        # A horizontal projection separates text lines even when characters touch.
        row_ink = np.count_nonzero(page_crop, axis=1)
        median_ink = float(np.median(row_ink))
        high_ink = float(np.percentile(row_ink, 95))
        row_threshold = max(
            5,
            int(page_width * 0.012),
            int(median_ink + 0.22 * max(0.0, high_ink - median_ink)),
        )
        active_rows = np.flatnonzero(row_ink >= row_threshold)
        row_runs = self._group_indexes(active_rows, max_gap=max(2, int(page_height * 0.012)))

        minimum_area = page_width * page_height * self.settings.minimum_region_area_ratio
        candidates: list[BoundingBox] = []
        for row_start, row_end in row_runs:
            line_crop = page_crop[row_start : row_end + 1, :]
            column_ink = np.count_nonzero(line_crop, axis=0)
            active_columns = np.flatnonzero(column_ink > 0)
            column_runs = self._group_indexes(
                active_columns, max_gap=max(3, int(page_width * 0.018))
            )
            for column_start, column_end in column_runs:
                width = column_end - column_start + 1
                height = row_end - row_start + 1
                if width * height < minimum_area:
                    continue
                if width >= page_width * 0.60 and height <= page_height * 0.012:
                    continue  # scan boundary, ruling line, or page frame
                x_min = page_box.x_min + column_start
                y_min = page_box.y_min + row_start
                candidates.append(
                    BoundingBox(
                        x_min,
                        y_min,
                        page_box.x_min + column_end,
                        page_box.y_min + row_end,
                    )
                )

        return candidates

    def _classify_and_merge(
        self,
        boxes: list[BoundingBox],
        page_box: BoundingBox,
        image_shape: tuple[int, int],
    ) -> list[Region]:
        if not boxes:
            return []

        page_width = page_box.x_max - page_box.x_min + 1
        page_height = page_box.y_max - page_box.y_min + 1
        page_area = page_width * page_height
        grouped: dict[str, list[BoundingBox]] = {label: [] for label in LABELS}
        body_box_keys = self._find_dominant_body(boxes, page_box)
        body_boxes = [
            box
            for box in boxes
            if (box.x_min, box.y_min, box.x_max, box.y_max) in body_box_keys
        ]
        typical_body_height = float(
            np.median([box.y_max - box.y_min + 1 for box in body_boxes])
        ) if body_boxes else 0.0
        body_top = min((box.y_min for box in body_boxes), default=page_box.y_min)
        body_bottom = max((box.y_max for box in body_boxes), default=page_box.y_max)

        for box in boxes:
            relative_x = ((box.x_min + box.x_max) / 2 - page_box.x_min) / page_width
            relative_y = ((box.y_min + box.y_max) / 2 - page_box.y_min) / page_height
            width_ratio = (box.x_max - box.x_min + 1) / page_width
            height_ratio = (box.y_max - box.y_min + 1) / page_height
            area_ratio = ((box.x_max - box.x_min + 1) * (box.y_max - box.y_min + 1)) / page_area

            box_key = (box.x_min, box.y_min, box.x_max, box.y_max)
            if box_key in body_box_keys:
                label = "main_text"
            elif (
                typical_body_height > 0
                and height_ratio < (typical_body_height / page_height) * 0.65
                and width_ratio >= 0.18
                and 0.20 <= relative_x <= 0.80
            ):
                # A short centered line above/below the body is a running header
                # or footer. Similar small elements inside the body remain filler.
                if box.y_max < body_top:
                    label = "header"
                elif box.y_min > body_bottom:
                    label = "footer"
                else:
                    label = "filler"
            elif relative_y < 0.18 and height_ratio < 0.16:
                label = "header"
            elif relative_y > 0.82 and height_ratio < 0.16:
                label = "footer"
            elif (relative_x < 0.17 or relative_x > 0.83) and width_ratio < 0.33:
                label = "side_text"
            elif area_ratio < 0.006 and width_ratio < 0.12:
                label = "filler"
            else:
                label = "main_text"
            grouped[label].append(box)

        regions: list[Region] = []
        for label, label_boxes in grouped.items():
            for merged_box, member_count in self._merge_nearby_boxes(label_boxes, page_width, page_height):
                box_area = (
                    (merged_box.x_max - merged_box.x_min + 1)
                    * (merged_box.y_max - merged_box.y_min + 1)
                )
                size_score = min(1.0, box_area / max(1, page_area * 0.08))
                member_score = min(1.0, member_count / 5)
                confidence = min(0.94, 0.52 + 0.20 * size_score + 0.18 * member_score)
                if label == "filler":
                    confidence = min(confidence, 0.72)
                regions.append(Region(label, confidence, self._clip_box(merged_box, image_shape)))

        return regions

    def _find_dominant_body(
        self, boxes: list[BoundingBox], page_box: BoundingBox
    ) -> set[tuple[int, int, int, int]]:
        """Find the largest sequence of central, regularly spaced text lines."""

        page_width = page_box.x_max - page_box.x_min + 1
        page_height = page_box.y_max - page_box.y_min + 1
        central_boxes = []
        for box in boxes:
            width_ratio = (box.x_max - box.x_min + 1) / page_width
            center_x = ((box.x_min + box.x_max) / 2 - page_box.x_min) / page_width
            if width_ratio >= 0.20 and 0.12 <= center_x <= 0.88:
                central_boxes.append(box)
        if not central_boxes:
            return set()

        typical_height = float(
            np.median([box.y_max - box.y_min + 1 for box in central_boxes])
        )
        central_boxes = [
            box
            for box in central_boxes
            if box.y_max - box.y_min + 1 >= max(3, typical_height * 0.60)
        ]
        vertical_gap = page_height * 0.100
        ordered = sorted(central_boxes, key=lambda box: (box.y_min, box.x_min))
        clusters: list[list[BoundingBox]] = []
        for box in ordered:
            matching_cluster = None
            for cluster in clusters:
                cluster_top = min(item.y_min for item in cluster)
                cluster_bottom = max(item.y_max for item in cluster)
                if box.y_min <= cluster_bottom + vertical_gap and box.y_max >= cluster_top - vertical_gap:
                    matching_cluster = cluster
                    break
            if matching_cluster is None:
                clusters.append([box])
            else:
                matching_cluster.append(box)

        dominant = max(
            clusters,
            key=lambda cluster: (
                len(cluster),
                sum((box.x_max - box.x_min + 1) * (box.y_max - box.y_min + 1) for box in cluster),
            ),
        )
        body_top = min(box.y_min for box in dominant)
        body_bottom = max(box.y_max for box in dominant)
        # Add short stroke fragments that fall inside the dominant text block.
        body_left = min(box.x_min for box in dominant)
        body_right = max(box.x_max for box in dominant)
        body_boxes = [
            box
            for box in boxes
            if box.y_max >= body_top
            and box.y_min <= body_bottom
            and box.x_max >= body_left
            and box.x_min <= body_right
        ]
        return {(box.x_min, box.y_min, box.x_max, box.y_max) for box in body_boxes}

    @staticmethod
    def _group_indexes(indexes: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
        """Turn sorted active indexes into runs and bridge small empty gaps."""

        if indexes.size == 0:
            return []
        runs: list[tuple[int, int]] = []
        start = previous = int(indexes[0])
        for value in indexes[1:]:
            current = int(value)
            if current - previous > max_gap:
                runs.append((start, previous))
                start = current
            previous = current
        runs.append((start, previous))
        return runs

    def _merge_nearby_boxes(
        self, boxes: list[BoundingBox], page_width: int, page_height: int
    ) -> list[tuple[BoundingBox, int]]:
        """Merge text bands that belong to the same visual region."""

        if not boxes:
            return []

        pending = sorted(boxes, key=lambda box: (box.y_min, box.x_min))
        groups: list[list[BoundingBox]] = []
        horizontal_gap = page_width * 0.045
        vertical_gap = page_height * 0.100

        for box in pending:
            matching_group = None
            for group in groups:
                union = self._union_boxes(group)
                x_overlap = min(union.x_max, box.x_max) - max(union.x_min, box.x_min)
                y_overlap = min(union.y_max, box.y_max) - max(union.y_min, box.y_min)
                close_x = box.x_min <= union.x_max + horizontal_gap and box.x_max >= union.x_min - horizontal_gap
                close_y = box.y_min <= union.y_max + vertical_gap and box.y_max >= union.y_min - vertical_gap
                if (x_overlap >= 0 and close_y) or (y_overlap >= 0 and close_x):
                    matching_group = group
                    break
            if matching_group is None:
                groups.append([box])
            else:
                matching_group.append(box)

        return [(self._union_boxes(group), len(group)) for group in groups]

    @staticmethod
    def _union_boxes(boxes: list[BoundingBox]) -> BoundingBox:
        return BoundingBox(
            min(box.x_min for box in boxes),
            min(box.y_min for box in boxes),
            max(box.x_max for box in boxes),
            max(box.y_max for box in boxes),
        )

    @staticmethod
    def _clip_box(box: BoundingBox, image_shape: tuple[int, int]) -> BoundingBox:
        height, width = image_shape
        return BoundingBox(
            max(0, min(width - 1, int(box.x_min))),
            max(0, min(height - 1, int(box.y_min))),
            max(0, min(width - 1, int(box.x_max))),
            max(0, min(height - 1, int(box.y_max))),
        )

    @staticmethod
    def _scale_box(
        box: BoundingBox, scale: float, image_width: int, image_height: int
    ) -> BoundingBox:
        return BoundingBox(
            max(0, min(image_width - 1, round(box.x_min * scale))),
            max(0, min(image_height - 1, round(box.y_min * scale))),
            max(0, min(image_width - 1, round(box.x_max * scale))),
            max(0, min(image_height - 1, round(box.y_max * scale))),
        )

    @staticmethod
    def _sort_regions(regions: list[Region]) -> list[Region]:
        label_order = {label: index for index, label in enumerate(LABELS)}
        return sorted(
            regions,
            key=lambda region: (
                label_order[region.label],
                region.bbox.y_min,
                region.bbox.x_min,
            ),
        )
