"""Typed data structures returned by the detector."""

from dataclasses import asdict, dataclass


@dataclass
class BoundingBox:
    """Bounding-box coordinates in pixels."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class Region:
    """One detected layout region."""

    label: str
    confidence: float
    bbox: BoundingBox

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox.to_dict(),
        }

