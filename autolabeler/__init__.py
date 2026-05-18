"""AutoLabeler-GS: GroundingDINO + SAM2 자동 라벨링 도구."""

from .config import AutoLabelConfig
from .datatypes import DetectionBox, ImageAnnotationResult, InstanceAnnotation

__all__ = [
    "AutoLabelConfig",
    "DetectionBox",
    "InstanceAnnotation",
    "ImageAnnotationResult",
]

__version__ = "0.1.0"
