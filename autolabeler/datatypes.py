"""AutoLabeler-GS에서 사용하는 공용 데이터 클래스 정의."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DetectionBox:
    """GroundingDINO가 만든 단일 박스 결과."""

    class_id: int
    class_name: str
    prompt: str
    score: float
    box_xyxy: Tuple[float, float, float, float]


@dataclass
class InstanceAnnotation:
    """이미지 하나에 속하는 단일 객체 어노테이션."""

    image_id: int
    image_path: str
    class_id: int
    class_name: str
    prompt: str
    score: float
    box_xyxy: Tuple[float, float, float, float]
    polygon_xy: List[List[float]]
    area: float
    accepted: bool = True


@dataclass
class ImageAnnotationResult:
    """이미지 하나에 대한 전체 자동 라벨링 결과."""

    image_id: int
    image_path: str
    width: int
    height: int
    instances: List[InstanceAnnotation] = field(default_factory=list)
    preview_path: Optional[str] = None
