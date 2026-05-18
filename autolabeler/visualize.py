"""오버레이 프리뷰 이미지 생성."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from .datatypes import InstanceAnnotation


def _color_for_class(class_id: int) -> tuple:
    """class_id 에 결정적으로 매핑되는 BGR 색."""

    rng = np.random.default_rng(class_id * 9973 + 17)
    color = rng.integers(low=64, high=255, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def draw_overlay(
    image: Image.Image,
    instances: Iterable[InstanceAnnotation],
    output_path: Path,
    alpha: float = 0.4,
) -> Path:
    """PIL RGB 이미지를 받아 OpenCV BGR 로 변환 후 박스/폴리곤을 그려 PNG 저장."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()

    for inst in instances:
        color = _color_for_class(inst.class_id)

        # 폴리곤 채우기 (반투명)
        if inst.polygon_xy and len(inst.polygon_xy) >= 3:
            pts = np.array(inst.polygon_xy, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [pts], color)

    blended = cv2.addWeighted(overlay, alpha, bgr, 1.0 - alpha, 0)

    for inst in instances:
        color = _color_for_class(inst.class_id)

        # 폴리곤 외곽선
        if inst.polygon_xy and len(inst.polygon_xy) >= 3:
            pts = np.array(inst.polygon_xy, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(blended, [pts], True, color, 2, cv2.LINE_AA)

        # 박스
        x1, y1, x2, y2 = [int(v) for v in inst.box_xyxy]
        cv2.rectangle(blended, (x1, y1), (x2, y2), color, 2)

        # 라벨
        label = f"{inst.class_name} {inst.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(0, y1 - 4)
        cv2.rectangle(
            blended,
            (x1, max(0, ly - th - 4)),
            (x1 + tw + 4, ly),
            color,
            -1,
        )
        cv2.putText(
            blended,
            label,
            (x1 + 2, ly - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), blended)
    return output_path
