"""마스크 후처리, 폴리곤 변환, 좌표 유틸 등 모음."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


def clean_mask(
    mask: np.ndarray,
    kernel_size: int = 3,
    apply_morphology: bool = True,
) -> np.ndarray:
    """이진 마스크에 모폴로지 open/close 를 적용한 uint8 마스크 반환."""

    if mask is None:
        raise ValueError("mask is None")

    if mask.dtype != np.uint8:
        m = (mask > 0).astype(np.uint8) * 255
    else:
        m = (mask > 0).astype(np.uint8) * 255

    if apply_morphology and kernel_size and kernel_size >= 1:
        k = max(1, int(kernel_size))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)

    return m


def mask_to_polygons(
    mask: np.ndarray,
    epsilon_ratio: float = 0.003,
    min_area: int = 100,
) -> List[List[List[float]]]:
    """이진 마스크 -> 외곽 폴리곤 리스트.

    각 폴리곤은 [[x, y], [x, y], ...] 형식이고 면적 필터와 approxPolyDP 를 거친다.
    """

    if mask is None:
        return []

    if mask.dtype != np.uint8:
        binary = (mask > 0).astype(np.uint8) * 255
    else:
        binary = (mask > 0).astype(np.uint8) * 255

    height, width = binary.shape[:2]
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    polygons: List[List[List[float]]] = []
    for cnt in contours:
        if cnt is None or len(cnt) < 3:
            continue
        area = float(cv2.contourArea(cnt))
        if area < float(min_area):
            continue
        perimeter = float(cv2.arcLength(cnt, True))
        eps = max(1.0, perimeter * float(epsilon_ratio))
        approx = cv2.approxPolyDP(cnt, eps, True)
        if approx is None or len(approx) < 3:
            continue
        poly = []
        for p in approx:
            x = max(0.0, min(float(p[0][0]), float(width - 1)))
            y = max(0.0, min(float(p[0][1]), float(height - 1)))
            poly.append([x, y])
        if len(poly) < 3 or polygon_area(poly) < float(min_area):
            continue
        polygons.append(poly)

    return polygons


def polygon_area(polygon: Sequence[Sequence[float]]) -> float:
    """단순 다각형의 면적(Shoelace 공식)."""

    n = len(polygon)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def clamp_box_xyxy(
    box: Sequence[float], width: int, height: int
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(x1), float(width - 1)))
    y1 = max(0.0, min(float(y1), float(height - 1)))
    x2 = max(0.0, min(float(x2), float(width - 1)))
    y2 = max(0.0, min(float(y2), float(height - 1)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def xyxy_to_yolo_xywh(
    box: Sequence[float], width: int, height: int
) -> Tuple[float, float, float, float]:
    """xyxy 픽셀 좌표 -> 정규화된 YOLO (cx, cy, w, h)."""

    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive")

    x1, y1, x2, y2 = clamp_box_xyxy(box, width, height)
    bw = max(0.0, float(x2) - float(x1))
    bh = max(0.0, float(y2) - float(y1))
    cx = float(x1) + bw / 2.0
    cy = float(y1) + bh / 2.0
    values = (
        cx / float(width),
        cy / float(height),
        bw / float(width),
        bh / float(height),
    )
    return tuple(min(1.0, max(0.0, v)) for v in values)  # type: ignore[return-value]


def normalize_polygon(
    polygon: Sequence[Sequence[float]], width: int, height: int
) -> List[float]:
    """폴리곤을 [x1, y1, x2, y2, ...] 정규화 1D 리스트로 변환."""

    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive")

    flat: List[float] = []
    for x, y in polygon:
        flat.append(float(x) / float(width))
        flat.append(float(y) / float(height))
    # [0, 1] 범위로 클램프
    flat = [min(1.0, max(0.0, v)) for v in flat]
    return flat


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def nms_per_class(detections: Iterable, iou_threshold: float = 0.5) -> List:
    """간단한 per-class NMS. DetectionBox 시퀀스를 받아서 살아남은 항목 반환."""

    dets = list(detections)
    if not dets:
        return []

    by_class: dict = {}
    for d in dets:
        by_class.setdefault(d.class_id, []).append(d)

    keep_all: List = []
    for _, items in by_class.items():
        items = sorted(items, key=lambda x: x.score, reverse=True)
        kept: List = []
        while items:
            head = items.pop(0)
            kept.append(head)
            items = [
                it
                for it in items
                if compute_iou(head.box_xyxy, it.box_xyxy) < iou_threshold
            ]
        keep_all.extend(kept)

    keep_all.sort(key=lambda x: x.score, reverse=True)
    return keep_all
