import math

from autolabeler.postprocess import (
    clamp_box_xyxy,
    compute_iou,
    normalize_polygon,
    xyxy_to_yolo_xywh,
)


def test_xyxy_to_yolo():
    cx, cy, w, h = xyxy_to_yolo_xywh((100, 100, 300, 500), 1000, 1000)
    assert math.isclose(cx, 0.2)
    assert math.isclose(cy, 0.3)
    assert math.isclose(w, 0.2)
    assert math.isclose(h, 0.4)


def test_clamp_box():
    box = clamp_box_xyxy((-5, -10, 1200, 1500), 800, 600)
    assert box == (0.0, 0.0, 799.0, 599.0)


def test_clamp_box_swap():
    box = clamp_box_xyxy((300, 400, 100, 200), 500, 500)
    assert box[0] <= box[2]
    assert box[1] <= box[3]


def test_compute_iou_identity():
    assert math.isclose(compute_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)


def test_compute_iou_disjoint():
    assert compute_iou((0, 0, 5, 5), (10, 10, 20, 20)) == 0.0


def test_normalize_polygon():
    poly = [[0, 0], [100, 0], [100, 100], [0, 100]]
    flat = normalize_polygon(poly, 100, 100)
    assert flat == [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
