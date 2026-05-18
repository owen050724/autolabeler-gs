import numpy as np

from autolabeler.postprocess import clean_mask, mask_to_polygons, polygon_area


def test_mask_to_polygons_rect():
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:150, 60:140] = 1
    polys = mask_to_polygons(mask, epsilon_ratio=0.01, min_area=100)
    assert len(polys) >= 1
    assert all(len(p) >= 3 for p in polys)
    # 첫 폴리곤이 (60~140, 50~150) 영역 안에 들어와야 함
    xs = [p[0] for p in polys[0]]
    ys = [p[1] for p in polys[0]]
    assert min(xs) >= 50
    assert max(xs) <= 150
    assert min(ys) >= 40
    assert max(ys) <= 160


def test_mask_to_polygons_filters_small():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:13, 10:13] = 1  # 매우 작은 영역
    polys = mask_to_polygons(mask, min_area=500)
    assert polys == []


def test_clean_mask_morphology():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[20:30, 20:30] = 1
    mask[0, 0] = 1  # noise
    cleaned = clean_mask(mask, kernel_size=3, apply_morphology=True)
    assert cleaned.dtype == np.uint8
    assert cleaned[0, 0] == 0  # noise 제거


def test_polygon_area_square():
    poly = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert polygon_area(poly) == 100.0
