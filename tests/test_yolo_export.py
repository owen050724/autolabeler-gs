from pathlib import Path

from PIL import Image

from autolabeler.datatypes import ImageAnnotationResult, InstanceAnnotation
from autolabeler.exporters.yolo import (
    export_yolo_detection,
    export_yolo_segmentation,
)


def _make_result():
    return _make_result_for_image("img1.png")


def _make_result_for_image(image_path: str):
    inst = InstanceAnnotation(
        image_id=0,
        image_path=image_path,
        class_id=2,
        class_name="dog",
        prompt="dog",
        score=0.8,
        box_xyxy=(100, 100, 300, 500),
        polygon_xy=[[100, 100], [300, 100], [300, 500], [100, 500]],
        area=80000.0,
    )
    return ImageAnnotationResult(
        image_id=0,
        image_path=image_path,
        width=1000,
        height=1000,
        instances=[inst],
    )


def test_yolo_detection_format(tmp_path: Path):
    res = _make_result()
    out = export_yolo_detection([res], tmp_path / "det", ["a", "b", "dog"])
    label = (out / "labels" / "img1.txt").read_text().strip()
    parts = label.split()
    assert int(parts[0]) == 2
    # cx=0.2, cy=0.3, w=0.2, h=0.4
    assert abs(float(parts[1]) - 0.2) < 1e-4
    assert abs(float(parts[2]) - 0.3) < 1e-4
    assert abs(float(parts[3]) - 0.2) < 1e-4
    assert abs(float(parts[4]) - 0.4) < 1e-4
    assert (out / "data.yaml").exists()


def test_yolo_segmentation_format(tmp_path: Path):
    res = _make_result()
    out = export_yolo_segmentation([res], tmp_path / "seg", ["a", "b", "dog"])
    label = (out / "labels" / "img1.txt").read_text().strip()
    parts = label.split()
    assert int(parts[0]) == 2
    coords = [float(v) for v in parts[1:]]
    assert len(coords) == 8  # 4점 * (x, y)
    # 첫 점: (0.1, 0.1)
    assert abs(coords[0] - 0.1) < 1e-4
    assert abs(coords[1] - 0.1) < 1e-4


def test_yolo_detection_copies_images(tmp_path: Path):
    src = tmp_path / "src" / "img1.png"
    src.parent.mkdir()
    Image.new("RGB", (10, 10), (255, 255, 255)).save(src)
    res = _make_result_for_image(str(src))

    out = export_yolo_detection([res], tmp_path / "det", ["a", "b", "dog"])

    assert (out / "images" / "img1.png").exists()
    data_yaml = (out / "data.yaml").read_text(encoding="utf-8")
    assert "train: images" in data_yaml
    assert "val: images" in data_yaml


def test_yolo_segmentation_copies_images(tmp_path: Path):
    src = tmp_path / "src" / "img1.png"
    src.parent.mkdir()
    Image.new("RGB", (10, 10), (255, 255, 255)).save(src)
    res = _make_result_for_image(str(src))

    out = export_yolo_segmentation([res], tmp_path / "seg", ["a", "b", "dog"])

    assert (out / "images" / "img1.png").exists()
