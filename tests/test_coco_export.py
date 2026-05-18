import json
from pathlib import Path

from autolabeler.datatypes import ImageAnnotationResult, InstanceAnnotation
from autolabeler.exporters.coco import export_coco


def _make_results():
    inst = InstanceAnnotation(
        image_id=0,
        image_path="img1.png",
        class_id=1,
        class_name="dog",
        prompt="dog",
        score=0.75,
        box_xyxy=(10, 20, 110, 220),
        polygon_xy=[[10, 20], [110, 20], [110, 220], [10, 220]],
        area=20000.0,
    )
    return [
        ImageAnnotationResult(
            image_id=0,
            image_path="img1.png",
            width=640,
            height=480,
            instances=[inst],
        )
    ]


def test_coco_basic_schema(tmp_path: Path):
    out = export_coco(
        _make_results(),
        tmp_path / "ann.json",
        ["person", "dog"],
    )
    data = json.loads(out.read_text())
    assert set(data.keys()) >= {"images", "annotations", "categories"}
    assert len(data["images"]) == 1
    assert data["images"][0]["width"] == 640
    assert data["images"][0]["height"] == 480

    assert len(data["categories"]) == 2
    assert data["categories"][1]["name"] == "dog"

    assert len(data["annotations"]) == 1
    a = data["annotations"][0]
    assert a["category_id"] == 1
    assert a["image_id"] == 0
    assert a["iscrowd"] == 0
    assert a["bbox"] == [10.0, 20.0, 100.0, 200.0]
    assert isinstance(a["segmentation"], list)
    assert len(a["segmentation"][0]) == 8
