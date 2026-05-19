from pathlib import Path

from autolabeler.datatypes import ImageAnnotationResult, InstanceAnnotation
from autolabeler.quality import (
    analyze_instance,
    build_quality_report,
    export_quality_report,
)


def _instance(
    score: float = 0.8,
    box=(100, 100, 300, 300),
    polygon=None,
    area: float = 40000.0,
):
    if polygon is None:
        polygon = [[100, 100], [300, 100], [300, 300], [100, 300]]
    return InstanceAnnotation(
        image_id=0,
        image_path="sample_images/laptop_1.jpg",
        class_id=0,
        class_name="laptop",
        prompt="laptop",
        score=score,
        box_xyxy=box,
        polygon_xy=polygon,
        area=area,
    )


def _result(instances):
    return ImageAnnotationResult(
        image_id=0,
        image_path="sample_images/laptop_1.jpg",
        width=1000,
        height=1000,
        instances=list(instances),
    )


def test_analyze_instance_low_priority_for_normal_annotation():
    row = analyze_instance(_instance(), 1000, 1000)

    assert row["review_priority"] == "LOW"
    assert row["issues"] == []
    assert row["polygon_points"] == 4
    assert abs(row["mask_bbox_ratio"] - 1.0) < 1e-6


def test_analyze_instance_low_score_is_high_priority():
    row = analyze_instance(_instance(score=0.31), 1000, 1000)

    assert row["review_priority"] == "HIGH"
    assert "LOW_CONFIDENCE" in row["issues"]


def test_analyze_instance_tiny_mask_is_high_priority():
    row = analyze_instance(
        _instance(
            box=(10, 10, 30, 30),
            polygon=[[10, 10], [11, 10], [11, 11], [10, 11]],
            area=1.0,
        ),
        1000,
        1000,
    )

    assert row["review_priority"] == "HIGH"
    assert "TINY_MASK" in row["issues"]


def test_build_quality_report_uses_result_dimensions():
    rows = build_quality_report([_result([_instance(), _instance(score=0.2)])])

    assert len(rows) == 2
    assert rows[0]["class_name"] == "laptop"
    assert rows[1]["review_priority"] == "HIGH"


def test_export_quality_report_creates_csv_and_markdown(tmp_path: Path):
    csv_path, md_path = export_quality_report(
        [_result([_instance(), _instance(score=0.2)])],
        tmp_path,
    )

    assert csv_path.exists()
    assert md_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    md_text = md_path.read_text(encoding="utf-8")
    assert "review_priority" in csv_text
    assert "LOW_CONFIDENCE" in csv_text
    assert "Annotation Quality Report" in md_text
    assert "Review Priority Count" in md_text
