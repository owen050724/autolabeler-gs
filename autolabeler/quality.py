"""자동 생성 어노테이션 품질 분석과 human-review 우선순위 리포트."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from statistics import mean

from .datatypes import ImageAnnotationResult, InstanceAnnotation
from .postprocess import polygon_area


DEFAULT_THRESHOLDS = {
    "low_confidence": 0.40,
    "tiny_mask_ratio": 0.001,
    "huge_mask_ratio": 0.80,
    "mask_bbox_ratio_low": 0.20,
    "mask_bbox_ratio_high": 1.10,
    "too_few_polygon_points": 3,
    "too_many_polygon_points": 200,
}

CSV_FIELDS = [
    "image_path",
    "class_name",
    "score",
    "bbox_area_ratio",
    "polygon_area_ratio",
    "mask_bbox_ratio",
    "polygon_points",
    "review_priority",
    "issues",
]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _bbox_area(box_xyxy) -> float:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def analyze_instance(
    instance: InstanceAnnotation,
    image_width: int,
    image_height: int,
    thresholds=None,
) -> dict:
    """단일 어노테이션의 confidence/geometry 기반 검수 우선순위를 계산한다."""

    cfg = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        cfg.update(thresholds)

    image_area = max(0.0, float(image_width) * float(image_height))
    bbox_area = _bbox_area(instance.box_xyxy)
    poly_area = float(instance.area)
    if poly_area <= 0 and instance.polygon_xy:
        poly_area = float(polygon_area(instance.polygon_xy))

    bbox_area_ratio = _safe_ratio(bbox_area, image_area)
    polygon_area_ratio = _safe_ratio(poly_area, image_area)
    mask_bbox_ratio = _safe_ratio(poly_area, bbox_area)
    polygon_points = len(instance.polygon_xy or [])

    issues = []
    if float(instance.score) < float(cfg["low_confidence"]):
        issues.append("LOW_CONFIDENCE")
    if polygon_area_ratio < float(cfg["tiny_mask_ratio"]):
        issues.append("TINY_MASK")
    if polygon_area_ratio > float(cfg["huge_mask_ratio"]):
        issues.append("HUGE_MASK")
    if (
        mask_bbox_ratio < float(cfg["mask_bbox_ratio_low"])
        or mask_bbox_ratio > float(cfg["mask_bbox_ratio_high"])
    ):
        issues.append("MASK_BOX_MISMATCH")
    if polygon_points < int(cfg["too_few_polygon_points"]):
        issues.append("TOO_FEW_POLYGON_POINTS")
    if polygon_points > int(cfg["too_many_polygon_points"]):
        issues.append("TOO_MANY_POLYGON_POINTS")

    if {"LOW_CONFIDENCE", "TINY_MASK", "MASK_BOX_MISMATCH"} & set(issues):
        review_priority = "HIGH"
    elif {"TOO_MANY_POLYGON_POINTS", "HUGE_MASK"} & set(issues):
        review_priority = "MEDIUM"
    else:
        review_priority = "LOW"

    return {
        "image_path": str(instance.image_path),
        "class_name": str(instance.class_name),
        "score": float(instance.score),
        "bbox_area_ratio": bbox_area_ratio,
        "polygon_area_ratio": polygon_area_ratio,
        "mask_bbox_ratio": mask_bbox_ratio,
        "polygon_points": int(polygon_points),
        "review_priority": review_priority,
        "issues": issues,
    }


def build_quality_report(results: list[ImageAnnotationResult]) -> list[dict]:
    """이미지 처리 결과 전체를 flat quality row 리스트로 변환한다."""

    rows = []
    for result in results:
        for instance in result.instances:
            if not instance.accepted:
                continue
            rows.append(analyze_instance(instance, result.width, result.height))
    return rows


def _format_csv_value(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, list):
        return ";".join(value) if value else ""
    return value


def _write_csv(rows: list[dict], csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_csv_value(row.get(key, "")) for key in CSV_FIELDS})


def _counter_lines(counter: Counter) -> list[str]:
    if not counter:
        return ["- none: 0"]
    return [f"- {key}: {counter[key]}" for key in sorted(counter)]


def _top_high_priority_rows(rows: list[dict], limit: int = 10) -> list[dict]:
    high = [row for row in rows if row["review_priority"] == "HIGH"]
    return sorted(high, key=lambda row: (float(row["score"]), row["image_path"]))[:limit]


def _write_markdown(
    rows: list[dict],
    results: list[ImageAnnotationResult],
    md_path: Path,
) -> None:
    class_counts = Counter(row["class_name"] for row in rows)
    priority_counts = Counter(row["review_priority"] for row in rows)
    issue_counts = Counter(issue for row in rows for issue in row["issues"])
    scores = [float(row["score"]) for row in rows]

    lines = [
        "# Annotation Quality Report",
        "",
        f"- total images: {len(results)}",
        f"- total instances: {len(rows)}",
    ]
    if scores:
        lines.append(f"- average confidence: {mean(scores):.3f}")
    lines.extend(["", "## Class-wise Count", ""])
    lines.extend(_counter_lines(class_counts))
    lines.extend(["", "## Review Priority Count", ""])
    lines.extend(_counter_lines(priority_counts))
    lines.extend(["", "## Issue Count", ""])
    lines.extend(_counter_lines(issue_counts))
    lines.extend(["", "## Top High-priority Annotations", ""])

    top_rows = _top_high_priority_rows(rows)
    if not top_rows:
        lines.append("No high-priority annotations.")
    else:
        lines.extend(
            [
                "| image | class | score | priority | issues |",
                "| --- | --- | ---: | --- | --- |",
            ]
        )
        for row in top_rows:
            image_name = Path(row["image_path"]).name
            issues = ", ".join(row["issues"]) if row["issues"] else "-"
            lines.append(
                f"| {image_name} | {row['class_name']} | "
                f"{float(row['score']):.3f} | {row['review_priority']} | {issues} |"
            )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_quality_report(
    results: list[ImageAnnotationResult],
    out_dir: Path,
) -> tuple[Path, Path]:
    """quality_report.csv와 quality_report.md를 생성하고 경로를 반환한다."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_quality_report(results)
    csv_path = out_dir / "quality_report.csv"
    md_path = out_dir / "quality_report.md"
    _write_csv(rows, csv_path)
    _write_markdown(rows, list(results), md_path)
    return csv_path, md_path


__all__ = [
    "analyze_instance",
    "build_quality_report",
    "export_quality_report",
]
