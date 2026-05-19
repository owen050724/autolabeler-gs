"""YOLO 형식 (detection / segmentation) 익스포터."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, List

from ..datatypes import ImageAnnotationResult
from ..postprocess import normalize_polygon, xyxy_to_yolo_xywh
from ..utils import ensure_dir


def _label_filename(image_path: str) -> str:
    return Path(image_path).stem + ".txt"


def _write_data_yaml(out_dir: Path, class_names: List[str]) -> Path:
    yaml_path = out_dir / "data.yaml"
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    content = (
        f"path: {out_dir.resolve()}\n"
        f"train: images\n"
        f"val: images\n"
        f"names:\n{names_block}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def _copy_source_images(results: Iterable[ImageAnnotationResult], out_dir: Path) -> None:
    """YOLO dataset 구조에 맞춰 원본 이미지를 images/ 아래 복사한다."""

    images_dir = ensure_dir(out_dir / "images")
    for res in results:
        src = Path(res.image_path)
        if not src.is_file():
            continue
        dst = images_dir / src.name
        if src.resolve() == dst.resolve():
            continue
        shutil.copy2(src, dst)


def export_yolo_detection(
    results: Iterable[ImageAnnotationResult],
    out_dir: Path,
    class_names: List[str],
    copy_images: bool = True,
) -> Path:
    """YOLO detection 라벨(class cx cy w h)을 labels/ 폴더에 저장."""

    results = list(results)
    out_dir = ensure_dir(Path(out_dir))
    labels_dir = ensure_dir(out_dir / "labels")
    if copy_images:
        _copy_source_images(results, out_dir)

    for res in results:
        lines: List[str] = []
        for inst in res.instances:
            if not inst.accepted:
                continue
            cx, cy, w, h = xyxy_to_yolo_xywh(inst.box_xyxy, res.width, res.height)
            lines.append(
                f"{inst.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            )
        (labels_dir / _label_filename(res.image_path)).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

    _write_data_yaml(out_dir, class_names)
    return out_dir


def export_yolo_segmentation(
    results: Iterable[ImageAnnotationResult],
    out_dir: Path,
    class_names: List[str],
    copy_images: bool = True,
) -> Path:
    """YOLO segmentation 라벨(class x1 y1 x2 y2 ...) 저장."""

    results = list(results)
    out_dir = ensure_dir(Path(out_dir))
    labels_dir = ensure_dir(out_dir / "labels")
    if copy_images:
        _copy_source_images(results, out_dir)

    for res in results:
        lines: List[str] = []
        for inst in res.instances:
            if not inst.accepted:
                continue
            if not inst.polygon_xy or len(inst.polygon_xy) < 3:
                continue
            norm = normalize_polygon(inst.polygon_xy, res.width, res.height)
            coords = " ".join(f"{v:.6f}" for v in norm)
            lines.append(f"{inst.class_id} {coords}")
        (labels_dir / _label_filename(res.image_path)).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

    _write_data_yaml(out_dir, class_names)
    return out_dir
