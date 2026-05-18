"""COCO JSON 익스포터 (pycocotools 미사용)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from ..datatypes import ImageAnnotationResult


def export_coco(
    results: Iterable[ImageAnnotationResult],
    out_json: Path,
    class_names: List[str],
) -> Path:
    """COCO JSON (images / annotations / categories)."""

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    categories = [
        {"id": i, "name": name, "supercategory": "object"}
        for i, name in enumerate(class_names)
    ]

    ann_id = 1
    for res in results:
        images.append(
            {
                "id": int(res.image_id),
                "file_name": Path(res.image_path).name,
                "width": int(res.width),
                "height": int(res.height),
            }
        )
        for inst in res.instances:
            if not inst.accepted:
                continue
            x1, y1, x2, y2 = inst.box_xyxy
            bw = max(0.0, float(x2 - x1))
            bh = max(0.0, float(y2 - y1))

            seg_flat: List[float] = []
            if inst.polygon_xy and len(inst.polygon_xy) >= 3:
                for x, y in inst.polygon_xy:
                    seg_flat.append(float(x))
                    seg_flat.append(float(y))

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": int(res.image_id),
                    "category_id": int(inst.class_id),
                    "bbox": [float(x1), float(y1), bw, bh],
                    "area": float(inst.area if inst.area else bw * bh),
                    "segmentation": [seg_flat] if seg_flat else [],
                    "iscrowd": 0,
                    "score": float(inst.score),
                }
            )
            ann_id += 1

    coco = {
        "info": {
            "description": "AutoLabeler-GS auto-generated COCO annotations",
            "version": "0.1",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    out_json.write_text(
        json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_json
