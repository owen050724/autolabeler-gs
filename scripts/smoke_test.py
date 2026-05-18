"""Mock 모드 smoke 테스트: 모델 다운로드 없이 파이프라인이 끝까지 돈다는 것만 확인."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# 패키지 import 가능하도록 경로 보정
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autolabeler.config import AutoLabelConfig  # noqa: E402
from autolabeler.pipeline import AutoLabelPipeline  # noqa: E402


def _make_synthetic_image(path: Path, color1=(40, 80, 200), color2=(220, 60, 60)):
    canvas = np.full((480, 640, 3), 255, dtype=np.uint8)
    # 직사각형
    canvas[80:280, 80:300] = color1
    # 원 (NumPy 슬라이싱으로 단순 마스킹)
    yy, xx = np.ogrid[:480, :640]
    mask = (xx - 460) ** 2 + (yy - 340) ** 2 < 90 ** 2
    canvas[mask] = color2
    Image.fromarray(canvas).save(path)


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="autolabeler_smoke_"))
    img_dir = work / "images"
    out_dir = work / "outputs"
    img_dir.mkdir(parents=True, exist_ok=True)

    _make_synthetic_image(img_dir / "demo1.png", (40, 80, 200), (220, 60, 60))
    _make_synthetic_image(img_dir / "demo2.png", (60, 180, 80), (180, 60, 200))

    config = AutoLabelConfig(mock_mode=True)
    pipeline = AutoLabelPipeline(config)

    results = pipeline.process_folder(
        image_dir=img_dir,
        raw_prompts="rectangle, circle",
        out_dir=out_dir,
    )

    assert len(results) == 2, f"이미지 2장이 처리되어야 합니다, 실제 {len(results)}"
    for r in results:
        assert r.preview_path and Path(r.preview_path).exists(), \
            f"preview 파일이 없습니다: {r.preview_path}"

    yolo_seg_yaml = out_dir / "yolo_seg" / "data.yaml"
    yolo_det_yaml = out_dir / "yolo_det" / "data.yaml"
    coco_json = out_dir / "coco" / "annotations.json"
    for f in [yolo_seg_yaml, yolo_det_yaml, coco_json]:
        assert f.exists(), f"누락된 출력 파일: {f}"

    print(f"[OK] smoke test passed -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
