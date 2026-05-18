"""sample_images/ 폴더에 합성 데모 이미지를 만들어두는 헬퍼."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def draw_demo(path: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    canvas = Image.new("RGB", (640, 480), (245, 245, 245))
    drw = ImageDraw.Draw(canvas)

    # 랜덤 색상 도형 3개
    for _ in range(3):
        x0 = int(rng.integers(20, 480))
        y0 = int(rng.integers(20, 320))
        w = int(rng.integers(80, 160))
        h = int(rng.integers(80, 160))
        color = tuple(int(v) for v in rng.integers(40, 220, size=3))
        if rng.integers(0, 2) == 0:
            drw.rectangle([x0, y0, x0 + w, y0 + h], fill=color)
        else:
            drw.ellipse([x0, y0, x0 + w, y0 + h], fill=color)

    canvas.save(path)


def main() -> int:
    target_dir = ROOT / "sample_images"
    target_dir.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        draw_demo(target_dir / f"demo_{i+1}.png", seed=i * 7 + 1)
    print(f"[OK] 데모 이미지 생성: {target_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
