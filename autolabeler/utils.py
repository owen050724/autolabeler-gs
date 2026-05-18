"""공용 유틸리티: 프롬프트 파싱, 이미지 로딩, 이미지 폴더 탐색 등."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image, ImageOps

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_class_prompts(raw: str) -> List[Dict]:
    """클래스 프롬프트 문자열을 구조화된 리스트로 변환한다.

    지원 입력 형식:
      1) "person, bicycle, dog"
      2) 줄바꿈으로 구분된 클래스 이름
      3) "bottle: a plastic bottle, a water bottle" 형태의 고급 표현
    """

    if raw is None:
        return []

    tokens: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        tokens.append(line)

    # 한 줄에 콤마가 있고 ":" 가 없을 때만 콤마로 split 한다.
    expanded: List[str] = []
    for tok in tokens:
        if ":" in tok:
            expanded.append(tok)
        else:
            for sub in tok.split(","):
                sub = sub.strip()
                if sub:
                    expanded.append(sub)

    classes: List[Dict] = []
    for idx, entry in enumerate(expanded):
        if ":" in entry:
            name_part, prompt_part = entry.split(":", 1)
            class_name = name_part.strip()
            prompt_text = prompt_part.strip()
            if not prompt_text:
                prompt_text = class_name
        else:
            class_name = entry.strip()
            prompt_text = class_name

        if not class_name:
            continue
        classes.append(
            {
                "class_id": idx,
                "class_name": class_name,
                "prompt": prompt_text,
            }
        )

    # 중복 클래스 제거 (이름 기준 첫 항목 유지)
    seen = {}
    deduped: List[Dict] = []
    for c in classes:
        key = c["class_name"].lower()
        if key in seen:
            continue
        seen[key] = True
        c = dict(c)
        c["class_id"] = len(deduped)
        deduped.append(c)

    return deduped


def list_image_files(folder: Path) -> List[Path]:
    """이미지 폴더에서 지원되는 확장자만 정렬해서 반환."""

    folder = Path(folder)
    if not folder.exists():
        return []
    files = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]
    return sorted(files)


def load_image_rgb(path: Path) -> Image.Image:
    """EXIF 회전을 보정한 후 RGB PIL 이미지로 로드."""

    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def chunks(seq: Iterable, n: int):
    """간단한 청크 분할 유틸."""

    buf: list = []
    for item in seq:
        buf.append(item)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf
