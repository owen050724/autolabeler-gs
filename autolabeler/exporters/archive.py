"""출력 디렉터리 ZIP 압축 헬퍼."""

from __future__ import annotations

import zipfile
from pathlib import Path


def make_zip(output_dir: Path, zip_path: Path | None = None) -> Path:
    """output_dir 전체를 ZIP 으로 묶어 경로를 반환."""

    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"디렉터리 없음: {output_dir}")

    if zip_path is None:
        zip_path = output_dir.with_suffix(".zip")
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    zip_resolved = zip_path.resolve()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in output_dir.rglob("*"):
            if p.is_file():
                if p.resolve() == zip_resolved:
                    continue
                zf.write(p, p.relative_to(output_dir))
    return zip_path
