"""README용 demo preview 이미지를 assets/screenshots/로 복사한다."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "runs" / "real_demo" / "previews"
DEFAULT_TARGET = ROOT / "assets" / "screenshots"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="runs/real_demo/previews의 preview 이미지를 README assets로 복사합니다."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="preview 폴더")
    parser.add_argument("--target", default=str(DEFAULT_TARGET), help="복사 대상 폴더")
    parser.add_argument(
        "--limit",
        type=int,
        default=6,
        help="복사할 최대 이미지 수",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.source)
    target = Path(args.target)
    if not source.is_absolute():
        source = ROOT / source
    if not target.is_absolute():
        target = ROOT / target
    if not source.is_dir():
        print(f"preview 폴더가 없습니다: {source}", file=sys.stderr)
        print(
            "먼저 real demo를 실행해 runs/real_demo/previews/를 생성하세요.",
            file=sys.stderr,
        )
        return 2

    target.mkdir(parents=True, exist_ok=True)
    images = sorted(
        p
        for p in source.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if not images:
        print(f"복사할 preview 이미지가 없습니다: {source}", file=sys.stderr)
        return 2

    copied = []
    for idx, src in enumerate(images[: max(1, args.limit)], start=1):
        dst = target / f"demo_preview_{idx}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        copied.append(dst)

    print(f"[OK] {len(copied)}개 preview 복사 완료 -> {target}")
    for path in copied:
        print(path.resolve().relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
