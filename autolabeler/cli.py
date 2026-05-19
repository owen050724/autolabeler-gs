"""CLI 진입점: python -m autolabeler.cli ..."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import AutoLabelConfig
from .pipeline import AutoLabelPipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autolabeler",
        description=(
            "GroundingDINO + SAM2 자동 라벨링 도구. "
            "텍스트 프롬프트로 객체 검출과 분할을 수행하고 "
            "YOLO / COCO 형식으로 라벨을 내보냅니다."
        ),
    )
    p.add_argument("--images", required=True, help="이미지 폴더 경로")
    p.add_argument(
        "--classes",
        required=True,
        help='클래스 프롬프트 (예: "person, bicycle, dog")',
    )
    p.add_argument("--out", required=True, help="출력 디렉터리")
    p.add_argument("--det-model-id", default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--sam-model-id", default="facebook/sam2.1-hiera-tiny")
    p.add_argument("--box-threshold", type=float, default=0.35)
    p.add_argument("--text-threshold", type=float, default=0.25)
    p.add_argument("--nms-iou-threshold", type=float, default=0.5)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--mock", action="store_true", help="모델 다운로드 없이 mock 실행")
    p.add_argument(
        "--no-morphology",
        action="store_true",
        help="모폴로지 연산 비활성화",
    )
    p.add_argument("--min-mask-area", type=int, default=100)
    p.add_argument("--polygon-epsilon-ratio", type=float, default=0.003)
    p.add_argument(
        "--formats",
        nargs="+",
        default=["yolo-seg", "yolo-det", "coco"],
        choices=["yolo-seg", "yolo-det", "coco"],
        help="출력 포맷 (yolo-seg, yolo-det, coco)",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = AutoLabelConfig(
        det_model_id=args.det_model_id,
        sam_model_id=args.sam_model_id,
        box_threshold=float(args.box_threshold),
        text_threshold=float(args.text_threshold),
        nms_iou_threshold=float(args.nms_iou_threshold),
        device=args.device,
        mock_mode=bool(args.mock),
        apply_morphology=not args.no_morphology,
        min_mask_area=int(args.min_mask_area),
        polygon_epsilon_ratio=float(args.polygon_epsilon_ratio),
        output_formats=list(args.formats),
    )

    pipeline = AutoLabelPipeline(config)
    try:
        results = pipeline.process_folder(
            image_dir=Path(args.images),
            raw_prompts=args.classes,
            out_dir=Path(args.out),
        )
    except Exception as e:
        if args.verbose:
            logging.exception("실행 실패")
        print(f"오류: {e}", file=sys.stderr)
        return 1

    total_inst = sum(len(r.instances) for r in results)
    print(
        f"완료: 이미지 {len(results)}장, 인스턴스 {total_inst}건 -> {args.out}"
    )
    zip_path = Path(args.out) / "autolabeler_output.zip"
    if zip_path.exists():
        print(f"ZIP: {zip_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
