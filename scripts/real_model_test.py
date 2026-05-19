"""GroundingDINO + SAM2 실제 모델 sanity check 스크립트."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autolabeler.config import AutoLabelConfig  # noqa: E402
from autolabeler.datatypes import ImageAnnotationResult, InstanceAnnotation  # noqa: E402
from autolabeler.exporters import (  # noqa: E402
    export_coco,
    export_yolo_detection,
    export_yolo_segmentation,
    make_zip,
)
from autolabeler.models import GroundingDINODetector, SAM2Segmenter  # noqa: E402
from autolabeler.postprocess import clean_mask, mask_to_polygons, polygon_area  # noqa: E402
from autolabeler.utils import ensure_dir, list_image_files, load_image_rgb, parse_class_prompts  # noqa: E402
from autolabeler.visualize import draw_overlay  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="실제 GroundingDINO + SAM2 모델 연결을 한 장의 이미지로 점검합니다."
    )
    parser.add_argument("--image", default=None, help="테스트할 이미지 경로")
    parser.add_argument("--classes", default="person, bottle, dog")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--out", default="runs/real_model_test")
    parser.add_argument("--det-model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam-model-id", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--min-mask-area", type=int, default=100)
    parser.add_argument("--polygon-epsilon-ratio", type=float, default=0.003)
    parser.add_argument("--no-morphology", action="store_true")
    return parser


def _resolve_image(image_arg: str | None) -> Path | None:
    if image_arg:
        image_path = Path(image_arg)
        return image_path if image_path.exists() else None

    sample_dir = ROOT / "sample_images"
    files = list_image_files(sample_dir)
    return files[0] if files else None


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    image_path = _resolve_image(args.image)
    if image_path is None:
        print(
            "테스트할 이미지가 없습니다. sample_images/ 에 실제 사진을 추가하거나 "
            "--image /path/to/image.jpg 를 지정하세요.",
            file=sys.stderr,
        )
        return 2

    class_prompts = parse_class_prompts(args.classes)
    if not class_prompts:
        print("클래스 프롬프트를 1개 이상 입력하세요.", file=sys.stderr)
        return 2

    config = AutoLabelConfig(
        det_model_id=args.det_model_id,
        sam_model_id=args.sam_model_id,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        device=args.device,
        mock_mode=False,
        apply_morphology=not args.no_morphology,
        min_mask_area=args.min_mask_area,
        polygon_epsilon_ratio=args.polygon_epsilon_ratio,
    )

    out_dir = ensure_dir(Path(args.out))
    preview_dir = ensure_dir(out_dir / "previews")

    try:
        image = load_image_rgb(image_path)
        detector = GroundingDINODetector(config)
        segmenter = SAM2Segmenter(config)
        detector.load()
        segmenter.load()
        detections = detector.predict(image, class_prompts)
        masks = segmenter.segment(image, detections)
    except RuntimeError as e:
        print(f"실제 모델 실행 실패: {e}", file=sys.stderr)
        return 1

    instances: list[InstanceAnnotation] = []
    for det, mask in zip(detections, masks):
        cleaned = clean_mask(
            mask,
            kernel_size=config.morphology_kernel_size,
            apply_morphology=config.apply_morphology,
        )
        polygons = mask_to_polygons(
            cleaned,
            epsilon_ratio=config.polygon_epsilon_ratio,
            min_area=config.min_mask_area,
        )
        if not polygons:
            continue
        polygons.sort(key=polygon_area, reverse=True)
        polygon = polygons[0]
        instances.append(
            InstanceAnnotation(
                image_id=0,
                image_path=str(image_path),
                class_id=det.class_id,
                class_name=det.class_name,
                prompt=det.prompt,
                score=det.score,
                box_xyxy=det.box_xyxy,
                polygon_xy=polygon,
                area=polygon_area(polygon),
            )
        )

    result = ImageAnnotationResult(
        image_id=0,
        image_path=str(image_path),
        width=image.size[0],
        height=image.size[1],
        instances=instances,
    )
    preview_path = draw_overlay(
        image,
        instances,
        preview_dir / f"{image_path.stem}_preview.png",
    )
    result.preview_path = str(preview_path)

    class_names = [c["class_name"] for c in class_prompts]
    export_yolo_segmentation([result], out_dir / "yolo_seg", class_names)
    export_yolo_detection([result], out_dir / "yolo_det", class_names)
    export_coco([result], out_dir / "coco" / "annotations.json", class_names)
    zip_path = make_zip(out_dir, out_dir / "autolabeler_output.zip")

    print(f"device: {config.resolved_device()}")
    print(f"detections: {len(detections)}")
    print(f"masks: {len(masks)}")
    print(f"accepted instances: {len(instances)}")
    print(f"preview: {preview_path}")
    print(f"output: {out_dir}")
    print(f"zip: {zip_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
