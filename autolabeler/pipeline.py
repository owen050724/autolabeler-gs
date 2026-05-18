"""GroundingDINO + SAM2 + 후처리 + 익스포트 파이프라인."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from .config import AutoLabelConfig
from .datatypes import ImageAnnotationResult, InstanceAnnotation
from .exporters import export_coco, export_yolo_detection, export_yolo_segmentation
from .models import GroundingDINODetector, SAM2Segmenter
from .postprocess import clean_mask, mask_to_polygons, polygon_area
from .utils import ensure_dir, list_image_files, load_image_rgb, parse_class_prompts
from .visualize import draw_overlay

log = logging.getLogger(__name__)


class AutoLabelPipeline:
    def __init__(self, config: AutoLabelConfig):
        self.config = config
        self.detector = GroundingDINODetector(config)
        self.segmenter = SAM2Segmenter(config)
        self._loaded = False

    def load_models(self) -> None:
        if self._loaded:
            return
        self.detector.load()
        self.segmenter.load()
        self._loaded = True

    # ------------------------------------------------------------------
    def process_image(
        self,
        image_path: Path,
        class_prompts: List[dict],
        out_dir: Path,
        image_id: int = 0,
    ) -> ImageAnnotationResult:
        image_path = Path(image_path)
        out_dir = ensure_dir(Path(out_dir))
        preview_dir = ensure_dir(out_dir / "previews")

        image = load_image_rgb(image_path)
        W, H = image.size

        detections = self.detector.predict(image, class_prompts)
        masks = self.segmenter.segment(image, detections) if detections else []

        instances: List[InstanceAnnotation] = []
        for det, mask in zip(detections, masks):
            try:
                m_u8 = clean_mask(
                    mask,
                    kernel_size=self.config.morphology_kernel_size,
                    apply_morphology=self.config.apply_morphology,
                )
                polygons = mask_to_polygons(
                    m_u8,
                    epsilon_ratio=self.config.polygon_epsilon_ratio,
                    min_area=self.config.min_mask_area,
                )
            except Exception as e:
                log.warning("마스크 후처리 실패: %s", e)
                continue

            if not polygons:
                continue

            # 가장 큰 폴리곤만 사용
            polygons.sort(key=lambda p: polygon_area(p), reverse=True)
            largest = polygons[0]
            area = float(polygon_area(largest))

            instances.append(
                InstanceAnnotation(
                    image_id=image_id,
                    image_path=str(image_path),
                    class_id=det.class_id,
                    class_name=det.class_name,
                    prompt=det.prompt,
                    score=det.score,
                    box_xyxy=det.box_xyxy,
                    polygon_xy=largest,
                    area=area,
                    accepted=True,
                )
            )

        # 미리보기 저장
        preview_path = preview_dir / (image_path.stem + "_preview.png")
        try:
            draw_overlay(image, instances, preview_path)
        except Exception as e:  # pragma: no cover
            log.warning("preview 생성 실패: %s", e)
            preview_path = None  # type: ignore

        return ImageAnnotationResult(
            image_id=image_id,
            image_path=str(image_path),
            width=int(W),
            height=int(H),
            instances=instances,
            preview_path=str(preview_path) if preview_path else None,
        )

    # ------------------------------------------------------------------
    def process_folder(
        self,
        image_dir: Path,
        raw_prompts: str,
        out_dir: Path,
        progress_callback: Optional[Callable[[int, int, Path], None]] = None,
    ) -> List[ImageAnnotationResult]:
        image_dir = Path(image_dir)
        out_dir = ensure_dir(Path(out_dir))
        class_prompts = parse_class_prompts(raw_prompts)
        if not class_prompts:
            raise ValueError("클래스 프롬프트가 비어있습니다.")

        self.load_models()

        files = list_image_files(image_dir)
        if not files:
            raise FileNotFoundError(f"이미지 파일이 없습니다: {image_dir}")

        results: List[ImageAnnotationResult] = []
        for idx, f in enumerate(files):
            log.info("[%d/%d] %s", idx + 1, len(files), f.name)
            try:
                res = self.process_image(f, class_prompts, out_dir, image_id=idx)
            except Exception as e:
                log.error("이미지 처리 실패 %s: %s", f, e)
                continue
            results.append(res)
            if progress_callback is not None:
                progress_callback(idx + 1, len(files), f)

        self.export_results(results, out_dir, class_prompts)
        return results

    # ------------------------------------------------------------------
    def export_results(
        self,
        results: List[ImageAnnotationResult],
        out_dir: Path,
        class_prompts: List[dict],
    ) -> None:
        out_dir = ensure_dir(Path(out_dir))
        class_names = [c["class_name"] for c in class_prompts]

        if "yolo-det" in self.config.output_formats:
            export_yolo_detection(results, out_dir / "yolo_det", class_names)
        if "yolo-seg" in self.config.output_formats:
            export_yolo_segmentation(results, out_dir / "yolo_seg", class_names)
        if "coco" in self.config.output_formats:
            export_coco(results, out_dir / "coco" / "annotations.json", class_names)


__all__ = ["AutoLabelPipeline"]
