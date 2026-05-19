"""SAM2 어댑터 (HuggingFace Transformers Sam2Processor / Sam2Model)."""

from __future__ import annotations

from typing import List, Sequence

import cv2
import numpy as np
from PIL import Image

from ..config import AutoLabelConfig
from ..datatypes import DetectionBox


class SAM2Segmenter:
    """SAM2 박스-프롬프트 세그멘터."""

    def __init__(self, config: AutoLabelConfig):
        self.config = config
        self.processor = None
        self.model = None
        self.predictor = None
        self.backend = ""
        self.device = config.resolved_device()
        self._loaded = False

    def load(self) -> None:
        if self._loaded or self.config.mock_mode:
            self._loaded = True
            return

        try:
            import torch
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "SAM2 로딩 실패: torch 가 필요합니다. "
                f"원본 오류: {e}"
            ) from e

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "SAM2 로딩 실패: --device cuda 가 지정되었지만 현재 환경에서 "
                "CUDA를 사용할 수 없습니다. --device auto 또는 --device cpu 를 "
                "사용하세요."
            )

        try:
            import transformers
        except Exception as e:  # pragma: no cover - optional backend
            transformers = None
            hf_error = e
        else:
            hf_error = None

        Sam2Processor = getattr(transformers, "Sam2Processor", None)
        Sam2Model = getattr(transformers, "Sam2Model", None)
        if Sam2Processor is not None and Sam2Model is not None:
            try:
                self.processor = Sam2Processor.from_pretrained(self.config.sam_model_id)
                self.model = Sam2Model.from_pretrained(self.config.sam_model_id).to(
                    self.device
                )
                self.model.eval()
                self.backend = "transformers"
                self._loaded = True
                return
            except Exception as e:  # pragma: no cover - 모델/네트워크 환경 의존
                hf_error = e

        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as e:  # pragma: no cover - optional backend
            details = (
                "현재 설치된 transformers 버전이 SAM2 "
                "(Sam2Processor / Sam2Model)를 지원하지 않거나 모델 로딩에 "
                "실패했고, facebookresearch/sam2 공식 패키지도 사용할 수 없습니다. "
                "requirements-real.txt 를 확인하세요."
            )
            if hf_error is not None:
                details += f" Transformers 오류: {hf_error}."
            details += f" 공식 SAM2 import 오류: {e}."
            raise RuntimeError(details) from e

        try:
            self.predictor = SAM2ImagePredictor.from_pretrained(
                self.config.sam_model_id
            )
            if hasattr(self.predictor, "model"):
                self.predictor.model.to(self.device)
                self.predictor.model.eval()
            self.backend = "official"
            self._loaded = True
        except Exception as e:  # pragma: no cover - optional backend
            raise RuntimeError(
                "SAM2 로딩 실패: Transformers SAM2와 facebookresearch/sam2 "
                "fallback 모두 사용할 수 없습니다. "
                f"Transformers 오류: {hf_error}. 공식 SAM2 오류: {e}"
            ) from e

    # ------------------------------------------------------------------
    def segment(
        self, image: Image.Image, detections: List[DetectionBox]
    ) -> List[np.ndarray]:
        if not detections:
            return []

        if not self._loaded:
            self.load()

        if self.config.mock_mode:
            return self._mock_segment(image, detections)

        if self.backend == "transformers":
            return self._segment_transformers(image, detections)
        if self.backend == "official":
            return self._segment_official(image, detections)
        raise RuntimeError("SAM2 모델이 로드되지 않았습니다.")

    # ------------------------------------------------------------------
    def _segment_transformers(
        self, image: Image.Image, detections: Sequence[DetectionBox]
    ) -> List[np.ndarray]:
        if self.model is None or self.processor is None:
            raise RuntimeError("SAM2 Transformers 모델이 로드되지 않았습니다.")

        import torch

        boxes = _boxes_for_sam2(detections)

        inputs = self.processor(
            images=image,
            input_boxes=[boxes],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            try:
                outputs = self.model(**inputs, multimask_output=False)
            except TypeError:
                outputs = self.model(**inputs)

        try:
            masks_list = self.processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu()
                if hasattr(inputs["original_sizes"], "cpu")
                else inputs["original_sizes"],
            )
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                f"SAM2 후처리 실패: {e}. transformers 버전 호환성을 확인하세요."
            ) from e

        if not masks_list:
            return []

        # masks_list[0] shape: (N, multimask, H, W)
        m = masks_list[0]
        if hasattr(m, "cpu"):
            m = m.cpu().numpy()
        elif not isinstance(m, np.ndarray):
            m = np.asarray(m)

        result: List[np.ndarray] = []
        for i in range(m.shape[0]):
            mi = m[i]
            if mi.ndim == 3:
                mi = mi[0]
            result.append(_ensure_hw_bool_mask(mi, image.size))
        return _match_detection_count(result, len(detections), image.size)

    # ------------------------------------------------------------------
    def _segment_official(
        self, image: Image.Image, detections: Sequence[DetectionBox]
    ) -> List[np.ndarray]:
        if self.predictor is None:
            raise RuntimeError("facebookresearch/sam2 predictor 가 로드되지 않았습니다.")

        import torch

        image_np = np.array(image.convert("RGB"))
        masks: List[np.ndarray] = []
        try:
            with torch.inference_mode():
                self.predictor.set_image(image_np)
                for box in _boxes_for_sam2(detections):
                    pred_masks, _, _ = self.predictor.predict(
                        box=np.asarray(box, dtype=np.float32),
                        multimask_output=False,
                    )
                    if pred_masks is None or len(pred_masks) == 0:
                        masks.append(np.zeros(image_np.shape[:2], dtype=bool))
                        continue
                    masks.append(_ensure_hw_bool_mask(pred_masks[0], image.size))
        except Exception as e:  # pragma: no cover - optional backend
            raise RuntimeError(
                "facebookresearch/sam2 box prompt segmentation 실패: "
                f"{e}. 설치된 SAM2 API와 모델 ID를 확인하세요."
            ) from e
        return _match_detection_count(masks, len(detections), image.size)

    # ------------------------------------------------------------------
    def _mock_segment(
        self, image: Image.Image, detections: List[DetectionBox]
    ) -> List[np.ndarray]:
        """박스에 맞춘 타원형 가짜 마스크."""

        W, H = image.size
        masks: List[np.ndarray] = []
        for det in detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det.box_xyxy]
            x1 = max(0, min(W - 1, x1))
            x2 = max(0, min(W - 1, x2))
            y1 = max(0, min(H - 1, y1))
            y2 = max(0, min(H - 1, y2))
            mask = np.zeros((H, W), dtype=np.uint8)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            ax = max(1, (x2 - x1) // 2)
            ay = max(1, (y2 - y1) // 2)
            cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
            masks.append(mask.astype(bool))
        return masks


def _boxes_for_sam2(detections: Sequence[DetectionBox]) -> List[List[float]]:
    """SAM2 processor/predictor가 기대하는 XYXY box 리스트."""

    return [
        [
            float(d.box_xyxy[0]),
            float(d.box_xyxy[1]),
            float(d.box_xyxy[2]),
            float(d.box_xyxy[3]),
        ]
        for d in detections
    ]


def _ensure_hw_bool_mask(mask: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """SAM2 출력 마스크를 H x W bool 배열로 정규화한다."""

    W, H = image_size
    m = mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)
    while m.ndim > 2:
        m = m[0]
    if m.shape != (H, W):
        m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    if m.dtype == bool:
        return m
    return m > 0


def _match_detection_count(
    masks: List[np.ndarray], detection_count: int, image_size: tuple[int, int]
) -> List[np.ndarray]:
    """검출 수와 마스크 수를 맞춰 downstream zip 누락을 방지한다."""

    W, H = image_size
    if len(masks) >= detection_count:
        return masks[:detection_count]
    padded = list(masks)
    for _ in range(detection_count - len(masks)):
        padded.append(np.zeros((H, W), dtype=bool))
    return padded
