"""SAM2 어댑터 (HuggingFace Transformers Sam2Processor / Sam2Model)."""

from __future__ import annotations

from typing import List

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
        self.device = config.resolved_device()
        self._loaded = False

    def load(self) -> None:
        if self._loaded or self.config.mock_mode:
            self._loaded = True
            return

        try:
            import torch  # noqa: F401
            import transformers
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "SAM2 로딩 실패: torch / transformers 가 필요합니다. "
                f"원본 오류: {e}"
            ) from e

        Sam2Processor = getattr(transformers, "Sam2Processor", None)
        Sam2Model = getattr(transformers, "Sam2Model", None)
        if Sam2Processor is None or Sam2Model is None:
            raise RuntimeError(
                "현재 설치된 transformers 버전이 SAM2 (Sam2Processor / Sam2Model) "
                "를 지원하지 않습니다. transformers >= 4.45 를 권장하며, "
                "지원되지 않는 경우 facebookresearch/sam2 공식 구현을 사용하도록 "
                "이 어댑터를 확장하세요."
            )

        self.processor = Sam2Processor.from_pretrained(self.config.sam_model_id)
        self.model = Sam2Model.from_pretrained(self.config.sam_model_id).to(
            self.device
        )
        self.model.eval()
        self._loaded = True

    # ------------------------------------------------------------------
    def segment(
        self, image: Image.Image, detections: List[DetectionBox]
    ) -> List[np.ndarray]:
        if not detections:
            return []

        if self.config.mock_mode or self.model is None:
            return self._mock_segment(image, detections)

        import torch

        input_boxes = [
            [[d.box_xyxy[0], d.box_xyxy[1], d.box_xyxy[2], d.box_xyxy[3]]]
            for d in detections
        ]

        inputs = self.processor(
            images=image,
            input_boxes=[input_boxes],
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
                if "original_sizes" in inputs
                else None,
                inputs["reshaped_input_sizes"].cpu()
                if "reshaped_input_sizes" in inputs
                else None,
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
            result.append(mi.astype(bool))
        return result

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
