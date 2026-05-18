"""GroundingDINO (HuggingFace Transformers) 어댑터."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from PIL import Image

from ..config import AutoLabelConfig
from ..datatypes import DetectionBox
from ..postprocess import clamp_box_xyxy, nms_per_class


def _best_class_match(label_text: str, class_prompts: List[Dict]) -> Dict:
    """모델이 반환한 문자열에 대해 가장 잘 맞는 클래스 항목을 찾는다."""

    if not class_prompts:
        return {"class_id": -1, "class_name": label_text, "prompt": label_text}

    label_low = (label_text or "").strip().lower()

    # 1) 정확 일치 (class_name)
    for c in class_prompts:
        if label_low == c["class_name"].lower():
            return c
    # 2) 정확 일치 (prompt)
    for c in class_prompts:
        if label_low == c["prompt"].lower():
            return c
    # 3) 부분 포함
    for c in class_prompts:
        if c["class_name"].lower() in label_low or label_low in c["class_name"].lower():
            return c
    for c in class_prompts:
        if c["prompt"].lower() in label_low or label_low in c["prompt"].lower():
            return c
    # 4) 토큰 단위 매칭
    label_tokens = set(label_low.split())
    best = None
    best_overlap = 0
    for c in class_prompts:
        tokens = set(c["class_name"].lower().split()) | set(
            c["prompt"].lower().split()
        )
        overlap = len(label_tokens & tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best = c
    if best is not None and best_overlap > 0:
        return best

    return class_prompts[0]


class GroundingDINODetector:
    """GroundingDINO 박스 디텍터."""

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
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )
        except Exception as e:  # pragma: no cover - 환경 의존
            raise RuntimeError(
                "GroundingDINO 로딩 실패: transformers / torch 가 필요합니다. "
                "requirements.txt 를 설치하거나 mock_mode 를 사용하세요. "
                f"원본 오류: {e}"
            ) from e

        self.processor = AutoProcessor.from_pretrained(self.config.det_model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.config.det_model_id
        ).to(self.device)
        self.model.eval()
        self._loaded = True

    # ------------------------------------------------------------------
    def predict(
        self, image: Image.Image, class_prompts: List[Dict]
    ) -> List[DetectionBox]:
        if not class_prompts:
            return []

        if self.config.mock_mode or self.model is None:
            return self._mock_predict(image, class_prompts)

        import torch

        # GroundingDINO 는 ". " 로 구분된 prompt 또는 nested list[str] 를 받는다.
        phrases = [c["prompt"].strip().lower() for c in class_prompts]
        text = ". ".join(phrases) + "."

        inputs = self.processor(images=image, text=text, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])  # (H, W)
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.config.box_threshold,
                text_threshold=self.config.text_threshold,
                target_sizes=target_sizes,
            )
        except TypeError:
            # 일부 transformers 버전은 인자명이 다를 수 있어 fallback
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.config.box_threshold,
                text_threshold=self.config.text_threshold,
                target_sizes=target_sizes,
            )

        if not results:
            return []
        result = results[0]

        boxes = result.get("boxes")
        scores = result.get("scores")
        labels = result.get("labels") or result.get("text_labels") or []

        if boxes is None or len(boxes) == 0:
            return []

        boxes_np = boxes.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()

        W, H = image.size
        detections: List[DetectionBox] = []
        for i in range(len(boxes_np)):
            label_text = str(labels[i]) if i < len(labels) else ""
            cls = _best_class_match(label_text, class_prompts)
            x1, y1, x2, y2 = boxes_np[i].tolist()
            x1, y1, x2, y2 = clamp_box_xyxy((x1, y1, x2, y2), W, H)
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            detections.append(
                DetectionBox(
                    class_id=int(cls["class_id"]),
                    class_name=str(cls["class_name"]),
                    prompt=str(cls["prompt"]),
                    score=float(scores_np[i]),
                    box_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                )
            )

        detections = nms_per_class(detections, self.config.nms_iou_threshold)
        return detections

    # ------------------------------------------------------------------
    def _mock_predict(
        self, image: Image.Image, class_prompts: List[Dict]
    ) -> List[DetectionBox]:
        """모델 다운로드 없이 동작하는 가짜 디텍터.

        이미지의 색상이 강한 영역을 contour 로 찾아 박스로 만든다.
        """

        W, H = image.size
        rgb = np.array(image.convert("RGB"))
        gray = np.mean(rgb, axis=2)
        # 배경이 흰색일 가능성을 가정하여 어두운 영역을 객체로 간주
        threshold = 220
        binary = (gray < threshold).astype(np.uint8) * 255

        try:
            import cv2

            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = sorted(
                contours, key=lambda c: cv2.contourArea(c), reverse=True
            )
        except Exception:  # pragma: no cover
            contours = []

        detections: List[DetectionBox] = []
        if not contours:
            # 중앙에 박스 하나 생성
            cx, cy = W / 2.0, H / 2.0
            bw, bh = W * 0.4, H * 0.4
            box = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
            cls = class_prompts[0]
            detections.append(
                DetectionBox(
                    class_id=int(cls["class_id"]),
                    class_name=str(cls["class_name"]),
                    prompt=str(cls["prompt"]),
                    score=0.5,
                    box_xyxy=tuple(float(v) for v in box),  # type: ignore
                )
            )
            return detections

        import cv2

        for i, cnt in enumerate(contours[: max(1, len(class_prompts))]):
            area = float(cv2.contourArea(cnt))
            if area < 50:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cls = class_prompts[i % len(class_prompts)]
            detections.append(
                DetectionBox(
                    class_id=int(cls["class_id"]),
                    class_name=str(cls["class_name"]),
                    prompt=str(cls["prompt"]),
                    score=0.9 - i * 0.05,
                    box_xyxy=(
                        float(x),
                        float(y),
                        float(x + w),
                        float(y + h),
                    ),
                )
            )

        return detections
