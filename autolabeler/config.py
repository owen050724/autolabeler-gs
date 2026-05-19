"""파이프라인 전역 설정."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

SUPPORTED_DEVICES = {"auto", "cuda", "cpu"}


@dataclass
class AutoLabelConfig:
    """AutoLabeler-GS 실행 파라미터."""

    det_model_id: str = "IDEA-Research/grounding-dino-tiny"
    sam_model_id: str = "facebook/sam2.1-hiera-tiny"

    box_threshold: float = 0.35
    text_threshold: float = 0.25
    nms_iou_threshold: float = 0.5

    min_mask_area: int = 100
    polygon_epsilon_ratio: float = 0.003
    morphology_kernel_size: int = 3
    apply_morphology: bool = True

    device: str = "auto"
    output_formats: List[str] = field(
        default_factory=lambda: ["yolo-seg", "yolo-det", "coco"]
    )

    mock_mode: bool = False

    def resolved_device(self) -> str:
        """auto 옵션을 실제 디바이스 문자열로 풀어준다."""

        requested = str(self.device).strip().lower()
        if requested not in SUPPORTED_DEVICES:
            raise ValueError(
                "device 는 'auto', 'cuda', 'cpu' 중 하나여야 합니다. "
                f"입력값: {self.device}"
            )

        if requested != "auto":
            return requested
        try:
            import torch  # local import to avoid hard dependency in mock mode

            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # pragma: no cover - torch 미설치 환경 보호
            pass
        return "cpu"
