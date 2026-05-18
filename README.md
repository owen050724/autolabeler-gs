# AutoLabeler-GS

> **AutoLabeler-GS**: GroundingDINO + SAM2 기반 자동 데이터셋 라벨링 도구
> Automatic Dataset Labeling Tool with GroundingDINO + SAM2

대학 컴퓨터비전 학기 프로젝트로, **텍스트 프롬프트만 입력하면** 객체 검출 박스와
인스턴스 세그멘테이션 마스크를 자동으로 만들어 YOLO / COCO 형식의 라벨을 내보내는
Streamlit 기반 데모입니다.

---

## 1. 프로젝트 소개

딥러닝 기반 객체 검출/세그멘테이션 모델을 학습시키려면 수천 장의 이미지에 박스나
폴리곤을 일일이 그리는 라벨링 노가다가 필수입니다. 본 프로젝트는 **GroundingDINO**
(zero-shot text-prompt object detection) 와 **SAM2** (Segment Anything Model 2)
를 결합해서 이 라벨링 작업을 자동화하는 도구를 만듭니다.

워크플로:

```
[이미지 + 텍스트 프롬프트]
        │
        ▼
   GroundingDINO  ──► 박스(BBox) 후보
        │
        ▼
       SAM2      ──► 박스를 prompt 로 사용한 인스턴스 마스크
        │
        ▼
   OpenCV 후처리  ──► 마스크 클린업 + Polygon 근사 (approxPolyDP)
        │
        ▼
   라벨 익스포트  ──► YOLO (det/seg) + COCO JSON + Preview PNG
```

## 2. 왜 이 문제가 중요한가

- 라벨링 비용은 ML 프로젝트 비용의 30~80%를 차지하는 가장 큰 병목입니다.
- 자동 라벨링이 사람보다 완벽하지 않아도, **사람의 검수 시간**을 줄여주는 것만으로
  파이프라인 전체의 처리량이 수 배 증가합니다.
- 본 도구는 학기 과제, 동아리 프로젝트, 학부 연구 등 데이터셋이 부족한 환경에서
  YOLO/COCO 학습용 라벨을 빠르게 부트스트랩하는 용도로 쓸 수 있습니다.

## 3. 강의 주제 매핑

| 강의 주제 | 본 프로젝트에서의 사용처 |
| --- | --- |
| OpenCV 이미지 표현, I/O | `cv2.imread/imwrite`, RGB/BGR 변환 (`visualize.py`) |
| NumPy 이미지 배열 | 마스크, 폴리곤 좌표 변환 (`postprocess.py`) |
| 영상 처리 (Thresholding, Morphology) | `clean_mask()`의 open/close 모폴로지 |
| Contour & Polygon | `cv2.findContours`, `cv2.approxPolyDP` |
| 인스턴스 세그멘테이션 | SAM2 박스-프롬프트 마스크 |
| 픽셀 좌표계 ↔ 정규화 좌표 | `xyxy_to_yolo_xywh`, `normalize_polygon` |
| CNN / Transfer Learning | 출력 YOLO/COCO 라벨로 추가 학습 가능 (Ultralytics 등) |
| Streamlit GUI | `app.py` 데모 (이전 과제 형식과 동일) |

## 4. 주요 기능

- **Text → Box → Mask → Polygon** 파이프라인을 한 번에 실행
- GroundingDINO / SAM2 (HuggingFace Transformers) 어댑터 모듈화
- **Mock 모드**: 모델 다운로드 없이도 파이프라인 전체 흐름과 익스포터를 테스트 가능
- 모폴로지 (open/close), `approxPolyDP` 기반 폴리곤 단순화
- 출력 포맷
  - `yolo-det` : `class cx cy w h` (정규화)
  - `yolo-seg` : `class x1 y1 x2 y2 ...` (정규화 폴리곤)
  - `coco` : 표준 COCO JSON (`images / annotations / categories`)
- Streamlit GUI 에서 임계값 슬라이더, 결과 미리보기 갤러리, ZIP 다운로드
- pytest 단위 테스트

## 5. 설치

```bash
git clone <repo-url> autolabeler-gs
cd autolabeler-gs

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> GPU 가 있는 경우 `torch` 는 CUDA 빌드 (https://pytorch.org/get-started/locally/)
> 를 별도로 설치하는 것을 권장합니다.

## 6. CLI 사용

```bash
python -m autolabeler.cli \
  --images sample_images \
  --classes "person, bicycle, dog" \
  --out runs/demo \
  --box-threshold 0.35 \
  --text-threshold 0.25 \
  --formats yolo-seg yolo-det coco
```

자주 쓰는 옵션:

| 옵션 | 설명 |
| --- | --- |
| `--mock` | 모델 다운로드 없이 mock 디텍터/세그멘터로 실행 |
| `--device {auto,cuda,cpu}` | 디바이스 강제 지정 |
| `--no-morphology` | 마스크 모폴로지 비활성화 |
| `--min-mask-area` | 너무 작은 마스크 폐기 (픽셀 단위) |
| `--polygon-epsilon-ratio` | `approxPolyDP` 의 epsilon = 둘레 × ratio |
| `--det-model-id` | GroundingDINO 모델 변경 (예: `IDEA-Research/grounding-dino-base`) |
| `--sam-model-id` | SAM2 모델 변경 (예: `facebook/sam2.1-hiera-large`) |

## 7. Streamlit 사용

```bash
streamlit run app.py
```

브라우저가 열리면:
1. 사이드바에서 모델 ID, 임계값, mock 모드 등을 조절합니다.
2. 클래스 프롬프트를 입력합니다 (영어 권장).
3. 이미지를 업로드하거나 로컬 폴더 경로를 입력합니다.
4. `Run Auto Labeling` 버튼을 누르면 진행률 바와 결과 갤러리/테이블이 표시됩니다.
5. 마지막에 모든 결과를 ZIP 파일로 다운로드할 수 있습니다.

## 8. 출력 디렉터리 구조

```
runs/demo/
├─ previews/
│  ├─ img1_preview.png
│  └─ ...
├─ yolo_det/
│  ├─ data.yaml
│  └─ labels/
│     ├─ img1.txt    # class cx cy w h
│     └─ ...
├─ yolo_seg/
│  ├─ data.yaml
│  └─ labels/
│     ├─ img1.txt    # class x1 y1 x2 y2 ...
│     └─ ...
└─ coco/
   └─ annotations.json
```

## 9. 데모

> 본 저장소에는 저작권상 안전한 합성 도형 이미지 생성기를 함께 제공합니다.
> 실제 사진 이미지는 학습용으로 직접 추가해주세요.

```bash
# 1) 합성 데모 이미지 4장 생성
python scripts/make_demo_assets.py

# 2) Mock 모드로 데모 실행 (모델 다운로드 없이)
python -m autolabeler.cli \
  --images sample_images \
  --classes "rectangle, circle" \
  --out runs/demo \
  --mock --verbose
```

생성되는 데모 아티팩트(자리 표시자):

- `runs/demo/previews/demo_1_preview.png` — 원본 위에 박스/폴리곤/라벨이 그려진 PNG
- `runs/demo/yolo_seg/labels/demo_1.txt` — YOLO segmentation 라벨
- `runs/demo/yolo_det/labels/demo_1.txt` — YOLO detection 라벨
- `runs/demo/coco/annotations.json` — COCO JSON

README 에 캡쳐 이미지를 첨부할 때 참고할 슬롯:

| 항목 | 자리 표시자 |
| --- | --- |
| 원본 이미지 | `docs/original.png` |
| GroundingDINO 박스 | `docs/det_boxes.png` |
| SAM2 마스크 | `docs/sam_masks.png` |
| 폴리곤 후처리 결과 | `docs/polygons.png` |
| 익스포트된 라벨 파일 | `docs/labels.png` |

## 10. 실험 (자리 표시자)

### 임계값 비교

| box_threshold | text_threshold | 검출 수 | 오검출 인상 |
| --- | --- | --- | --- |
| 0.25 | 0.15 | (TBD) | 작은 객체까지 다 잡지만 noise ↑ |
| 0.35 | 0.25 | (TBD) | 기본값, 균형 좋음 |
| 0.50 | 0.40 | (TBD) | 보수적, 누락 ↑ |

### 마스크 후처리 비교

| 후처리 | 폴리곤 점 수 | 경계 부드러움 | 비고 |
| --- | --- | --- | --- |
| 모폴로지 OFF, eps=0.001 | 많음 | 들쭉날쭉 | 디테일 보존 |
| 모폴로지 ON, eps=0.003 | 중간 | 부드러움 | 기본값 |
| 모폴로지 ON, eps=0.010 | 적음 | 매우 단순 | YOLO seg 텍스트가 짧음 |

## 11. 한계

- 영어 프롬프트(`"a yellow school bus"` 등)에서 성능이 가장 좋습니다.
- 작은 객체, 가려진 객체, 비슷한 색의 배경에서는 검출/분할 모두 실패할 수 있습니다.
- GPU 가 강력히 권장됩니다. CPU 만으로도 동작하지만 매우 느립니다.
- GroundingDINO / SAM2 가중치 다운로드 용량이 큽니다 (수백 MB 이상).
- 자동 라벨은 항상 **사람의 검수**가 필요합니다. 본 도구는 라벨러를 대체하지 않고
  보조합니다.
- 본 저장소의 `mock_mode` 는 데모/개발용이며 실제 라벨 품질이 보장되지 않습니다.

## 12. 개발 / 테스트

```bash
# 단위 테스트
pytest

# 의존성 없이 가능한 smoke test
python scripts/smoke_test.py
```

## 13. 참고 자료 및 라이선스

- **GroundingDINO** — Liu et al., 2023, https://github.com/IDEA-Research/GroundingDINO
- **Segment Anything Model 2 (SAM2)** — Meta AI, 2024, https://github.com/facebookresearch/sam2
- **Hugging Face Transformers** — https://github.com/huggingface/transformers
- **OpenCV** — https://opencv.org/
- **Streamlit** — https://streamlit.io/
- **PyTorch** — https://pytorch.org/

본 저장소의 코드는 **MIT License** 로 제공됩니다 (`LICENSE` 파일 참고).
GroundingDINO / SAM2 모델 가중치와 코드는 각 프로젝트의 원본 라이선스
(Apache 2.0 등) 를 따릅니다.
