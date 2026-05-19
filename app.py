"""AutoLabeler-GS Streamlit 데모 앱."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

import streamlit as st

from autolabeler.config import AutoLabelConfig
from autolabeler.exporters.archive import make_zip
from autolabeler.pipeline import AutoLabelPipeline
from autolabeler.utils import parse_class_prompts


st.set_page_config(
    page_title="AutoLabeler-GS",
    page_icon=":label:",
    layout="wide",
)


# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="모델을 불러오는 중...")
def _get_pipeline(config_key: tuple, _config: AutoLabelConfig) -> AutoLabelPipeline:
    """설정이 바뀌지 않는 한 동일한 파이프라인을 재사용한다."""

    pipeline = AutoLabelPipeline(_config)
    pipeline.load_models()
    return pipeline


# ----------------------------------------------------------------------
def sidebar_controls() -> AutoLabelConfig:
    st.sidebar.header("설정")

    st.sidebar.subheader("모델 ID")
    det_model_id = st.sidebar.text_input(
        "GroundingDINO 모델", value="IDEA-Research/grounding-dino-tiny"
    )
    sam_model_id = st.sidebar.text_input(
        "SAM2 모델", value="facebook/sam2.1-hiera-tiny"
    )

    st.sidebar.subheader("디텍션 임계값")
    box_threshold = st.sidebar.slider("box_threshold", 0.05, 0.95, 0.35, 0.05)
    text_threshold = st.sidebar.slider("text_threshold", 0.05, 0.95, 0.25, 0.05)
    nms_iou_threshold = st.sidebar.slider("NMS IoU", 0.1, 0.95, 0.5, 0.05)

    st.sidebar.subheader("후처리")
    apply_morphology = st.sidebar.checkbox("모폴로지 적용", value=True)
    min_mask_area = st.sidebar.slider("최소 마스크 면적", 0, 5000, 100, 50)
    polygon_epsilon_ratio = st.sidebar.slider(
        "폴리곤 epsilon ratio", 0.0, 0.02, 0.003, 0.001
    )

    st.sidebar.subheader("실행 모드")
    mock_mode = st.sidebar.checkbox(
        "Mock 모드 (모델 다운로드 없이 실행)", value=False
    )
    device = st.sidebar.selectbox("device", ["auto", "cuda", "cpu"], index=0)

    output_formats = st.sidebar.multiselect(
        "출력 포맷",
        options=["yolo-seg", "yolo-det", "coco"],
        default=["yolo-seg", "yolo-det", "coco"],
    )

    return AutoLabelConfig(
        det_model_id=det_model_id,
        sam_model_id=sam_model_id,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        nms_iou_threshold=nms_iou_threshold,
        apply_morphology=apply_morphology,
        min_mask_area=int(min_mask_area),
        polygon_epsilon_ratio=float(polygon_epsilon_ratio),
        device=device,
        mock_mode=mock_mode,
        output_formats=output_formats,
    )


def _save_uploaded_files(uploaded_files, target_dir: Path) -> List[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for uf in uploaded_files:
        dest = target_dir / uf.name
        with open(dest, "wb") as f:
            f.write(uf.getbuffer())
        saved.append(dest)
    return saved


# ----------------------------------------------------------------------
def main():
    st.title("AutoLabeler-GS")
    st.caption(
        "GroundingDINO + SAM2 + OpenCV 후처리로 텍스트 프롬프트만 입력하면 "
        "객체 검출과 분할, 그리고 YOLO/COCO 라벨까지 자동으로 만들어주는 도구입니다."
    )

    config = sidebar_controls()
    display_score_threshold = st.sidebar.slider(
        "결과 표 confidence 필터",
        0.0,
        1.0,
        0.0,
        0.05,
        help="검출 결과 표와 클래스별 요약에 표시할 최소 score입니다.",
    )

    st.subheader("1) 클래스 프롬프트")
    raw_prompts = st.text_area(
        "쉼표 또는 줄바꿈으로 클래스를 입력하세요. "
        "고급 사용 예: `bottle: a plastic bottle, a water bottle`. "
        "프롬프트는 영어가 더 잘 동작합니다.",
        value="person, dog, bicycle",
        height=120,
    )

    st.subheader("2) 이미지 입력")
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        uploaded = st.file_uploader(
            "이미지 업로드 (여러 장 가능)",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
        )
    with col_in2:
        folder_path = st.text_input(
            "또는 로컬 폴더 경로 입력",
            value="",
            placeholder="/path/to/images",
        )

    run = st.button("Run Auto Labeling", type="primary")

    if run:
        class_prompts = parse_class_prompts(raw_prompts)
        if not class_prompts:
            st.error("클래스 프롬프트를 1개 이상 입력해주세요.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("runs") / f"streamlit_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        if uploaded:
            img_dir = out_dir / "uploaded_images"
            _save_uploaded_files(uploaded, img_dir)
        elif folder_path and Path(folder_path).is_dir():
            img_dir = Path(folder_path)
        else:
            st.error("이미지를 업로드하거나 유효한 로컬 폴더 경로를 입력해주세요.")
            return

        try:
            pipeline = _get_pipeline(
                config_key=(
                    config.det_model_id,
                    config.sam_model_id,
                    config.device,
                    config.mock_mode,
                ),
                _config=config,
            )
        except RuntimeError as e:
            st.error(f"모델 로딩 실패: {e}")
            st.info("Mock 모드를 활성화하면 모델 없이 데모를 실행할 수 있습니다.")
            return

        # 사용자가 슬라이더 등을 바꿔도 동일 파이프라인을 쓰므로 설정만 갱신
        pipeline.config = config
        pipeline.detector.config = config
        pipeline.segmenter.config = config

        progress = st.progress(0.0)
        status = st.empty()

        def _cb(done: int, total: int, path: Path):
            progress.progress(done / max(1, total))
            status.write(f"처리 중: [{done}/{total}] {path.name}")

        try:
            results = pipeline.process_folder(
                image_dir=img_dir,
                raw_prompts=raw_prompts,
                out_dir=out_dir,
                progress_callback=_cb,
            )
        except Exception as e:
            st.error(f"파이프라인 실행 중 오류: {e}")
            return

        st.success(
            f"완료: 이미지 {len(results)}장 / 인스턴스 "
            f"{sum(len(r.instances) for r in results)}건"
        )

        # 갤러리
        st.subheader("3) 미리보기")
        cols = st.columns(3)
        for i, res in enumerate(results):
            with cols[i % 3]:
                if res.preview_path and Path(res.preview_path).exists():
                    st.image(
                        res.preview_path,
                        caption=Path(res.image_path).name,
                        use_container_width=True,
                    )
                else:
                    st.write(f"(미리보기 없음) {Path(res.image_path).name}")

        # 검출 테이블
        st.subheader("4) 검출 결과")
        rows = []
        for res in results:
            for inst in res.instances:
                if inst.score < display_score_threshold:
                    continue
                rows.append(
                    {
                        "image": Path(res.image_path).name,
                        "class": inst.class_name,
                        "score": round(inst.score, 3),
                        "area": round(inst.area, 1),
                        "accepted": inst.accepted,
                    }
                )
        if rows:
            summary = []
            by_class = {}
            for row in rows:
                bucket = by_class.setdefault(row["class"], [])
                bucket.append(float(row["score"]))
            for class_name, scores in sorted(by_class.items()):
                summary.append(
                    {
                        "class_name": class_name,
                        "count": len(scores),
                        "average_score": round(sum(scores) / len(scores), 3),
                    }
                )
            st.caption(f"표시 confidence threshold: {display_score_threshold:.2f}")
            st.dataframe(summary, use_container_width=True)
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("표시 조건을 만족하는 검출 객체가 없습니다.")

        # 다운로드
        st.subheader("5) 결과 다운로드")
        try:
            zip_path = out_dir / "autolabeler_output.zip"
            if not zip_path.exists():
                zip_path = make_zip(out_dir, zip_path)
            with open(zip_path, "rb") as f:
                st.download_button(
                    "라벨 ZIP 다운로드",
                    data=f.read(),
                    file_name="autolabels.zip",
                    mime="application/zip",
                )
        except Exception as e:
            st.warning(f"ZIP 생성 실패: {e}")

        st.caption(f"출력 디렉터리: `{out_dir}`")


if __name__ == "__main__":  # pragma: no cover
    main()
