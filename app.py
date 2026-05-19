"""AutoLabeler-GS Streamlit 데모 앱."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List

import streamlit as st

from autolabeler.config import AutoLabelConfig
from autolabeler.exporters.archive import make_zip
from autolabeler.pipeline import AutoLabelPipeline
from autolabeler.quality import build_quality_report
from autolabeler.utils import parse_class_prompts


st.set_page_config(
    page_title="AutoLabeler-GS",
    page_icon=":label:",
    layout="wide",
)


PROMPT_PRESETS = {
    "Desk objects": "laptop, cup, bottle, mouse, keyboard, book, phone",
    "Campus objects": "person, chair, desk, laptop, backpack, bottle",
    "Recycling objects": "plastic bottle, paper cup, can, cardboard box, plastic bag",
    "Basic COCO-like objects": "person, dog, cat, bicycle, car, chair, bottle",
}

PRIORITY_ORDER = ["HIGH", "MEDIUM", "LOW"]


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


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .hero {
            border: 1px solid rgba(49, 51, 63, 0.16);
            border-radius: 8px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1rem;
            background: linear-gradient(180deg, #ffffff 0%, #f7f9fb 100%);
        }
        .hero h1 {
            margin: 0 0 0.25rem 0;
            font-size: 2.1rem;
            letter-spacing: 0;
        }
        .hero .subtitle {
            margin: 0 0 0.55rem 0;
            font-size: 1.02rem;
            color: #425466;
        }
        .hero .body {
            margin: 0 0 0.9rem 0;
            color: #536471;
            line-height: 1.5;
        }
        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
        }
        .badge {
            display: inline-flex;
            align-items: center;
            border: 1px solid rgba(49, 51, 63, 0.18);
            border-radius: 999px;
            padding: 0.22rem 0.58rem;
            background: #ffffff;
            color: #273444;
            font-size: 0.84rem;
            font-weight: 600;
        }
        .muted {
            color: #667085;
            font-size: 0.92rem;
        }
        .ok-pill {
            color: #047857;
            font-weight: 700;
        }
        .missing-pill {
            color: #b42318;
            font-weight: 700;
        }
        .skip-pill {
            color: #667085;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>AutoLabeler-GS</h1>
          <p class="subtitle">GroundingDINO + SAM2 기반 자동 Segmentation Annotation 생성 및 검수 도구</p>
          <p class="body">
            텍스트 프롬프트만 입력하면 객체 탐지, segmentation mask 생성,
            OpenCV polygon 변환, YOLO/COCO export, 품질 리포트까지 자동으로 수행합니다.
          </p>
          <div class="badge-row">
            <span class="badge">GroundingDINO → SAM2 → OpenCV → YOLO/COCO → Quality Report</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _init_session_state() -> None:
    st.session_state.setdefault("raw_prompts", "person, dog, bicycle")
    st.session_state.setdefault("results", None)
    st.session_state.setdefault("out_dir", None)
    st.session_state.setdefault("last_output_formats", [])
    st.session_state.setdefault("last_run_summary", "")


def _get_results() -> list:
    return st.session_state.get("results") or []


def _get_out_dir() -> Path | None:
    out_dir = st.session_state.get("out_dir")
    return Path(out_dir) if out_dir else None


def _class_summary(results: list) -> list[dict]:
    by_class: dict[str, list[float]] = {}
    for res in results:
        for inst in res.instances:
            by_class.setdefault(inst.class_name, []).append(float(inst.score))

    return [
        {
            "class_name": class_name,
            "count": len(scores),
            "average_score": round(sum(scores) / len(scores), 3),
        }
        for class_name, scores in sorted(by_class.items())
    ]


def _annotation_rows(results: list, score_threshold: float, class_filter: list[str]) -> list[dict]:
    rows = []
    selected = set(class_filter)
    for res in results:
        for inst in res.instances:
            if inst.score < score_threshold:
                continue
            if selected and inst.class_name not in selected:
                continue
            rows.append(
                {
                    "image": Path(res.image_path).name,
                    "class": inst.class_name,
                    "score": round(float(inst.score), 3),
                    "area": round(float(inst.area), 1),
                    "accepted": inst.accepted,
                    "polygon_points": len(inst.polygon_xy or []),
                }
            )
    return rows


def _priority_table(quality_rows: list[dict]) -> list[dict]:
    counts = Counter(row["review_priority"] for row in quality_rows)
    ordered = [p for p in PRIORITY_ORDER if counts[p] > 0]
    ordered.extend(sorted(p for p in counts if p not in PRIORITY_ORDER))
    return [{"review_priority": key, "count": counts[key]} for key in ordered]


def _issue_table(quality_rows: list[dict]) -> list[dict]:
    counts = Counter(issue for row in quality_rows for issue in row["issues"])
    return [{"issue": key, "count": counts[key]} for key in sorted(counts)]


def _high_priority_rows(quality_rows: list[dict]) -> list[dict]:
    rows = []
    for row in quality_rows:
        if row["review_priority"] != "HIGH":
            continue
        rows.append(
            {
                "image": Path(row["image_path"]).name,
                "class": row["class_name"],
                "score": round(float(row["score"]), 3),
                "priority": row["review_priority"],
                "issues": ", ".join(row["issues"]) if row["issues"] else "-",
            }
        )
    return sorted(rows, key=lambda row: (row["score"], row["image"]))


def _top_classes_for_result(result) -> str:
    counts = Counter(inst.class_name for inst in result.instances)
    if not counts:
        return "-"
    return ", ".join(f"{name} ({count})" for name, count in counts.most_common(3))


def _run_autolabeling(config: AutoLabelConfig, raw_prompts: str, uploaded, folder_path: str) -> None:
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

    total_instances = sum(len(r.instances) for r in results)
    st.session_state["results"] = results
    st.session_state["out_dir"] = str(out_dir)
    st.session_state["last_output_formats"] = list(config.output_formats)
    st.session_state["last_run_summary"] = (
        f"이미지 {len(results)}장 / 인스턴스 {total_instances}건"
    )
    st.success(f"완료: {st.session_state['last_run_summary']}")


def _render_metric_row(results: list, quality_rows: list[dict]) -> None:
    total_instances = sum(len(r.instances) for r in results)
    class_count = len({inst.class_name for r in results for inst in r.instances})
    high_count = sum(1 for row in quality_rows if row["review_priority"] == "HIGH")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images", len(results))
    c2.metric("Instances", total_instances)
    c3.metric("Classes", class_count)
    c4.metric("High priority", high_count)


def _render_export_status(out_dir: Path, output_formats: list[str]) -> None:
    items = [
        ("previews/", out_dir / "previews", True),
        ("yolo_det/", out_dir / "yolo_det", "yolo-det" in output_formats),
        ("yolo_seg/", out_dir / "yolo_seg", "yolo-seg" in output_formats),
        ("coco/annotations.json", out_dir / "coco" / "annotations.json", "coco" in output_formats),
        ("quality_report.csv", out_dir / "quality_report.csv", True),
        ("quality_report.md", out_dir / "quality_report.md", True),
        ("autolabeler_output.zip", out_dir / "autolabeler_output.zip", True),
    ]

    for label, path, requested in items:
        if path.exists():
            status = '<span class="ok-pill">OK</span>'
        elif requested:
            status = '<span class="missing-pill">Missing</span>'
        else:
            status = '<span class="skip-pill">Not requested</span>'
        st.markdown(f"- {status} `{label}`", unsafe_allow_html=True)


# ----------------------------------------------------------------------
def main():
    _inject_css()
    _init_session_state()
    _render_header()
    config = sidebar_controls()
    results = _get_results()
    quality_rows = build_quality_report(results) if results else []
    out_dir = _get_out_dir()

    tab_upload, tab_gallery, tab_table, tab_quality, tab_export = st.tabs(
        [
            "1. Upload & Prompt",
            "2. Results Gallery",
            "3. Annotation Table",
            "4. Quality Report",
            "5. Export",
        ]
    )

    with tab_upload:
        st.subheader("Class Prompt")
        preset = st.selectbox("Prompt preset", ["직접 입력", *PROMPT_PRESETS.keys()])
        if st.button("Preset 적용", disabled=preset == "직접 입력"):
            st.session_state["raw_prompts"] = PROMPT_PRESETS[preset]

        raw_prompts = st.text_area(
            "쉼표 또는 줄바꿈으로 클래스를 입력하세요. "
            "고급 사용 예: `bottle: a plastic bottle, a water bottle`. "
            "프롬프트는 영어가 더 잘 동작합니다.",
            key="raw_prompts",
            height=120,
        )

        st.subheader("Image Input")
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
            st.caption("업로드 이미지가 있으면 업로드 파일을 우선 사용합니다.")

        if st.button("Run Auto Labeling", type="primary", use_container_width=True):
            _run_autolabeling(config, raw_prompts, uploaded, folder_path)

    with tab_gallery:
        if not results:
            st.info("먼저 Upload & Prompt 탭에서 이미지를 처리하세요.")
        else:
            _render_metric_row(results, quality_rows)
            st.caption(st.session_state.get("last_run_summary", ""))
            st.subheader("Preview Gallery")
            cols = st.columns(3)
            for i, res in enumerate(results):
                with cols[i % 3]:
                    with st.container(border=True):
                        name = Path(res.image_path).name
                        if res.preview_path and Path(res.preview_path).exists():
                            st.image(res.preview_path, use_container_width=True)
                        else:
                            st.write(f"(미리보기 없음) {name}")
                        st.markdown(f"**{name}**")
                        st.caption(
                            f"instances: {len(res.instances)} | "
                            f"top classes: {_top_classes_for_result(res)}"
                        )
            summary = _class_summary(results)
            if summary:
                st.subheader("Class-wise Summary")
                st.dataframe(summary, use_container_width=True)

    with tab_table:
        if not results:
            st.info("처리된 결과가 없습니다.")
        else:
            st.subheader("Annotation Table")
            all_classes = sorted({inst.class_name for res in results for inst in res.instances})
            c1, c2 = st.columns([1, 2])
            with c1:
                display_score_threshold = st.slider(
                    "Score filter",
                    0.0,
                    1.0,
                    0.0,
                    0.05,
                    help="표에 표시할 최소 confidence score입니다.",
                )
            with c2:
                class_filter = st.multiselect(
                    "Class filter",
                    options=all_classes,
                    default=[],
                    help="비워두면 모든 클래스를 표시합니다.",
                )
            rows = _annotation_rows(results, display_score_threshold, class_filter)
            st.caption(f"표시 confidence threshold: {display_score_threshold:.2f}")
            if rows:
                st.dataframe(rows, use_container_width=True)
            else:
                st.info("표시 조건을 만족하는 annotation이 없습니다.")

    with tab_quality:
        st.subheader("Annotation Quality Report")
        st.markdown(
            '<p class="muted">품질 점수는 정답 품질을 보장하는 지표가 아니라, '
            "사람이 먼저 검수할 annotation을 찾기 위한 heuristic입니다.</p>",
            unsafe_allow_html=True,
        )
        if not quality_rows:
            st.info("품질 분석 대상 어노테이션이 없습니다.")
        else:
            _render_metric_row(results, quality_rows)
            st.dataframe(_priority_table(quality_rows), use_container_width=True)
            issue_rows = _issue_table(quality_rows)
            if issue_rows:
                st.dataframe(issue_rows, use_container_width=True)
            else:
                st.info("감지된 품질 issue가 없습니다.")

            high_rows = _high_priority_rows(quality_rows)
            if high_rows:
                st.subheader("High-priority annotations")
                st.dataframe(high_rows, use_container_width=True)
            else:
                st.success("HIGH priority annotation이 없습니다.")

            if out_dir:
                csv_path = out_dir / "quality_report.csv"
                md_path = out_dir / "quality_report.md"
                if csv_path.exists():
                    st.caption(f"CSV: `{csv_path}`")
                if md_path.exists():
                    st.caption(f"Markdown: `{md_path}`")
                    with st.expander("quality_report.md 미리보기"):
                        st.markdown(md_path.read_text(encoding="utf-8"))

    with tab_export:
        if not out_dir:
            st.info("처리 후 export 결과가 표시됩니다.")
        else:
            st.subheader("Export")
            st.caption(f"출력 디렉터리: `{out_dir}`")
            _render_export_status(out_dir, st.session_state.get("last_output_formats", []))
            st.markdown(
                "YOLO export는 이동 가능한 dataset 폴더를 만들기 위해 "
                "`data.yaml`에 `path: .`, `train: images`, `val: images`를 기록합니다."
            )
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
                        use_container_width=True,
                    )
            except Exception as e:
                st.warning(f"ZIP 생성 실패: {e}")


if __name__ == "__main__":  # pragma: no cover
    main()
