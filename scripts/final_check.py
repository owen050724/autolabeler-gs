"""최종 제출 전 경량 검증 스크립트.

실제 모델 가중치는 다운로드하지 않고 mock 파이프라인, README/asset 경로,
소스 배포 구성만 점검한다.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FORBIDDEN_DIST_DIRS = {".git", ".venv", "runs", "__pycache__", ".pytest_cache"}

REQUIRED_FILES = [
    "app.py",
    "README.md",
    "requirements.txt",
    "requirements-real.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "autolabeler/cli.py",
    "scripts/smoke_test.py",
    "scripts/real_model_test.py",
]

README_TERMS = {
    "GroundingDINO": ["groundingdino"],
    "SAM2": ["sam2"],
    "OpenCV": ["opencv"],
    "Streamlit": ["streamlit"],
    "YOLO": ["yolo"],
    "COCO": ["coco"],
    "mock mode": ["mock mode", "mock 모드"],
    "real mode": ["real mode", "real 모드"],
}

EXPECTED_MOCK_OUTPUTS = [
    "previews",
    "yolo_det/data.yaml",
    "yolo_det/images",
    "yolo_det/labels",
    "yolo_seg/data.yaml",
    "yolo_seg/images",
    "yolo_seg/labels",
    "coco/annotations.json",
    "quality_report.csv",
    "quality_report.md",
    "autolabeler_output.zip",
]


class Reporter:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def ok(self, message: str) -> None:
        print(f"[OK] {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"[WARN] {message}")

    def fail(self, message: str) -> None:
        self.failures.append(message)
        print(f"[FAIL] {message}")

    def summary(self) -> int:
        print()
        print("Final check summary")
        print(f"  OK/WARN/FAIL: warnings={len(self.warnings)}, failures={len(self.failures)}")
        if self.failures:
            print("  Result: FAIL")
            return 1
        print("  Result: PASS")
        return 0


def _run_command(reporter: Reporter, args: list[str], description: str) -> bool:
    printable = " ".join(args)
    reporter.ok(f"running {description}: {printable}")
    try:
        completed = subprocess.run(
            args,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
            check=False,
        )
    except FileNotFoundError as e:
        reporter.fail(f"{description} 실행 파일을 찾을 수 없습니다: {e}")
        return False
    except subprocess.TimeoutExpired:
        reporter.fail(f"{description} 시간이 초과되었습니다: {printable}")
        return False

    if completed.returncode != 0:
        reporter.fail(f"{description} 실패(exit={completed.returncode})")
        if completed.stdout:
            print(completed.stdout)
        return False

    if completed.stdout:
        print(completed.stdout.rstrip())
    reporter.ok(f"{description} 통과")
    return True


def _check_required_files(reporter: Reporter) -> None:
    for rel in REQUIRED_FILES:
        path = ROOT / rel
        if path.is_file():
            reporter.ok(f"required file exists: {rel}")
        else:
            reporter.fail(f"required file missing: {rel}")


def _check_readme(reporter: Reporter) -> None:
    readme = ROOT / "README.md"
    if not readme.is_file():
        reporter.fail("README.md가 없습니다.")
        return

    text = readme.read_text(encoding="utf-8")
    if text.strip():
        reporter.ok("README.md is not empty")
    else:
        reporter.fail("README.md가 비어 있습니다.")
        return

    lower = text.lower()
    for label, needles in README_TERMS.items():
        if any(needle in lower for needle in needles):
            reporter.ok(f"README mentions {label}")
        else:
            reporter.fail(f"README missing reference: {label}")

    image_refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    screenshot_refs = []
    for raw_ref in image_refs:
        ref = raw_ref.strip().strip("<>")
        ref = ref.split()[0].split("#", 1)[0].split("?", 1)[0]
        if ref.startswith("assets/screenshots/"):
            screenshot_refs.append(ref)

    if not screenshot_refs:
        reporter.warn("README에 assets/screenshots 이미지 참조가 없습니다.")
    for ref in screenshot_refs:
        if (ROOT / ref).is_file():
            reporter.ok(f"README image exists: {ref}")
        else:
            reporter.fail(f"README image missing: {ref}")


def _check_assets_and_samples(reporter: Reporter) -> None:
    screenshots = ROOT / "assets" / "screenshots"
    if screenshots.is_dir():
        reporter.ok("assets/screenshots exists")
    else:
        reporter.fail("assets/screenshots 폴더가 없습니다.")

    sample_images = ROOT / "sample_images"
    if not sample_images.is_dir():
        reporter.fail("sample_images 폴더가 없습니다.")
        return

    images = [
        p
        for p in sample_images.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if images:
        reporter.ok(f"sample_images contains {len(images)} supported image(s)")
    else:
        reporter.fail("sample_images에 지원 이미지 파일이 없습니다.")


def _git_output(args: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError:
        return None


def _check_distribution_hygiene(reporter: Reporter) -> None:
    existing = sorted(d for d in FORBIDDEN_DIST_DIRS if (ROOT / d).exists())
    if existing:
        reporter.warn(
            "로컬 작업 폴더에 배포 제외 대상이 존재합니다: "
            + ", ".join(existing)
            + ". git archive에는 포함되지 않아야 합니다."
        )
    else:
        reporter.ok("forbidden distribution directories are absent locally")

    ls_files = _git_output(["ls-files"])
    if ls_files is None:
        reporter.warn("git 명령을 찾을 수 없어 추적 파일 검사를 건너뜁니다.")
        return
    if ls_files.returncode != 0:
        reporter.warn(f"git ls-files 실패: {ls_files.stdout.strip()}")
        return

    tracked_forbidden = []
    for line in ls_files.stdout.splitlines():
        first = line.split("/", 1)[0]
        if first in FORBIDDEN_DIST_DIRS or line.endswith(".pyc"):
            tracked_forbidden.append(line)
    if tracked_forbidden:
        reporter.fail("배포 제외 대상이 Git에 추적 중입니다: " + ", ".join(tracked_forbidden))
    else:
        reporter.ok("no forbidden generated paths are tracked by Git")

    status = _git_output(["status", "--porcelain"])
    if status is not None and status.returncode == 0 and status.stdout.strip():
        reporter.warn("커밋 전 변경사항이 있습니다. git archive HEAD에는 아직 포함되지 않습니다.")

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        proc = subprocess.run(
            ["git", "archive", "--format=zip", "--output", str(tmp_path), "HEAD"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    try:
        if proc.returncode != 0:
            reporter.warn(f"임시 source zip 생성 실패: {proc.stdout.strip()}")
            return
        with zipfile.ZipFile(tmp_path) as zf:
            bad = []
            for name in zf.namelist():
                first = name.split("/", 1)[0]
                if first in FORBIDDEN_DIST_DIRS or name.endswith(".pyc"):
                    bad.append(name)
        if bad:
            reporter.fail("git archive에 배포 제외 대상이 포함됩니다: " + ", ".join(bad))
        else:
            reporter.ok("git archive source zip is clean")
    finally:
        tmp_path.unlink(missing_ok=True)


def _check_mock_outputs(reporter: Reporter, out_dir: Path) -> None:
    for rel in EXPECTED_MOCK_OUTPUTS:
        path = out_dir / rel
        if path.exists():
            reporter.ok(f"mock output exists: {rel}")
        else:
            reporter.fail(f"mock output missing: {rel}")

    for rel in ["yolo_det/data.yaml", "yolo_seg/data.yaml"]:
        path = out_dir / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if "path: ." in text and "train: images" in text and "val: images" in text:
            reporter.ok(f"portable YOLO data.yaml: {rel}")
        else:
            reporter.fail(f"YOLO data.yaml is not portable: {rel}")


def main() -> int:
    reporter = Reporter()

    _check_required_files(reporter)
    _check_readme(reporter)
    _check_assets_and_samples(reporter)
    _check_distribution_hygiene(reporter)

    _run_command(
        reporter,
        [sys.executable, "scripts/smoke_test.py"],
        "smoke test",
    )
    _run_command(
        reporter,
        [sys.executable, "-m", "pytest", "-q"],
        "pytest",
    )

    mock_out = ROOT / "runs" / "final_check_mock"
    if mock_out.exists():
        shutil.rmtree(mock_out)
    if _run_command(
        reporter,
        [
            sys.executable,
            "-m",
            "autolabeler.cli",
            "--images",
            "sample_images",
            "--classes",
            "object",
            "--out",
            str(mock_out),
            "--mock",
        ],
        "CLI mock final check",
    ):
        _check_mock_outputs(reporter, mock_out)

    return reporter.summary()


if __name__ == "__main__":
    sys.exit(main())
