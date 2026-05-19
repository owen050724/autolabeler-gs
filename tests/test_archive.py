import zipfile
from pathlib import Path

from autolabeler.exporters.archive import make_zip


def test_make_zip_skips_archive_itself(tmp_path: Path):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    (out_dir / "label.txt").write_text("0 0.5 0.5 1 1\n", encoding="utf-8")

    zip_path = make_zip(out_dir, out_dir / "autolabeler_output.zip")

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    assert "label.txt" in names
    assert "autolabeler_output.zip" not in names
