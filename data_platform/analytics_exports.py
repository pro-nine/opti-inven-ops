from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ROOT


def _csv_files_before(root: Path) -> set[Path]:
    return set(root.rglob("*.csv"))


def _new_csv_files(root: Path, before: set[Path]) -> list[Path]:
    return sorted(p for p in _csv_files_before(root) - before if p.is_file())


def run_existing_analytics(output_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(output_root or ROOT)
    before = _csv_files_before(root)
    result = subprocess.run(["python", "-m", "src.run_all"], cwd=root, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError("Existing analytics runner failed.\n" + result.stdout[-4000:] + "\n" + result.stderr[-4000:])
    return {"returncode": result.returncode, "stdout": result.stdout, "files": _new_csv_files(root, before)}


def read_output_tables(files: list[Path]) -> dict[str, pd.DataFrame]:
    tables = {}
    for path in files:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if not frame.empty:
            tables[path.stem.lower().replace("-", "_").replace(" ", "_")] = frame
    return tables


def output_manifest(files: list[Path], tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame([{"file_name": str(path.relative_to(ROOT)), "table_name": path.stem.lower().replace("-", "_").replace(" ", "_"), "row_count": int(len(tables.get(path.stem.lower().replace("-", "_").replace(" ", "_"), pd.DataFrame()))), "column_count": int(tables.get(path.stem.lower().replace("-", "_").replace(" ", "_"), pd.DataFrame()).shape[1])} for path in files])
