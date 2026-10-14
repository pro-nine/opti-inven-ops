"""Strict CSV exports for Power BI ingestion; no Power BI service deployment."""
from pathlib import Path
import os
import tempfile
import duckdb


def export_powerbi(db_path: str | Path, output_dir: str | Path, tables: list[str] | None = None) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        available = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        requested = sorted(available) if tables is None else tables
        if not requested:
            raise ValueError("No warehouse tables selected for export")
        missing = set(requested) - available
        if missing:
            raise ValueError(f"Missing warehouse tables: {sorted(missing)}")
        for table in requested:
            if Path(table).name != table or table in {".", ".."} or "\\" in table:
                raise ValueError(f"Unsafe export filename: {table!r}")
            identifier = '"' + table.replace('"', '""') + '"'
            frame = con.execute(f"SELECT * FROM {identifier}").fetchdf()
            fd, temp = tempfile.mkstemp(dir=output, suffix=".tmp")
            os.close(fd)
            try:
                frame.to_csv(temp, index=False, encoding="utf-8")
                os.replace(temp, output / f"{table}.csv")
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
    finally:
        con.close()
