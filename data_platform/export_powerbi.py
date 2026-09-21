from pathlib import Path
import duckdb


def export_powerbi(db_path: str | Path, output_dir: str | Path, tables: list[str] | None = None) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=True)
    available = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    for table in tables or sorted(available):
        if table in available:
            destination = output / f"{table}.csv"
            con.execute(f"COPY \"{table}\" TO '{destination.as_posix()}' (HEADER, DELIMITER ',')")
    con.close()
