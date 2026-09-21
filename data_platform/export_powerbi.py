from pathlib import Path
import duckdb


def export_powerbi(db_path: str | Path, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path), read_only=True)
    for table in ["raw_demand", "demand", "demand_summary"]:
        con.execute(f"COPY {table} TO '{(output / (table + '.csv')).as_posix()}' (HEADER, DELIMITER ',')")
    con.close()
