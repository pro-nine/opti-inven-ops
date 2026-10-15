from pathlib import Path
import duckdb
import pandas as pd
from .run_history import record_run


def build_warehouse(raw: pd.DataFrame, demand_long: pd.DataFrame, db_path: str | Path, analytics_tables: dict[str, pd.DataFrame] | None = None, run_log: pd.DataFrame | None = None) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute("BEGIN TRANSACTION")
        con.register("raw_df", raw)
        con.register("demand_df", demand_long)
        con.execute("CREATE OR REPLACE TABLE raw_demand AS SELECT * FROM raw_df")
        con.execute("CREATE OR REPLACE TABLE demand AS SELECT date, sku, demand FROM demand_df")
        con.execute("CREATE OR REPLACE TABLE demand_summary AS SELECT sku, MIN(date) AS first_date, MAX(date) AS last_date, COUNT(*) AS observations, AVG(demand) AS average_demand, STDDEV_SAMP(demand) AS demand_stddev, MIN(demand) AS minimum_demand, MAX(demand) AS maximum_demand FROM demand GROUP BY sku ORDER BY sku")
        for name, frame in (analytics_tables or {}).items():
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                safe_name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
                if safe_name in {"raw_demand", "demand", "demand_summary", "run_log"}:
                    raise ValueError(f"Reserved warehouse table: {safe_name}")
                con.register("analytics_frame", frame)
                con.execute(f'CREATE OR REPLACE TABLE "{safe_name}" AS SELECT * FROM analytics_frame')
                con.unregister("analytics_frame")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()
    if run_log is not None:
        for row in run_log.to_dict("records"):
            record_run(db_path, row)


def list_tables(db_path: str | Path) -> list[str]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute("SHOW TABLES").fetchdf()["name"].tolist()
    finally:
        con.close()
