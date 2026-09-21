from pathlib import Path
import duckdb
import pandas as pd


def build_warehouse(raw: pd.DataFrame, demand_long: pd.DataFrame, db_path: str | Path, analytics_tables: dict[str, pd.DataFrame] | None = None, run_log: pd.DataFrame | None = None) -> None:
    con = duckdb.connect(str(db_path))
    con.register("raw_df", raw)
    con.register("demand_df", demand_long)
    con.execute("CREATE OR REPLACE TABLE raw_demand AS SELECT * FROM raw_df")
    con.execute("CREATE OR REPLACE TABLE demand AS SELECT date, sku, demand FROM demand_df")
    con.execute("CREATE OR REPLACE TABLE demand_summary AS SELECT sku, MIN(date) AS first_date, MAX(date) AS last_date, COUNT(*) AS observations, AVG(demand) AS average_demand, STDDEV_SAMP(demand) AS demand_stddev, MIN(demand) AS minimum_demand, MAX(demand) AS maximum_demand FROM demand GROUP BY sku ORDER BY sku")
    for name, frame in (analytics_tables or {}).items():
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            safe_name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
            con.register("analytics_frame", frame)
            con.execute(f'CREATE OR REPLACE TABLE "{safe_name}" AS SELECT * FROM analytics_frame')
            con.unregister("analytics_frame")
    if run_log is not None:
        con.register("run_log_frame", run_log)
        con.execute("CREATE OR REPLACE TABLE run_log AS SELECT * FROM run_log_frame")
        con.unregister("run_log_frame")
    con.close()


def list_tables(db_path: str | Path) -> list[str]:
    con = duckdb.connect(str(db_path), read_only=True)
    result = con.execute("SHOW TABLES").fetchdf()["name"].tolist()
    con.close()
    return result
