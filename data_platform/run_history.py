"""Persistent run records; updating one run never replaces earlier history."""
from pathlib import Path
import duckdb
import pandas as pd


def record_run(db_path, row):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute("BEGIN TRANSACTION")
        con.execute("""CREATE TABLE IF NOT EXISTS run_log (
            run_id VARCHAR, started_at_utc VARCHAR, status VARCHAR,
            source_rows BIGINT, demand_rows BIGINT, analytics_tables BIGINT,
            duration_seconds DOUBLE, error VARCHAR)""")
        con.execute("ALTER TABLE run_log ADD COLUMN IF NOT EXISTS stage VARCHAR")
        con.execute("ALTER TABLE run_log ADD COLUMN IF NOT EXISTS finished_at_utc VARCHAR")
        frame = pd.DataFrame([row])
        con.register("current_run", frame)
        con.execute("DELETE FROM run_log WHERE run_id = ?", [row["run_id"]])
        con.execute("INSERT INTO run_log BY NAME SELECT * FROM current_run")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()
