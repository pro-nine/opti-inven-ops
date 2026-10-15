from __future__ import annotations
from datetime import datetime, timezone
from uuid import uuid4
import time
from .analytics_exports import output_manifest, read_output_tables, run_existing_analytics
from .config import EXPORT_DIR, SOURCE_FILE, WAREHOUSE_FILE
from .ingestion import load_demand_csv, to_long_format
from .quality import validate_demand
from .warehouse import build_warehouse, list_tables
from .export_powerbi import export_powerbi
from .run_history import record_run


def run() -> dict:
    started = time.perf_counter()
    row = {"run_id": uuid4().hex, "started_at_utc": datetime.now(timezone.utc).isoformat(),
           "status": "RUNNING", "source_rows": 0, "demand_rows": 0, "analytics_tables": 0,
           "duration_seconds": 0.0, "error": "", "stage": "initialise", "finished_at_utc": ""}
    try:
        record_run(WAREHOUSE_FILE, row)
        row["stage"] = "ingestion"
        raw = load_demand_csv(SOURCE_FILE)
        row["source_rows"] = len(raw)
        demand = to_long_format(raw)
        row["demand_rows"] = len(demand)
        row["stage"] = "quality"
        quality = validate_demand(demand)
        row["stage"] = "analytics"
        analytics = run_existing_analytics()
        row["stage"] = "read_outputs"
        analytics_tables = read_output_tables(analytics["files"])
        row["analytics_tables"] = len(analytics_tables)
        manifest = output_manifest(analytics["files"], analytics_tables)
        row["stage"] = "warehouse"
        build_warehouse(raw, demand, WAREHOUSE_FILE, analytics_tables)
        row["stage"] = "export"
        export_powerbi(WAREHOUSE_FILE, EXPORT_DIR)
        row.update(status="SUCCESS", stage="complete",
                   duration_seconds=round(time.perf_counter() - started, 3),
                   finished_at_utc=datetime.now(timezone.utc).isoformat())
        record_run(WAREHOUSE_FILE, row)
        row["stage"] = "export_run_log"
        export_powerbi(WAREHOUSE_FILE, EXPORT_DIR, tables=["run_log"])
        return {"status": "SUCCESS", "run_id": row["run_id"], "quality": quality,
                "tables": list_tables(WAREHOUSE_FILE), "manifest": manifest.to_dict("records")}
    except Exception as exc:
        row.update(status="FAILED", error=f"{type(exc).__name__}: {exc}"[:8000],
                   duration_seconds=round(time.perf_counter() - started, 3),
                   finished_at_utc=datetime.now(timezone.utc).isoformat())
        try:
            record_run(WAREHOUSE_FILE, row)
        except Exception as log_error:
            exc.add_note(f"Run history persistence also failed: {log_error}")
        raise


if __name__ == "__main__":
    print(run())
