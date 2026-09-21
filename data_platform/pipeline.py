from __future__ import annotations
from datetime import datetime, timezone
import time
import pandas as pd
from .analytics_exports import output_manifest, read_output_tables, run_existing_analytics
from .config import EXPORT_DIR, SOURCE_FILE, WAREHOUSE_FILE
from .ingestion import load_demand_csv, to_long_format
from .quality import validate_demand
from .warehouse import build_warehouse, list_tables
from .export_powerbi import export_powerbi


def run() -> dict:
    started = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw = load_demand_csv(SOURCE_FILE)
    demand = to_long_format(raw)
    quality = validate_demand(demand)
    analytics = run_existing_analytics()
    analytics_tables = read_output_tables(analytics["files"])
    manifest = output_manifest(analytics["files"], analytics_tables)
    log = pd.DataFrame([{"run_id": run_id, "started_at_utc": run_id, "status": "SUCCESS", "source_rows": int(len(raw)), "demand_rows": int(len(demand)), "analytics_tables": int(len(analytics_tables)), "duration_seconds": round(time.perf_counter() - started, 3), "error": ""}])
    build_warehouse(raw, demand, WAREHOUSE_FILE, analytics_tables, log)
    export_powerbi(WAREHOUSE_FILE, EXPORT_DIR)
    return {"status": "SUCCESS", "quality": quality, "tables": list_tables(WAREHOUSE_FILE), "manifest": manifest.to_dict("records")}


if __name__ == "__main__":
    print(run())
