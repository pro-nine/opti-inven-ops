from pathlib import Path
from .config import EXPORT_DIR, SOURCE_FILE, WAREHOUSE_FILE
from .ingestion import load_demand_csv, to_long_format
from .quality import validate_demand
from .warehouse import build_warehouse
from .export_powerbi import export_powerbi


def run() -> dict:
    raw = load_demand_csv(SOURCE_FILE)
    demand = to_long_format(raw)
    quality = validate_demand(demand)
    build_warehouse(raw, demand, WAREHOUSE_FILE)
    export_powerbi(WAREHOUSE_FILE, EXPORT_DIR)
    return quality


if __name__ == "__main__":
    print(run())
