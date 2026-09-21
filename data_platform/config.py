from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILE = ROOT / "data" / "demand_weekly.csv"
WAREHOUSE_FILE = ROOT / "data" / "warehouse.duckdb"
EXPORT_DIR = ROOT / "data" / "powerbi"
