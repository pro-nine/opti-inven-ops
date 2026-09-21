import pandas as pd
import pytest
from data_platform.ingestion import to_long_format
from data_platform.quality import validate_demand
from data_platform.warehouse import build_warehouse, list_tables


def sample():
    return pd.DataFrame({"date": pd.to_datetime(["2023-01-02"]), "SKU_A_Yoghurt": [10], "SKU_B_Detergent": [20], "SKU_C_XmasCookies": [30], "SKU_D_Coffee": [40], "SKU_E_Bread": [50], "SKU_F_Cleaning": [60]})


def test_to_long_format():
    result = to_long_format(sample())
    assert result.shape == (6, 3)
    assert set(result.columns) == {"date", "sku", "demand"}


def test_quality_passes():
    result = validate_demand(to_long_format(sample()))
    assert result["passed"] is True
    assert result["sku_count"] == 6


def test_negative_demand_fails():
    data = to_long_format(sample())
    data.loc[0, "demand"] = -1
    with pytest.raises(ValueError):
        validate_demand(data)


def test_warehouse_contains_core_tables(tmp_path):
    raw = sample()
    long_df = to_long_format(raw)
    db = tmp_path / "warehouse.duckdb"
    build_warehouse(raw, long_df, db, {"inventory_policy": pd.DataFrame({"sku": ["A"], "q": [10]})})
    tables = list_tables(db)
    assert {"raw_demand", "demand", "demand_summary", "inventory_policy"}.issubset(tables)
