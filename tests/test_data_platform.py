import pandas as pd
import pytest
from data_platform.ingestion import to_long_format
from data_platform.quality import validate_demand


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
