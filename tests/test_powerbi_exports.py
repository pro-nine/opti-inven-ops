import duckdb
import pandas as pd
import pytest
from data_platform.export_powerbi import export_powerbi


def database(tmp_path):
    db = tmp_path / "warehouse.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE demand AS SELECT DATE '2023-01-02' AS date, 'A_Yoghurt' AS sku, 10 AS demand")
    con.execute("CREATE TABLE empty_table (sku VARCHAR)")
    con.close()
    return db


def test_export_roundtrip_and_repeated_run(tmp_path):
    db = database(tmp_path)
    out = tmp_path / "manager's exports"
    export_powerbi(db, out)
    first = (out / "demand.csv").read_bytes()
    export_powerbi(db, out)
    assert (out / "demand.csv").read_bytes() == first
    frame = pd.read_csv(out / "demand.csv")
    assert frame.loc[0, "demand"] == 10
    assert frame.loc[0, "sku"] == "A_Yoghurt"
    assert list(pd.read_csv(out / "empty_table.csv").columns) == ["sku"]
    assert not list(out.glob("*.tmp"))


def test_missing_table_fails_before_writing_anything(tmp_path):
    db = database(tmp_path)
    out = tmp_path / "exports"
    with pytest.raises(ValueError, match="Missing warehouse tables"):
        export_powerbi(db, out, ["demand", "missing"])
    assert not list(out.glob("*.csv"))


def test_explicit_empty_selection_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="No warehouse tables"):
        export_powerbi(database(tmp_path), tmp_path / "exports", [])
