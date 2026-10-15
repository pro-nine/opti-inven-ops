import duckdb
import pandas as pd
import pytest
from data_platform import pipeline as p
from data_platform.run_history import record_run
from data_platform.warehouse import build_warehouse


def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(p, "WAREHOUSE_FILE", tmp_path / "warehouse.duckdb")
    monkeypatch.setattr(p, "EXPORT_DIR", tmp_path / "exports")
    raw = pd.DataFrame({"date": pd.to_datetime(["2023-01-02"]), "SKU_A_Yoghurt": [10]})
    monkeypatch.setattr(p, "load_demand_csv", lambda _: raw)
    monkeypatch.setattr(p, "run_existing_analytics", lambda: {"files": []})
    monkeypatch.setattr(p, "read_output_tables", lambda _: {})
    monkeypatch.setattr(p, "output_manifest", lambda *_: pd.DataFrame())


def logs(db):
    with duckdb.connect(str(db), read_only=True) as con:
        return con.execute("SELECT * FROM run_log ORDER BY started_at_utc, run_id").fetchdf()


def test_two_successes_preserve_history_and_export(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path)
    a, b = p.run(), p.run()
    assert a["run_id"] != b["run_id"]
    history = logs(p.WAREHOUSE_FILE)
    assert len(history) == 2 and set(history.status) == {"SUCCESS"}
    assert len(pd.read_csv(p.EXPORT_DIR / "run_log.csv")) == 2


@pytest.mark.parametrize("failure_stage", ["quality", "analytics", "warehouse", "export"])
def test_failure_is_logged_and_original_exception_is_raised(monkeypatch, tmp_path, failure_stage):
    setup(monkeypatch, tmp_path)
    p.run()
    original = RuntimeError("injected failure")
    def fail(*args, **kwargs): raise original
    target = {"quality": "validate_demand", "analytics": "run_existing_analytics",
              "warehouse": "build_warehouse", "export": "export_powerbi"}[failure_stage]
    monkeypatch.setattr(p, target, fail)
    with pytest.raises(RuntimeError) as caught: p.run()
    assert caught.value is original
    history = logs(p.WAREHOUSE_FILE)
    assert len(history) == 2
    failed = history[history.status == "FAILED"].iloc[0]
    assert failed.stage == failure_stage and "injected failure" in failed.error
    assert failed.finished_at_utc


def test_migrate_old_log_without_deleting_old_rows(tmp_path):
    db = tmp_path / "legacy.duckdb"
    with duckdb.connect(str(db)) as con:
        con.execute("CREATE TABLE run_log AS SELECT 'old' AS run_id, '2020' AS started_at_utc, 'SUCCESS' AS status, 1 AS source_rows, 1 AS demand_rows, 0 AS analytics_tables, 0.0 AS duration_seconds, '' AS error")
    record_run(db, {"run_id": "new", "started_at_utc": "2026", "status": "FAILED",
                    "source_rows": 0, "demand_rows": 0, "analytics_tables": 0,
                    "duration_seconds": 1.0, "error": "failure", "stage": "quality", "finished_at_utc": "2026"})
    assert set(logs(db).run_id) == {"old", "new"}


def test_warehouse_transaction_rolls_back_reserved_table_collision(tmp_path):
    db = tmp_path / "transaction.duckdb"
    raw = pd.DataFrame({"date": pd.to_datetime(["2023-01-02"]), "SKU_A_Yoghurt": [10]})
    demand = pd.DataFrame({"date": raw.date, "sku": ["A_Yoghurt"], "demand": [10]})
    build_warehouse(raw, demand, db)
    changed = demand.assign(demand=99)
    with pytest.raises(ValueError): build_warehouse(raw, changed, db, {"run_log": pd.DataFrame({"x": [1]})})
    with duckdb.connect(str(db)) as con:
        assert con.execute("SELECT demand FROM demand").fetchone()[0] == 10
