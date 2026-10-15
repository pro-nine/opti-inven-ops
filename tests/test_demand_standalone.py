import numpy as np
import pytest
from scipy import stats
from src import module1_demand as demand


def test_direct_fitter_uses_explicit_cdfs(monkeypatch):
    original = stats.kstest
    calls = []
    def checked(samples, cdf, *args, **kwargs):
        assert callable(cdf)
        assert not kwargs.get("args")
        calls.append(cdf)
        return original(samples, cdf, *args, **kwargs)
    monkeypatch.setattr(stats, "kstest", checked)
    result = demand.fit_residual_distribution(np.random.default_rng(42).normal(3, 2, 156))
    assert len(calls) == 2
    for name in ["normal", "gamma"]:
        assert 0 <= result["distributions"][name]["ks_p"] <= 1


def test_module1_direct_run_completes_without_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(demand, "RNG", np.random.default_rng(42))
    figures, data = tmp_path / "figures", tmp_path / "data"
    result = demand.run_module1(fig_dir=str(figures), data_dir=str(data))
    assert result["df"].shape == (156, 7)
    assert len(result["fit_results"]) == 6
    assert len(list(figures.glob("*.png"))) == 5
    assert (data / "demand_weekly.csv").exists()
