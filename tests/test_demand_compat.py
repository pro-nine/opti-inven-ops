import numpy as np
from scipy import stats
from src.demand_compat import fit_residual_distribution


def test_fitter_avoids_named_distribution_dispatch(monkeypatch):
    original = stats.kstest
    calls = []
    def guard(samples, cdf, *args, **kwargs):
        assert callable(cdf), "CDF must be explicit, not a distribution string"
        assert not kwargs.get("args"), "Parameters must be bound in frozen CDF"
        calls.append(cdf)
        return original(samples, cdf, *args, **kwargs)
    monkeypatch.setattr(stats, "kstest", guard)
    result = fit_residual_distribution(np.random.default_rng(42).normal(3, 2, 156))
    assert len(calls) == 2
    for name in ["normal", "gamma"]:
        assert 0 <= result["distributions"][name]["ks_p"] <= 1


def test_fitter_preserves_normal_fit_and_empirical_quantiles():
    x = np.random.default_rng(7).normal(10, 3, 156)
    result = fit_residual_distribution(x)
    assert np.isclose(result["distributions"]["normal"]["params"]["mu"], x.mean())
    assert np.allclose(result["distributions"]["empirical"]["params"]["quantiles"],
                       np.percentile(x, [1,5,10,25,50,75,90,95,99]))
