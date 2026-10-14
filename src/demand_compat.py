"""SciPy-compatible residual fitting used by the integrated analytics runner.

Frozen CDF callables avoid named-distribution dispatch to bare ndtr. The return
schema and selection rules match module1_demand; fitted KS p-values remain
reference diagnostics, not calibrated tests after parameter estimation.
"""
import numpy as np
from scipy import stats


def fit_residual_distribution(residuals):
    r = np.asarray(residuals, dtype=float)
    r = r[~np.isnan(r)]
    sw_stat, sw_p = stats.shapiro(r)
    mu, sigma = stats.norm.fit(r)
    ks_stat, ks_p = stats.kstest(r, stats.norm(loc=mu, scale=sigma).cdf)
    results = {"normal": {"params": {"mu": mu, "sigma": sigma},
        "sw_stat": sw_stat, "sw_p": sw_p, "ks_stat": ks_stat,
        "ks_p": ks_p, "fits": sw_p > 0.05}}
    shift = r.min() - 0.01 if r.min() <= 0 else 0
    shifted = r - shift
    try:
        a, loc, scale = stats.gamma.fit(shifted, floc=0)
        ks_g, ks_pg = stats.kstest(shifted, stats.gamma(a, loc=loc, scale=scale).cdf)
        results["gamma"] = {"params": {"a": a, "loc": loc + shift, "scale": scale},
            "ks_stat": ks_g, "ks_p": ks_pg, "fits": ks_pg > 0.05}
    except Exception:
        results["gamma"] = {"fits": False, "ks_p": 0}
    results["empirical"] = {"params": {"quantiles": np.percentile(r, [1,5,10,25,50,75,90,95,99])},
        "fits": True, "ks_p": 1.0}
    best = "normal" if results["normal"]["fits"] else (
        "gamma" if results["gamma"]["fits"] else "empirical")
    return {"distributions": results, "best_fit": best}


def run_module1_compatible(**kwargs):
    """Scoped adapter; restores the module's original fitter even on failure.

    Does not change scipy.stats globally or suppress the analytics runner.
    """
    from src import module1_demand
    original = module1_demand.fit_residual_distribution
    module1_demand.fit_residual_distribution = fit_residual_distribution
    try:
        return module1_demand.run_module1(**kwargs)
    finally:
        module1_demand.fit_residual_distribution = original
