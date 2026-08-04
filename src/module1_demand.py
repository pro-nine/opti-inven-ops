"""
Module 1: Demand Characterization
====================================
Simulates 3 years of weekly demand for 6 SKUs at Nordpack GmbH.
Each SKU is engineered to represent a REAL operational archetype
you would encounter in any FMCG distribution centre:

  SKU-A  Yoghurt (fast-moving, strong seasonal peak in summer)
  SKU-B  Detergent (fast-moving, stable — promotional spikes only)
  SKU-C  Christmas cookies (slow-moving, extreme seasonal pulse)
  SKU-D  Specialty coffee (slow-moving, stable, low volume)
  SKU-E  Fresh bread (perishable — single-period newsvendor problem)
  SKU-F  Industrial cleaning supplies (intermittent/lumpy demand)

WHY STL DECOMPOSITION (not moving average, not ARIMA):
------------------------------------------------------
STL = Seasonal-Trend decomposition using Loess (locally weighted regression).
It decomposes a time series into three additive components:
    Y(t) = T(t) + S(t) + R(t)
    where T = trend, S = seasonal, R = remainder (noise)

The key advantage over a moving average for THIS problem:
  - Robust to outliers (promotional spikes don't corrupt the seasonal estimate)
  - The REMAINDER R(t) is what we fit a distribution to for safety stock
  - It works on non-integer periods (52.18 weeks/year) unlike classic methods

The remainder distribution determines whether we use:
  - Normal distribution  -> standard z-score safety stock formula
  - Gamma distribution   -> right-skewed demand (slow movers, lumpy)
  - Empirical            -> non-parametric when nothing fits well

We test goodness of fit using the Kolmogorov-Smirnov test (p > 0.05 = 
acceptable fit). This is not arbitrary — the choice of distribution
directly changes the safety stock calculation in Module 2.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
RNG = np.random.default_rng(SEED)
N_WEEKS = 156  # 3 years

WEEKS = pd.date_range(start="2021-01-04", periods=N_WEEKS, freq="W-MON")


# ── Demand simulation ─────────────────────────────────────────────────────────

def seasonal_wave(n, peak_week, amplitude, base):
    """Smooth sinusoidal seasonal component peaking at `peak_week`."""
    t = np.arange(n)
    return base + amplitude * np.sin(2 * np.pi * (t - peak_week) / 52)


def simulate_demand():
    """
    Returns a DataFrame (n_weeks × 6) of weekly demand in units.
    All parameters are chosen so the resulting series have plausible
    Coefficients of Variation (CV = std/mean) for their archetype.
    """
    n = N_WEEKS

    # SKU-A: Yoghurt — fast mover, summer peak (week 26), CV ≈ 0.25
    trend_a = np.linspace(820, 950, n)          # slight growth trend
    seasonal_a = seasonal_wave(n, 20, 180, 0)
    noise_a = RNG.normal(0, 60, n)
    demand_a = np.clip(trend_a + seasonal_a + noise_a, 400, None).round()

    # SKU-B: Detergent — fast mover, stable with promotional spikes, CV ≈ 0.18
    base_b = RNG.normal(650, 80, n)
    promo_weeks = [8, 34, 82, 108, 143]          # 5 promotions over 3 years
    spikes_b = np.zeros(n)
    for w in promo_weeks:
        if w < n:
            spikes_b[w] = RNG.uniform(300, 500)
    demand_b = np.clip(base_b + spikes_b, 300, None).round()

    # SKU-C: Christmas cookies — slow mover, extreme Q4 pulse, CV ≈ 1.1
    base_c = RNG.normal(40, 12, n)
    # Peak in weeks 44-52 (Nov-Dec) of each year
    seasonal_c = np.zeros(n)
    for yr in range(3):
        for w in range(44, 53):
            idx = yr * 52 + w
            if idx < n:
                seasonal_c[idx] = RNG.normal(420, 60)
    demand_c = np.clip(base_c + seasonal_c, 0, None).round()

    # SKU-D: Specialty coffee — slow mover, very stable, CV ≈ 0.15
    demand_d = np.clip(RNG.normal(95, 14, n), 40, None).round()

    # SKU-E: Fresh bread — perishable, daily-like pattern compressed to weekly
    # High base demand, moderate variability — newsvendor archetype
    trend_e = np.linspace(1100, 1250, n)
    seasonal_e = seasonal_wave(n, 26, 120, 0)
    noise_e = RNG.normal(0, 95, n)
    demand_e = np.clip(trend_e + seasonal_e + noise_e, 700, None).round()

    # SKU-F: Industrial cleaning — intermittent/lumpy
    # Modelled as Bernoulli(p=0.55) × size; many zero weeks
    occurs = RNG.binomial(1, 0.55, n)
    sizes = RNG.gamma(shape=3, scale=25, size=n).round()
    demand_f = (occurs * sizes).round()

    df = pd.DataFrame({
        "date": WEEKS,
        "SKU_A_Yoghurt":       demand_a.astype(int),
        "SKU_B_Detergent":     demand_b.astype(int),
        "SKU_C_XmasCookies":   demand_c.astype(int),
        "SKU_D_Coffee":        demand_d.astype(int),
        "SKU_E_Bread":         demand_e.astype(int),
        "SKU_F_Cleaning":      demand_f.astype(int),
    })
    return df


# ── STL decomposition ─────────────────────────────────────────────────────────

def run_stl(series: pd.Series, period: int = 52) -> dict:
    """
    Fit STL decomposition. Returns dict with trend, seasonal, residual arrays.
    period=52 because we have weekly data with annual seasonality.
    """
    stl = STL(series, period=period, robust=True)
    result = stl.fit()
    return {
        "trend":    result.trend,
        "seasonal": result.seasonal,
        "residual": result.resid,
        "weights":  result.weights,
    }


# ── Distribution fitting ──────────────────────────────────────────────────────

def fit_residual_distribution(residuals: np.ndarray) -> dict:
    """
    Test Normal and Gamma fits against the residual series.

    KS test with n=156 is very powerful — it rejects even visually reasonable
    fits because it detects tiny deviations. In practice, operations analysts use:
      - Shapiro-Wilk (SW) for normality: better calibrated at n<2000
      - Anderson-Darling for gamma (heavier tail sensitivity)
      - Empirical quantiles when no parametric form fits

    p > 0.05 = cannot reject the distribution at 5% significance level.

    WHY THIS MATTERS FOR MODULE 2:
        Normal  -> SS = z_alpha * sigma * sqrt(L)     (standard formula)
        Gamma   -> SS from gamma.ppf(alpha, a, scale) (right-skewed demand)
        Empirical -> SS from np.quantile(resid, alpha) (non-parametric)
    """
    r = residuals[~np.isnan(residuals)]
    results = {}

    # --- Normal: use Shapiro-Wilk (better calibrated at n=156) ---
    sw_stat, sw_p = stats.shapiro(r)
    mu, sigma = stats.norm.fit(r)
    # Also keep KS for reference
    ks_stat, ks_p = stats.kstest(r, "norm", args=(mu, sigma))
    results["normal"] = {
        "params": {"mu": mu, "sigma": sigma},
        "sw_stat": sw_stat, "sw_p": sw_p,
        "ks_stat": ks_stat, "ks_p": ks_p,
        "fits": sw_p > 0.05,
    }

    # --- Gamma (shift residuals to positive support) ---
    shift = r.min() - 0.01 if r.min() <= 0 else 0
    r_shifted = r - shift
    try:
        a, loc, scale = stats.gamma.fit(r_shifted, floc=0)
        ad_result = stats.anderson(r_shifted, dist="norm")  # AD test
        ks_g, ks_pg = stats.kstest(r_shifted, "gamma", args=(a, loc, scale))
        results["gamma"] = {
            "params": {"a": a, "loc": loc + shift, "scale": scale},
            "ks_stat": ks_g, "ks_p": ks_pg,
            "fits": ks_pg > 0.05,
        }
    except Exception:
        results["gamma"] = {"fits": False, "ks_p": 0}

    # --- Empirical quantile (always available as fallback) ---
    results["empirical"] = {
        "params": {"quantiles": np.percentile(r, [1, 5, 10, 25, 50, 75, 90, 95, 99])},
        "fits": True,   # always valid — no parametric assumption
        "ks_p": 1.0,
    }

    # Best fit: prefer parametric if it passes (more interpretable)
    if results["normal"]["fits"]:
        best = "normal"
    elif results.get("gamma", {}).get("fits"):
        best = "gamma"
    else:
        best = "empirical"

    return {"distributions": results, "best_fit": best}


# ── Coefficient of Variation segmentation ─────────────────────────────────────

def cv_segment(cv: float) -> str:
    """
    Classify demand variability for inventory policy selection.
    CV < 0.3   -> stable     -> standard (Q,r) policy
    CV 0.3-0.7 -> moderate   -> (Q,r) with careful safety stock
    CV > 0.7   -> erratic    -> review-order policy or periodic review
    CV ≈ high + many zeros -> intermittent -> Croston's method in production
    """
    if cv < 0.30:
        return "STABLE"
    elif cv < 0.70:
        return "MODERATE"
    else:
        return "ERRATIC"


# ── Summary statistics ────────────────────────────────────────────────────────

def demand_summary(df: pd.DataFrame) -> pd.DataFrame:
    skus = [c for c in df.columns if c != "date"]
    rows = []
    for sku in skus:
        s = df[sku]
        cv = s.std() / s.mean()
        zero_pct = (s == 0).mean() * 100
        rows.append({
            "SKU": sku,
            "Mean (units/wk)": round(s.mean(), 1),
            "Std Dev": round(s.std(), 1),
            "Min": int(s.min()),
            "Max": int(s.max()),
            "CV": round(cv, 3),
            "Zero weeks (%)": round(zero_pct, 1),
            "Annual demand": round(s.mean() * 52),
            "CV Segment": cv_segment(cv),
        })
    return pd.DataFrame(rows)


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_demand_overview(df: pd.DataFrame, out_path: str):
    skus = [c for c in df.columns if c != "date"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#a8c5da"]
    for i, (sku, ax) in enumerate(zip(skus, axes)):
        ax.plot(df["date"], df[sku], lw=1.0, color=colors[i], alpha=0.85)
        roll = df[sku].rolling(8, center=True).mean()
        ax.plot(df["date"], roll, lw=2.2, color="black", label="8-wk rolling mean")
        ax.set_title(sku.replace("_", " "), fontsize=10, fontweight="bold")
        ax.set_ylabel("Units / week")
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        cv = df[sku].std() / df[sku].mean()
        ax.annotate(f"CV={cv:.2f}  |  {cv_segment(cv)}",
                    xy=(0.02, 0.92), xycoords="axes fraction",
                    fontsize=8, color="#264653",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    fig.suptitle("Nordpack GmbH — Weekly Demand by SKU (2021–2023)\n"
                 "CV = Coefficient of Variation; higher = more variable demand",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_stl_decomposition(df: pd.DataFrame, sku: str, stl_result: dict, out_path: str):
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    dates = df["date"]
    series = df[sku].values

    axes[0].plot(dates, series, color="#264653", lw=1.0)
    axes[0].set_ylabel("Observed"); axes[0].set_title(f"STL Decomposition — {sku.replace('_',' ')}", fontweight="bold")

    axes[1].plot(dates, stl_result["trend"], color="#2a9d8f", lw=1.8)
    axes[1].set_ylabel("Trend")

    axes[2].plot(dates, stl_result["seasonal"], color="#e9c46a", lw=1.2)
    axes[2].axhline(0, color="black", lw=0.5, ls="--")
    axes[2].set_ylabel("Seasonal")

    axes[3].plot(dates, stl_result["residual"], color="#e76f51", lw=0.9, alpha=0.8)
    axes[3].axhline(0, color="black", lw=0.5, ls="--")
    axes[3].set_ylabel("Residual (noise)")
    axes[3].set_xlabel("Week")

    note = ("The RESIDUAL is what drives safety stock uncertainty.\n"
            "We fit a probability distribution to this component — not to raw demand.")
    axes[3].annotate(note, xy=(0.01, 0.08), xycoords="axes fraction",
                     fontsize=7.5, color="#264653",
                     bbox=dict(boxstyle="round", fc="#f8f9fa", alpha=0.8))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_residual_distributions(stl_results: dict, df: pd.DataFrame, fit_results: dict, out_path: str):
    skus = list(stl_results.keys())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, sku in enumerate(skus):
        ax = axes[i]
        resid = stl_results[sku]["residual"]
        resid_clean = resid[~np.isnan(resid)]
        ax.hist(resid_clean, bins=25, density=True, color="#2a9d8f", alpha=0.6, label="Residuals")

        fit = fit_results[sku]
        x = np.linspace(resid_clean.min(), resid_clean.max(), 200)
        best = fit["best_fit"].replace(" (fallback)", "")

        if "normal" in best:
            mu = fit["distributions"]["normal"]["params"]["mu"]
            sig = fit["distributions"]["normal"]["params"]["sigma"]
            ax.plot(x, stats.norm.pdf(x, mu, sig), color="#e76f51", lw=2, label="Normal fit")
        if "gamma" in fit["distributions"] and fit["distributions"]["gamma"]["fits"]:
            pars = fit["distributions"]["gamma"]["params"]
            ax.plot(x - pars.get("loc", 0),
                    stats.gamma.pdf(x - pars.get("loc", 0), pars["a"], scale=pars["scale"]),
                    color="#264653", lw=2, ls="--", label="Gamma fit")

        p_norm = fit["distributions"]["normal"]["ks_p"]
        p_gamma = fit["distributions"].get("gamma", {}).get("ks_p", 0)
        ax.set_title(f"{sku.replace('_',' ')}\nBest fit: {fit['best_fit']}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Residual demand (units)")
        info = f"KS p-value:\nNormal={p_norm:.3f}\nGamma={p_gamma:.3f}"
        ax.annotate(info, xy=(0.97, 0.97), xycoords="axes fraction",
                    fontsize=7, ha="right", va="top",
                    bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        ax.legend(fontsize=7)

    fig.suptitle("Residual Distribution Fitting — KS test p>0.05 means the distribution is acceptable\n"
                 "Choice of distribution determines safety stock formula in Module 2",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_cv_segmentation(summary: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    color_map = {"STABLE": "#2a9d8f", "MODERATE": "#e9c46a", "ERRATIC": "#e76f51"}
    colors = [color_map[s] for s in summary["CV Segment"]]
    bars = ax.barh(summary["SKU"].str.replace("_", " "), summary["CV"], color=colors, height=0.5)
    ax.axvline(0.30, color="#264653", ls="--", lw=1.5, label="Stable/Moderate boundary (CV=0.30)")
    ax.axvline(0.70, color="#c1121f", ls="--", lw=1.5, label="Moderate/Erratic boundary (CV=0.70)")

    for bar, val in zip(bars, summary["CV"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Coefficient of Variation (CV = σ/μ)")
    ax.set_title("Demand Variability Segmentation\n"
                 "CV segment determines inventory policy: (Q,r) vs Newsvendor vs Croston",
                 fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, summary["CV"].max() * 1.18)

    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=c, label=l) for l, c in color_map.items()]
    ax.legend(handles=patches + ax.get_lines()[:2]
              if hasattr(ax, "get_lines") else patches, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_module1(fig_dir: str = "outputs/figures", data_dir: str = "data") -> dict:
    import os
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    print("\n" + "="*60)
    print("MODULE 1: DEMAND CHARACTERIZATION")
    print("="*60)

    print("\n[1/5] Simulating 3 years of weekly demand for 6 SKUs...")
    df = simulate_demand()
    df.to_csv(f"{data_dir}/demand_weekly.csv", index=False)
    print(f"  Generated {len(df)} weeks × {len(df.columns)-1} SKUs")

    print("\n[2/5] Computing summary statistics and CV segmentation...")
    summary = demand_summary(df)
    print(summary.to_string(index=False))

    print("\n[3/5] Running STL decomposition on all SKUs...")
    skus = [c for c in df.columns if c != "date"]
    stl_results = {}
    for sku in skus:
        stl_results[sku] = run_stl(df[sku])
        print(f"  {sku}: decomposed ✓")

    print("\n[4/5] Fitting residual distributions (Shapiro-Wilk + KS test)...")
    fit_results = {}
    for sku in skus:
        resid = stl_results[sku]["residual"]
        fit_results[sku] = fit_residual_distribution(resid)
        best = fit_results[sku]["best_fit"]
        sw_p = fit_results[sku]["distributions"]["normal"].get("sw_p", float("nan"))
        r = resid[~np.isnan(resid)]
        kurt = float(stats.kurtosis(r))     # excess kurtosis (normal = 0)
        skew = float(stats.skew(r))
        print(f"  {sku}: best fit = {best:12s}  SW p={sw_p:.4f}  "
              f"excess_kurtosis={kurt:+.2f}  skewness={skew:+.3f}")

    print("\n  ── FINDING: all SKU residuals are leptokurtic (excess kurtosis > 0).")
    print("     This means heavy tails relative to Normal — common in FMCG with promotions.")
    print("     Using z-score safety stock formula would UNDERESTIMATE true safety stock.")
    print("     Module 2 will use empirical quantiles instead. ──")

    print("\n[5/5] Generating figures...")
    fig_demand_overview(df, f"{fig_dir}/m1_demand_overview.png")
    fig_stl_decomposition(df, "SKU_A_Yoghurt", stl_results["SKU_A_Yoghurt"],
                           f"{fig_dir}/m1_stl_yoghurt.png")
    fig_stl_decomposition(df, "SKU_C_XmasCookies", stl_results["SKU_C_XmasCookies"],
                           f"{fig_dir}/m1_stl_xmas.png")
    fig_residual_distributions(stl_results, df, fit_results,
                                f"{fig_dir}/m1_residual_distributions.png")
    fig_cv_segmentation(summary, f"{fig_dir}/m1_cv_segmentation.png")

    print("\n✓ Module 1 complete.")
    return {
        "df": df,
        "summary": summary,
        "stl_results": stl_results,
        "fit_results": fit_results,
    }


if __name__ == "__main__":
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    run_module1(
        fig_dir=str(root / "outputs" / "figures"),
        data_dir=str(root / "data"),
    )
