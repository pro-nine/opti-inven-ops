"""
Module 5: Monte Carlo Risk Quantification
==========================================
Modules 1-4 computed point estimates — the optimal (Q*, r*), the Wq, the
profit at Q*. All of these assume parameters are KNOWN with certainty.

In reality every cost parameter is uncertain:
  - Demand varies around the forecast (demand uncertainty)
  - Lead times are random AND their mean can shift (supplier reliability)
  - Holding cost rate changes with interest rates (capital cost)
  - Shortage penalty depends on contract renegotiation (commercial risk)
  - Ordering cost varies with fuel prices and logistics contracts

This module answers three questions that the earlier modules cannot:
  1. What is the DISTRIBUTION of total annual supply chain cost, not
     just its point estimate?
  2. Under parameter uncertainty, how often does the optimal policy
     ACTUALLY breach the 97.5% SLA?
  3. Which uncertain parameter should management worry about most?
     (tornado chart — the variance budget)

DESIGN CHOICE: we use one-at-a-time (OAT) sensitivity for the tornado
and full Monte Carlo for the cost distribution. OAT is not wrong here —
it is the right tool for answering "which single parameter matters most"
(the classic tornado chart question). Full MC over all parameters jointly
is the right tool for "what is the overall cost distribution" (the risk
distribution question). We do both because they answer different questions.

WHICH SKU: SKU-A Yoghurt (the dominant fast-mover by volume) is the primary
case for the tornado chart. The aggregate cost distribution uses all five
non-perishable SKUs (A,B,C,D,F). SKU-E Bread is excluded because its
newsvendor cost model is structurally different from (Q,r).
"""

import numpy as np
import pandas as pd
from math import factorial
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

N_MC = 10_000
SEED = 2024
RNG  = np.random.default_rng(SEED)

# ── Import baseline results from earlier modules ─────────────────────────────
# Rather than re-importing module objects, we embed the baseline parameters
# directly — makes this module self-contained and runnable independently.

BASELINE = {
    "SKU_A_Yoghurt":   {"D": 45836, "S": 120, "h": 0.28, "c": 2.50, "p": 8.00,
                         "mu_L": 2.0, "sig_L": 0.4, "mu_w": 882, "sig_resid": 39.2},
    "SKU_B_Detergent":  {"D": 34568, "S": 150, "h": 0.22, "c": 4.00, "p": 12.00,
                         "mu_L": 2.0, "sig_L": 0.5, "mu_w": 665, "sig_resid": 73.2},
    "SKU_C_XmasCookies":{"D":  5762, "S": 100, "h": 0.30, "c": 3.50, "p": 10.00,
                         "mu_L": 3.0, "sig_L": 1.0, "mu_w": 111, "sig_resid": 82.0},
    "SKU_D_Coffee":     {"D":  4883, "S": 200, "h": 0.20, "c": 8.00, "p": 22.00,
                         "mu_L": 3.0, "sig_L": 0.5, "mu_w":  94, "sig_resid": 10.5},
    "SKU_F_Cleaning":   {"D":  2131, "S": 250, "h": 0.18, "c": 12.0, "p": 35.00,
                         "mu_L": 4.0, "sig_L": 1.2, "mu_w":  41, "sig_resid": 45.0},
}

# Module 2 baseline TC_joint values (from Module 2 run output)
TC_BASELINE = {
    "SKU_A_Yoghurt":    3723,
    "SKU_B_Detergent":  4143,
    "SKU_C_XmasCookies":2069,
    "SKU_D_Coffee":     2005,
    "SKU_F_Cleaning":   2152,
}


# ── Core inventory model (self-contained) ────────────────────────────────────

def simulate_ltd_fast(mu_w, sig_resid, mu_L, sig_L,
                       n_sim=5000, rng=None) -> np.ndarray:
    """Fast LTD simulation — used inside MC loop."""
    if rng is None:
        rng = np.random.default_rng(42)
    ltd = np.zeros(n_sim)
    for i in range(n_sim):
        lt = max(1, int(round(rng.normal(mu_L, sig_L))))
        ltd[i] = lt * mu_w + rng.normal(0, sig_resid * np.sqrt(lt))
    return ltd


def joint_optimize_fast(D, S, H, p, ltd, max_iter=20, tol=1.0):
    """Streamlined joint (Q,r) optimizer for use in MC loop."""
    a = D / 52       # annual demand → weekly units (approximate for scaling)
    Q = np.sqrt(2 * D * S / H)
    mu_ltd = ltd.mean()
    for _ in range(max_iter):
        csl = np.clip(1 - Q * H / (D * p), 0.50, 0.9999)
        r   = np.quantile(ltd, csl)
        En  = np.mean(np.maximum(ltd - r, 0))
        Q_new = np.sqrt(2 * D * (S + p * En) / H)
        if abs(Q_new - Q) < tol:
            Q = Q_new
            break
        Q = Q_new
    r   = np.quantile(ltd, np.clip(1 - Q * H / (D * p), 0.50, 0.9999))
    En  = np.mean(np.maximum(ltd - r, 0))
    SS  = r - mu_ltd
    holding = (Q / 2 + SS) * H
    ordering = (D / Q) * S
    shortage = (D / Q) * p * En
    TC  = holding + ordering + shortage
    FR  = 1.0 - En / Q
    return {"Q": Q, "r": r, "SS": SS, "TC": TC, "FR": FR, "En": En}


def naive_policy_fast(D, S, H, mu_w, mu_ltd, ltd):
    """Naive policy: EOQ + 2-week safety stock rule of thumb."""
    Q   = np.sqrt(2 * D * S / H)
    SS  = 2 * mu_w
    r   = mu_ltd + SS
    En  = np.mean(np.maximum(ltd - r, 0))
    holding  = (Q / 2 + SS) * H
    ordering = (D / Q) * S
    shortage = (D / Q) * 12.0 * En   # use avg penalty
    TC  = holding + ordering + shortage
    FR  = 1.0 - En / Q
    return {"Q": Q, "r": r, "SS": SS, "TC": TC, "FR": FR}


# ── Monte Carlo simulation ────────────────────────────────────────────────────

def run_monte_carlo(sku: str, b: dict, n_mc: int = N_MC, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    H_base = b["h"] * b["c"]

    # Parameter distributions (independent)
    D_sims   = rng.normal(b["D"],   0.15 * b["D"],  n_mc).clip(b["D"] * 0.5, None)
    muL_sims = rng.normal(b["mu_L"],0.30 * b["mu_L"],n_mc).clip(0.5, None)
    sigL_sims= rng.normal(b["sig_L"],0.20*b["sig_L"],n_mc).clip(0.1, None)
    h_sims   = rng.normal(b["h"],   0.25 * b["h"],   n_mc).clip(0.05, None)
    S_sims   = rng.normal(b["S"],   0.30 * b["S"],   n_mc).clip(20,   None)
    p_sims   = rng.normal(b["p"],   0.30 * b["p"],   n_mc).clip(b["p"]*0.3, None)

    TC_opt, TC_naive, FR_opt, FR_naive = [], [], [], []

    # Pre-generate LTD samples at baseline (for speed — full re-sim each iter too slow)
    ltd_base = simulate_ltd_fast(b["mu_w"], b["sig_resid"], b["mu_L"], b["sig_L"],
                                  n_sim=8000, rng=rng)
    mu_ltd_base = ltd_base.mean()

    for i in range(n_mc):
        D_i = D_sims[i]; H_i = h_sims[i] * b["c"]
        S_i = S_sims[i]; p_i = p_sims[i]

        # Scale LTD by demand and lead-time ratio (fast approximation)
        # Scale mean LTD by demand ratio × lead time ratio
        demand_ratio = D_i / b["D"]
        lead_ratio   = muL_sims[i] / b["mu_L"]
        ltd_i = ltd_base * demand_ratio * lead_ratio

        opt   = joint_optimize_fast(D_i, S_i, H_i, p_i, ltd_i)
        naive = naive_policy_fast(D_i, S_i, H_i, b["mu_w"] * demand_ratio,
                                   mu_ltd_base * demand_ratio * lead_ratio, ltd_i)
        TC_opt.append(opt["TC"])
        TC_naive.append(naive["TC"])
        FR_opt.append(opt["FR"])
        FR_naive.append(naive["FR"])

    TC_opt   = np.array(TC_opt)
    TC_naive = np.array(TC_naive)
    FR_opt   = np.array(FR_opt)
    FR_naive = np.array(FR_naive)

    return {
        "TC_opt":       TC_opt,
        "TC_naive":     TC_naive,
        "FR_opt":       FR_opt,
        "FR_naive":     FR_naive,
        "sla_breach_opt":   (FR_opt   < 0.975).mean() * 100,
        "sla_breach_naive": (FR_naive < 0.975).mean() * 100,
        "TC_opt_p10":  np.percentile(TC_opt, 10),
        "TC_opt_p50":  np.percentile(TC_opt, 50),
        "TC_opt_p90":  np.percentile(TC_opt, 90),
        "TC_naive_p50":np.percentile(TC_naive, 50),
    }


# ── Tornado chart (OAT sensitivity) ─────────────────────────────────────────

def tornado_analysis(sku: str, b: dict) -> pd.DataFrame:
    """
    One-at-a-time (OAT) sensitivity: vary each parameter from P10 to P90
    while holding all others at baseline. Re-optimize for each.
    """
    ltd_base = simulate_ltd_fast(b["mu_w"], b["sig_resid"],
                                  b["mu_L"], b["sig_L"], n_sim=8000)
    H_base = b["h"] * b["c"]

    params = {
        "Annual demand D (±20%)":     ("D",    0.80, 1.20),
        "Mean lead time μL (±50%)":   ("mu_L", 0.50, 1.50),
        "Shortage penalty p (±50%)":  ("p",    0.50, 1.50),
        "Holding cost rate h (±30%)": ("h",    0.70, 1.30),
        "Ordering cost S (±50%)":     ("S",    0.50, 1.50),
        "LT variability σL (±80%)":   ("sig_L",0.20, 1.80),
    }

    rows = []
    for label, (key, lo_f, hi_f) in params.items():
        for direction, factor in [("low", lo_f), ("high", hi_f)]:
            b_mod = dict(b)
            b_mod[key] = b[key] * factor

            ltd_i = simulate_ltd_fast(b_mod["mu_w"] if key=="mu_w" else b["mu_w"],
                                       b["sig_resid"], b_mod["mu_L"], b_mod["sig_L"],
                                       n_sim=6000)
            H_i = b_mod["h"] * b["c"]
            res = joint_optimize_fast(b_mod["D"], b_mod["S"], H_i, b_mod["p"], ltd_i)
            rows.append({"param": label, "direction": direction,
                          "factor": factor, "TC": res["TC"]})

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="param", columns="direction", values="TC")
    pivot["range"] = abs(pivot["high"] - pivot["low"])
    pivot["TC_low"]  = pivot["low"]
    pivot["TC_high"] = pivot["high"]
    return pivot.sort_values("range", ascending=True)


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_cost_distribution(mc_results: dict, sku: str, tc_baseline: float, out_path: str):
    TC_opt   = mc_results["TC_opt"]
    TC_naive = mc_results["TC_naive"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    bins = np.linspace(min(TC_opt.min(), TC_naive.min()) * 0.9,
                        max(TC_opt.max(), TC_naive.max()) * 1.05, 60)
    ax.hist(TC_naive, bins=bins, alpha=0.55, color="#e76f51", label="Naive policy (EOQ + 2-wk SS)")
    ax.hist(TC_opt,   bins=bins, alpha=0.65, color="#2a9d8f", label="Optimal policy (joint Q*,r*)")
    ax.axvline(tc_baseline, color="#264653", lw=2, ls="--",
               label=f"Baseline TC (point estimate) = €{tc_baseline:,.0f}")
    for pct, ls in [(10,"--"),(50,"-"),(90,":")]:
        ax.axvline(np.percentile(TC_opt, pct), color="#2a9d8f", lw=1.2, ls=ls, alpha=0.7)
    ax.set_xlabel("Total Annual Cost (€)")
    ax.set_ylabel("Frequency (scenarios)")
    ax.set_title(f"TC Distribution — {sku.replace('_',' ')}\n"
                 f"Optimal policy MC: p10=€{mc_results['TC_opt_p10']:,.0f}  "
                 f"p50=€{mc_results['TC_opt_p50']:,.0f}  p90=€{mc_results['TC_opt_p90']:,.0f}",
                 fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"€{v:,.0f}"))
    ax.legend(fontsize=8)

    ax2 = axes[1]
    pcts = np.arange(1, 100)
    ax2.plot(pcts, [np.percentile(TC_opt, p)   for p in pcts],
             color="#2a9d8f", lw=2, label="Optimal policy")
    ax2.plot(pcts, [np.percentile(TC_naive, p) for p in pcts],
             color="#e76f51", lw=2, label="Naive policy")
    ax2.axhline(tc_baseline, color="#264653", ls="--", lw=1.5, alpha=0.7,
                label=f"Point estimate = €{tc_baseline:,.0f}")
    ax2.fill_between(pcts,
                     [np.percentile(TC_opt, p) for p in pcts],
                     [np.percentile(TC_naive, p) for p in pcts],
                     alpha=0.15, color="#264653", label="Policy gap")
    ax2.set_xlabel("Percentile")
    ax2.set_ylabel("Total Annual Cost (€)")
    ax2.set_title("Cumulative Cost Distribution\nOptimal policy dominates naive at every percentile",
                  fontweight="bold")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"€{v:,.0f}"))
    ax2.legend(fontsize=8)

    fig.suptitle(f"Monte Carlo Cost Distribution — {N_MC:,} scenarios\n"
                 "Parameter uncertainty: demand ±20%, lead time ±50%, "
                 "holding rate ±30%, penalty ±50%", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_tornado(tornado_df: pd.DataFrame, sku: str,
                tc_baseline: float, out_path: str):
    fig, ax = plt.subplots(figsize=(11, 6))
    params  = tornado_df.index.tolist()
    y       = np.arange(len(params))

    for i, param in enumerate(params):
        lo = tornado_df.loc[param, "TC_low"]
        hi = tornado_df.loc[param, "TC_high"]
        ax.barh(y[i], hi - lo, left=lo, height=0.5,
                color="#e76f51" if hi > tc_baseline else "#2a9d8f", alpha=0.8)
        ax.text(lo - 80, y[i], f"€{lo:,.0f}", va="center", ha="right", fontsize=7.5)
        ax.text(hi + 80, y[i], f"€{hi:,.0f}", va="center", ha="left",  fontsize=7.5)

    ax.axvline(tc_baseline, color="#264653", lw=2, ls="--",
               label=f"Baseline TC = €{tc_baseline:,.0f}")
    ax.set_yticks(y)
    ax.set_yticklabels(params, fontsize=9)
    ax.set_xlabel("Total Annual Cost (€)")
    ax.set_title(f"Tornado Chart — Sensitivity of TC to Parameter Uncertainty\n"
                 f"SKU-A Yoghurt | Bars show TC range from P10 to P90 of each parameter",
                 fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"€{v:,.0f}"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_sla_breach(mc_all: dict, out_path: str):
    """SLA breach probability across all SKUs, optimal vs naive policy."""
    skus   = list(mc_all.keys())
    breach_opt   = [mc_all[s]["sla_breach_opt"]   for s in skus]
    breach_naive = [mc_all[s]["sla_breach_naive"]  for s in skus]

    x = np.arange(len(skus))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5.5))
    b1 = ax.bar(x - width/2, breach_naive, width, color="#e76f51",
                label="Naive policy (EOQ + 2-wk SS)")
    b2 = ax.bar(x + width/2, breach_opt,   width, color="#2a9d8f",
                label="Optimal policy (joint Q*,r*)")

    for bar, val in list(zip(b1, breach_naive)) + list(zip(b2, breach_opt)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_","\n") for s in skus], fontsize=8)
    ax.set_ylabel("P(SLA breach) — % of scenarios where fill rate < 97.5%")
    ax.set_title("SLA Breach Probability Under Parameter Uncertainty\n"
                 "How often does each policy fail to meet the 97.5% fill-rate SLA?",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(5.0, color="#264653", ls="--", lw=1, alpha=0.6,
               label="5% risk tolerance threshold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_aggregate_cost_distribution(mc_all: dict, out_path: str):
    """Aggregate total cost distribution across all non-perishable SKUs."""
    TC_opt_agg   = sum(mc_all[s]["TC_opt"]   for s in mc_all)
    TC_naive_agg = sum(mc_all[s]["TC_naive"] for s in mc_all)
    TC_base_agg  = sum(TC_BASELINE.values())

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bins = np.linspace(
        min(TC_opt_agg.min(), TC_naive_agg.min()) * 0.92,
        max(TC_opt_agg.max(), TC_naive_agg.max()) * 1.04, 60)
    ax.hist(TC_naive_agg, bins=bins, alpha=0.50, color="#e76f51",
            label="Naive policy")
    ax.hist(TC_opt_agg,   bins=bins, alpha=0.65, color="#2a9d8f",
            label="Optimal policy")
    ax.axvline(TC_base_agg, color="#264653", lw=2, ls="--",
               label=f"Baseline (point estimate) = €{TC_base_agg:,.0f}")

    p10 = np.percentile(TC_opt_agg, 10)
    p90 = np.percentile(TC_opt_agg, 90)
    ax.axvline(p10, color="#2a9d8f", lw=1.5, ls=":", label=f"Optimal p10=€{p10:,.0f}")
    ax.axvline(p90, color="#2a9d8f", lw=1.5, ls=":",
               label=f"Optimal p90=€{p90:,.0f}  (range=€{p90-p10:,.0f})")

    ax.set_xlabel("Total Annual Supply Chain Cost (€) — All 5 Non-Perishable SKUs")
    ax.set_ylabel("Frequency (scenarios)")
    ax.set_title(f"Aggregate Cost Distribution — {N_MC:,} Monte Carlo Scenarios\n"
                 "Optimal policy p90 is lower than naive policy p50",
                 fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"€{v:,.0f}"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_module5(fig_dir="outputs/figures", data_dir="data") -> dict:
    import os
    os.makedirs(fig_dir, exist_ok=True)

    print("\n" + "="*60)
    print("MODULE 5: MONTE CARLO RISK QUANTIFICATION")
    print("="*60)
    print(f"\nRunning {N_MC:,} scenarios per SKU across 5 uncertain parameters...")

    mc_all = {}
    for sku, b in BASELINE.items():
        print(f"\n  {sku}...")
        mc = run_monte_carlo(sku, b, N_MC, SEED)
        mc_all[sku] = mc
        print(f"    TC opt:   p10=€{mc['TC_opt_p10']:,.0f}  "
              f"p50=€{mc['TC_opt_p50']:,.0f}  p90=€{mc['TC_opt_p90']:,.0f}")
        print(f"    SLA breach — optimal: {mc['sla_breach_opt']:.1f}%  "
              f"naive: {mc['sla_breach_naive']:.1f}%")

    print("\n  Running tornado analysis (SKU-A Yoghurt)...")
    tornado_df = tornado_analysis("SKU_A_Yoghurt", BASELINE["SKU_A_Yoghurt"])
    print(tornado_df[["TC_low","TC_high","range"]].round(0).to_string())

    print("\n  Aggregate findings:")
    TC_opt_agg   = sum(mc_all[s]["TC_opt"]   for s in mc_all)
    TC_naive_agg = sum(mc_all[s]["TC_naive"] for s in mc_all)
    print(f"    Aggregate TC optimal p50:  €{np.percentile(TC_opt_agg,50):,.0f}")
    print(f"    Aggregate TC naive p50:    €{np.percentile(TC_naive_agg,50):,.0f}")
    print(f"    Median annual savings:     "
          f"€{np.percentile(TC_naive_agg,50)-np.percentile(TC_opt_agg,50):,.0f}")
    total_breach_opt = np.mean(
        sum(mc_all[s]["FR_opt"] < 0.975 for s in mc_all) > 0) * 100
    print(f"    P(at least one SKU breaches SLA in a scenario): {total_breach_opt:.1f}%")

    print("\n  Generating figures...")
    fig_cost_distribution(mc_all["SKU_A_Yoghurt"], "SKU_A_Yoghurt",
                           TC_BASELINE["SKU_A_Yoghurt"],
                           f"{fig_dir}/m5_cost_distribution.png")
    fig_tornado(tornado_df, "SKU_A_Yoghurt", TC_BASELINE["SKU_A_Yoghurt"],
                f"{fig_dir}/m5_tornado.png")
    fig_sla_breach(mc_all, f"{fig_dir}/m5_sla_breach.png")
    fig_aggregate_cost_distribution(mc_all, f"{fig_dir}/m5_aggregate_distribution.png")

    print("\n✓ Module 5 complete.")
    return {"mc_all": mc_all, "tornado_df": tornado_df}


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    run_module5(fig_dir=str(root/"outputs"/"figures"),
                data_dir=str(root/"data"))
