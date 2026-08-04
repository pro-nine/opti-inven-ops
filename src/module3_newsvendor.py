"""
Module 3: Newsvendor Model — Perishable Inventory (SKU-E Fresh Bread)
=======================================================================
WHY (Q, r) IS WRONG FOR PERISHABLES:
----------------------------------------
The (Q, r) model from Module 2 assumes unsold inventory carries forward
into the next cycle. It sits in the warehouse counting toward your
reorder point. This is fine for yoghurt, detergent, and coffee.

Fresh bread has a shelf life of roughly one week. Unsold units at the
end of the period are DISPOSED OF — they cannot be carried forward.
Every period starts with zero inventory. The (Q, r) policy breaks down
because there is no "reorder point" logic: you order fresh every cycle.

This is the classic NEWSVENDOR problem (also called the "single-period"
or "perishable inventory" model). Name comes from a newspaper vendor:
order too many → unsold papers are worthless tonight; order too few →
lost sales you cannot recover.

THE NEWSVENDOR FORMULA:
-------------------------
Optimal order quantity Q* = F^{-1}(CR)

where:
    CR = Critical Ratio = Cu / (Cu + Co)
    Cu = cost of ordering ONE UNIT TOO FEW  (underage cost)
       = lost margin per unit + stockout penalty
    Co = cost of ordering ONE UNIT TOO MANY (overage cost)
       = unit cost - salvage value + disposal cost
    F  = CDF of weekly demand (empirical, from Module 1)

INTUITION:
    CR > 0.5  →  Q* > median demand (order more than you expect)
                  Happens when margin is high relative to waste cost
    CR < 0.5  →  Q* < median demand (order less than you expect)
                  Happens when margin is low and waste is expensive

For Bread: CR = 1.30 / (1.30 + 1.00) = 0.565 → order slightly above median.

THREE THINGS INTERVIEWERS LOVE ASKING:
---------------------------------------
1. "Why not just order the mean demand?" → Because E[profit] is NOT
   maximised at Q=μ unless CR=0.5 exactly. It's a common mistake.

2. "How do you handle the fact that you don't know the distribution?"
   → Empirical quantile from 3 years of history. No parametric
     assumption needed. Robust to leptokurtic residuals (Module 1).

3. "What's the Value of Perfect Information here?" → Computable. It's
   E[profit with known demand] - E[profit at Q*]. It quantifies exactly
   how much you'd pay for a better demand forecast.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

# ── Bread-specific cost parameters ───────────────────────────────────────────
# Nordpack buys from a central bakery and distributes to grocery retailers.
# All costs are per unit (one bread unit ≈ one loaf or standard bread pack).

BREAD_PARAMS = {
    "unit_cost":      1.20,   # € cost to buy from bakery
    "sell_price":     2.00,   # € wholesale price to retailer
    "salvage_value":  0.20,   # € recovered by selling surplus to food bank
    "disposal_cost":  0.10,   # € cost to dispose of waste (collection fee)
    "sla_penalty":    0.50,   # € per unit short — retailer penalty clause
}

# ── Derived cost parameters ───────────────────────────────────────────────────

def compute_costs(p: dict) -> dict:
    """
    Co = overage cost  = what you lose per unit ordered too many
       = unit_cost - salvage_value + disposal_cost
       (you paid for it, got less back, plus had to pay disposal)

    Cu = underage cost = what you lose per unit ordered too few
       = (sell_price - unit_cost) + sla_penalty
       = lost margin + SLA breach penalty
    """
    Co = p["unit_cost"] - p["salvage_value"] + p["disposal_cost"]
    Cu = (p["sell_price"] - p["unit_cost"]) + p["sla_penalty"]
    CR = Cu / (Cu + Co)
    return {"Co": Co, "Cu": Cu, "CR": CR}


# ── Core newsvendor functions ─────────────────────────────────────────────────

def newsvendor_q(demand_samples: np.ndarray, CR: float) -> float:
    """
    Q* = F^{-1}(CR) from the EMPIRICAL demand distribution.
    No parametric assumption. Uses historical weekly demand directly.
    """
    return float(np.quantile(demand_samples, CR))


def expected_waste(Q: float, demand_samples: np.ndarray) -> float:
    """E[max(Q - D, 0)] — expected units wasted per period."""
    return float(np.mean(np.maximum(Q - demand_samples, 0)))


def expected_stockout(Q: float, demand_samples: np.ndarray) -> float:
    """E[max(D - Q, 0)] — expected units short per period."""
    return float(np.mean(np.maximum(demand_samples - Q, 0)))


def expected_sales(Q: float, demand_samples: np.ndarray) -> float:
    """E[min(D, Q)] — expected units actually sold per period."""
    return float(np.mean(np.minimum(demand_samples, Q)))


def expected_profit(Q: float, demand_samples: np.ndarray,
                     Cu: float, Co: float,
                     unit_cost: float, sell_price: float,
                     salvage_value: float, disposal_cost: float) -> float:
    """
    E[profit(Q)] = revenue from sales
                 - cost of all units ordered
                 + salvage value recovered from unsold units
                 - disposal cost for unsold units
                 - SLA penalty for units short

    Note: this is gross profit per period (one week), before fixed costs.
    """
    D = demand_samples
    sold    = np.minimum(D, Q)
    wasted  = np.maximum(Q - D, 0)
    short   = np.maximum(D - Q, 0)
    penalty = Cu - (sell_price - unit_cost)   # SLA penalty portion of Cu only

    profit  = (sell_price * sold
               - unit_cost * Q
               + salvage_value * wasted
               - disposal_cost * wasted
               - penalty * short)
    return float(np.mean(profit))


def value_of_perfect_information(demand_samples: np.ndarray,
                                  Q_star: float,
                                  Cu: float, Co: float,
                                  unit_cost: float, sell_price: float,
                                  salvage_value: float, disposal_cost: float) -> float:
    """
    VOPI = E[profit(D, D)] - E[profit(D, Q*)]

    E[profit(D, D)] = profit if you KNEW demand in advance and ordered exactly D.
    Every unit is sold, no waste, no stockout, no penalty.
    = (sell_price - unit_cost) × D for every realisation.

    VOPI quantifies what a perfect demand forecast is worth per week.
    """
    D = demand_samples
    # Perfect foresight: order exactly demand, sell everything
    profit_perfect = (sell_price - unit_cost) * D
    E_profit_perfect = float(np.mean(profit_perfect))

    E_profit_q_star = expected_profit(Q_star, demand_samples, Cu, Co,
                                       unit_cost, sell_price,
                                       salvage_value, disposal_cost)
    return E_profit_perfect - E_profit_q_star


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_profit_curve(demand_samples: np.ndarray, params: dict,
                      costs: dict, Q_star: float, out_path: str):
    """Expected profit as a function of Q. Shows Q* at the maximum."""
    mu = demand_samples.mean()
    Q_range = np.linspace(mu * 0.60, mu * 1.40, 300)

    profits = [expected_profit(q, demand_samples, costs["Cu"], costs["Co"],
                                params["unit_cost"], params["sell_price"],
                                params["salvage_value"], params["disposal_cost"])
               for q in Q_range]

    # Three comparison policies
    Q_mean    = mu
    Q_type1   = float(np.quantile(demand_samples, 0.975))  # 97.5% CSL — WRONG for perishable
    Q_median  = float(np.quantile(demand_samples, 0.50))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(Q_range, profits, color="#264653", lw=2.5, label="E[Profit(Q)]")

    # Q* newsvendor
    p_star   = expected_profit(Q_star, demand_samples, costs["Cu"], costs["Co"],
                                params["unit_cost"], params["sell_price"],
                                params["salvage_value"], params["disposal_cost"])
    ax.axvline(Q_star, color="#2a9d8f", lw=2, ls="-",
               label=f"Q* newsvendor = {Q_star:.0f}  (CR={costs['CR']:.3f})")
    ax.scatter([Q_star], [p_star], color="#2a9d8f", s=80, zorder=5)
    ax.annotate(f"€{p_star:,.0f}/wk", xy=(Q_star, p_star),
                xytext=(Q_star + 30, p_star - 12), fontsize=8,
                color="#2a9d8f", fontweight="bold")

    # Q = mean
    p_mean = expected_profit(Q_mean, demand_samples, costs["Cu"], costs["Co"],
                              params["unit_cost"], params["sell_price"],
                              params["salvage_value"], params["disposal_cost"])
    ax.axvline(Q_mean, color="#e9c46a", lw=1.8, ls="--",
               label=f"Q = mean demand = {Q_mean:.0f}")
    ax.scatter([Q_mean], [p_mean], color="#e9c46a", s=60, zorder=5)

    # Q = 97.5th percentile (Type 1)
    p_type1 = expected_profit(Q_type1, demand_samples, costs["Cu"], costs["Co"],
                               params["unit_cost"], params["sell_price"],
                               params["salvage_value"], params["disposal_cost"])
    ax.axvline(Q_type1, color="#e76f51", lw=1.8, ls=":",
               label=f"Q = 97.5th pct (Type 1) = {Q_type1:.0f}  ← WRONG for perishables")
    ax.scatter([Q_type1], [p_type1], color="#e76f51", s=60, zorder=5)

    ax.set_xlabel("Weekly Order Quantity Q (units)")
    ax.set_ylabel("Expected Weekly Profit (€)")
    ax.set_title("Newsvendor: Expected Profit Curve — SKU-E Fresh Bread\n"
                 "Ordering at 97.5% CSL destroys €{:.0f}/week vs. optimal Q*".format(
                     p_star - p_type1), fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"€{v:,.0f}"))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_waste_stockout_tradeoff(demand_samples: np.ndarray,
                                 Q_star: float, out_path: str):
    """
    As Q increases, waste rises and stockouts fall.
    Q* sits at the mathematically optimal balance point — NOT at zero waste.
    This is the key insight: trying to eliminate waste entirely is suboptimal.
    """
    mu = demand_samples.mean()
    Q_range = np.linspace(mu * 0.60, mu * 1.40, 300)
    waste    = [expected_waste(q, demand_samples)   for q in Q_range]
    stockout = [expected_stockout(q, demand_samples) for q in Q_range]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax2 = ax.twinx()

    l1, = ax.plot(Q_range, waste,    color="#e76f51", lw=2, label="E[Waste] (units/week)")
    l2, = ax2.plot(Q_range, stockout, color="#2a9d8f", lw=2, label="E[Stockout] (units/week)")

    ax.axvline(Q_star, color="#264653", lw=2, ls="--",
               label=f"Q* = {Q_star:.0f} units")

    w_star = expected_waste(Q_star, demand_samples)
    s_star = expected_stockout(Q_star, demand_samples)
    ax.scatter([Q_star], [w_star],  color="#264653", s=80, zorder=5)
    ax2.scatter([Q_star], [s_star], color="#264653", s=80, zorder=5)

    ax.annotate(f"Waste at Q*: {w_star:.0f} units/wk",
                xy=(Q_star, w_star), xytext=(Q_star + 20, w_star + 5),
                fontsize=8, color="#e76f51", fontweight="bold")
    ax2.annotate(f"Stockout at Q*: {s_star:.0f} units/wk",
                 xy=(Q_star, s_star), xytext=(Q_star + 20, s_star + 2),
                 fontsize=8, color="#2a9d8f", fontweight="bold")

    ax.set_xlabel("Weekly Order Quantity Q (units)")
    ax.set_ylabel("Expected Waste (units/week)", color="#e76f51")
    ax2.set_ylabel("Expected Stockout (units/week)", color="#2a9d8f")
    ax.set_title("Waste vs Stockout Tradeoff — SKU-E Fresh Bread\n"
                 "Q* does NOT eliminate waste — it finds the cost-optimal balance",
                 fontweight="bold")

    lines = [l1, l2, plt.Line2D([0],[0], color="#264653", lw=2, ls="--",
                                  label=f"Q* = {Q_star:.0f}")]
    ax.legend(handles=lines, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_critical_ratio_sensitivity(demand_samples: np.ndarray,
                                    params: dict, out_path: str):
    """
    Sweep Cu/Co ratio → show how Q* changes.
    Key insight: small changes in cost structure can shift ordering
    policy significantly because the demand distribution has a steep
    CDF in the middle.
    """
    # Fix Co, sweep Cu from 0.3 to 3.0
    Co_base = params["unit_cost"] - params["salvage_value"] + params["disposal_cost"]
    Cu_vals = np.linspace(0.30, 3.50, 200)
    CR_vals  = Cu_vals / (Cu_vals + Co_base)
    Q_vals   = [np.quantile(demand_samples, cr) for cr in CR_vals]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: Q* vs Cu
    axes[0].plot(Cu_vals, Q_vals, color="#264653", lw=2)
    cu_base = params["sell_price"] - params["unit_cost"] + params["sla_penalty"]
    q_base  = np.quantile(demand_samples, cu_base / (cu_base + Co_base))
    axes[0].scatter([cu_base], [q_base], color="#2a9d8f", s=100, zorder=5,
                    label=f"Current Cu=€{cu_base:.2f} → Q*={q_base:.0f}")
    axes[0].axhline(demand_samples.mean(), color="#e76f51", ls="--", lw=1.5,
                    label=f"Mean demand={demand_samples.mean():.0f}")
    axes[0].set_xlabel("Underage Cost Cu (€/unit)")
    axes[0].set_ylabel("Optimal Order Quantity Q* (units)")
    axes[0].set_title("Q* increases as underage cost (Cu) rises\n"
                      "(higher margin → order more)", fontweight="bold")
    axes[0].legend(fontsize=8)

    # Right: CR vs Q* — shows how the quantile function maps
    axes[1].plot(CR_vals, Q_vals, color="#264653", lw=2)
    axes[1].scatter([cu_base/(cu_base+Co_base)], [q_base],
                    color="#2a9d8f", s=100, zorder=5,
                    label=f"Current CR={cu_base/(cu_base+Co_base):.3f}")
    axes[1].axvline(0.5, color="#e76f51", ls="--", lw=1.2,
                    label="CR=0.5 (Q*=median demand)")
    axes[1].set_xlabel("Critical Ratio CR = Cu/(Cu+Co)")
    axes[1].set_ylabel("Optimal Order Quantity Q* (units)")
    axes[1].set_title("Q* = F⁻¹(CR) — the empirical quantile function\n"
                      "CR>0.5 → order above median; CR<0.5 → order below",
                      fontweight="bold")
    axes[1].legend(fontsize=8)

    fig.suptitle("Critical Ratio Sensitivity Analysis — SKU-E Fresh Bread\n"
                 "A 20% change in penalty clause shifts weekly order by ~{:.0f} units".format(
                     abs(np.quantile(demand_samples, (cu_base*1.2)/(cu_base*1.2+Co_base))
                         - q_base)),
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_policy_comparison_table(results: dict, out_path: str):
    """Summary table comparing four ordering policies."""
    cols = ["Policy", "Q\n(units/wk)", "E[Waste]\n(units/wk)",
            "E[Stockout]\n(units/wk)", "E[Profit]\n(€/wk)",
            "Annual Profit\n(€/yr)", "Gap vs Q*\n(€/yr)"]
    rows = []
    q_star_annual = results["profit_q_star"] * 52
    for name, res in results["policies"].items():
        annual = res["profit"] * 52
        rows.append([
            name,
            f"{res['Q']:.0f}",
            f"{res['waste']:.1f}",
            f"{res['stockout']:.1f}",
            f"€{res['profit']:,.0f}",
            f"€{annual:,.0f}",
            f"€{annual - q_star_annual:+,.0f}",
        ])

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.5)

    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#264653")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight Q* row (row 1 = index 1 in table)
    for j in range(len(cols)):
        tbl[1, j].set_facecolor("#d4edda")

    # Red for Type 1 row (last row)
    for j in range(len(cols)):
        tbl[len(rows), j].set_facecolor("#fde8e8")

    ax.set_title("Newsvendor Policy Comparison — SKU-E Fresh Bread\n"
                 "Q* (green) maximises profit | Type 1 CSL (red) is worst — "
                 "too much waste for too little extra SLA benefit",
                 fontsize=10, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_demand_and_q_star(demand_samples: np.ndarray,
                           Q_star: float, params: dict, costs: dict,
                           out_path: str):
    """Histogram of weekly demand with key quantile lines overlaid."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.hist(demand_samples, bins=45, density=False,
            color="#264653", alpha=0.55, label="Weekly demand (historical)")

    quantiles = {
        "Median (Q at CR=0.5)":         (0.50, "#e9c46a", "--"),
        f"Q* newsvendor (CR={costs['CR']:.3f})": (costs["CR"], "#2a9d8f", "-"),
        "97.5th pct (Type 1 CSL)":      (0.975, "#e76f51", ":"),
    }
    for label, (q_pct, color, ls) in quantiles.items():
        val = np.quantile(demand_samples, q_pct)
        ax.axvline(val, color=color, lw=2, ls=ls,
                   label=f"{label} = {val:.0f} units")

    ax.set_xlabel("Weekly Demand (units)")
    ax.set_ylabel("Frequency (weeks)")
    ax.set_title(
        f"Weekly Demand Distribution — SKU-E Fresh Bread\n"
        f"Q* = {Q_star:.0f} units/week  |  Co=€{costs['Co']:.2f}  "
        f"Cu=€{costs['Cu']:.2f}  CR={costs['CR']:.3f}",
        fontweight="bold"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_module3(demand_df: pd.DataFrame = None,
                fig_dir: str = "outputs/figures",
                data_dir: str = "data") -> dict:
    import os
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    if demand_df is None:
        from src.module1_demand import simulate_demand
        demand_df = simulate_demand()

    print("\n" + "="*60)
    print("MODULE 3: NEWSVENDOR MODEL — PERISHABLE SKU (BREAD)")
    print("="*60)

    demand_samples = demand_df["SKU_E_Bread"].values.astype(float)
    mu  = demand_samples.mean()
    std = demand_samples.std()
    print(f"\n  Weekly demand: mean={mu:.0f}  std={std:.0f}  "
          f"CV={std/mu:.3f}  n={len(demand_samples)} weeks")

    print("\n[1/4] Computing cost parameters...")
    costs = compute_costs(BREAD_PARAMS)
    print(f"  Overage cost  Co = unit_cost - salvage + disposal = "
          f"€{BREAD_PARAMS['unit_cost']:.2f} - €{BREAD_PARAMS['salvage_value']:.2f} "
          f"+ €{BREAD_PARAMS['disposal_cost']:.2f} = €{costs['Co']:.2f}/unit")
    print(f"  Underage cost Cu = lost_margin + sla_penalty = "
          f"€{BREAD_PARAMS['sell_price']-BREAD_PARAMS['unit_cost']:.2f} "
          f"+ €{BREAD_PARAMS['sla_penalty']:.2f} = €{costs['Cu']:.2f}/unit")
    print(f"  Critical Ratio CR = {costs['Cu']:.2f} / ({costs['Cu']:.2f} + {costs['Co']:.2f}) "
          f"= {costs['CR']:.4f}")
    print(f"  CR > 0.5 → Q* will be ABOVE the median demand (margin > waste cost)")

    print("\n[2/4] Computing optimal order quantity...")
    Q_star = newsvendor_q(demand_samples, costs["CR"])
    print(f"  Q* = F^{{-1}}({costs['CR']:.4f}) = {Q_star:.0f} units/week")
    print(f"  Median demand = {np.quantile(demand_samples, 0.5):.0f} units/week")
    print(f"  Mean demand   = {mu:.0f} units/week")
    print(f"  Q* > mean: {Q_star > mu}  (expected for CR={costs['CR']:.3f} > 0.5)")

    print("\n[3/4] Computing performance metrics at Q*...")
    waste    = expected_waste(Q_star, demand_samples)
    stockout = expected_stockout(Q_star, demand_samples)
    sales    = expected_sales(Q_star, demand_samples)
    profit   = expected_profit(Q_star, demand_samples,
                                costs["Cu"], costs["Co"],
                                **{k: BREAD_PARAMS[k] for k in
                                   ["unit_cost","sell_price","salvage_value","disposal_cost"]})
    vopi     = value_of_perfect_information(
                   demand_samples, Q_star, costs["Cu"], costs["Co"],
                   **{k: BREAD_PARAMS[k] for k in
                      ["unit_cost","sell_price","salvage_value","disposal_cost"]})

    print(f"  E[Sales]   = {sales:.1f} units/week  ({sales/Q_star*100:.1f}% of ordered)")
    print(f"  E[Waste]   = {waste:.1f} units/week  ({waste/Q_star*100:.1f}% of ordered)")
    print(f"  E[Stockout]= {stockout:.1f} units/week")
    print(f"  E[Profit]  = €{profit:,.0f}/week  →  €{profit*52:,.0f}/year")
    print(f"  VOPI       = €{vopi:,.2f}/week (value of a perfect demand forecast)")

    # Compare four policies
    policies = {}
    for name, Q in [
        ("Q* Newsvendor (optimal)", Q_star),
        (f"Q = Mean demand ({mu:.0f})", mu),
        (f"Q = Median demand ({np.quantile(demand_samples, 0.5):.0f})", np.quantile(demand_samples, 0.5)),
        (f"Q = 97.5th pct / Type1 CSL ({np.quantile(demand_samples, 0.975):.0f})", np.quantile(demand_samples, 0.975)),
    ]:
        p = expected_profit(Q, demand_samples, costs["Cu"], costs["Co"],
                             **{k: BREAD_PARAMS[k] for k in
                                ["unit_cost","sell_price","salvage_value","disposal_cost"]})
        policies[name] = {
            "Q": Q,
            "waste":    expected_waste(Q, demand_samples),
            "stockout": expected_stockout(Q, demand_samples),
            "profit":   p,
        }

    print("\n[4/4] Policy comparison:")
    print(f"  {'Policy':<45} {'Q':>7} {'Waste':>8} {'Stockout':>10} {'Profit/wk':>12} {'Annual':>12}")
    print("  " + "-"*98)
    for name, res in policies.items():
        print(f"  {name:<45} {res['Q']:>7.0f} {res['waste']:>8.1f} "
              f"{res['stockout']:>10.1f} €{res['profit']:>10,.0f} "
              f"€{res['profit']*52:>10,.0f}")

    q_star_annual = profit * 52
    worst_policy  = min(policies.values(), key=lambda x: x["profit"])
    print(f"\n  Gap between optimal Q* and worst policy: "
          f"€{(profit - worst_policy['profit'])*52:,.0f}/year")

    print("\n  Generating figures...")
    full_results = {
        "Q_star": Q_star, "costs": costs, "params": BREAD_PARAMS,
        "waste": waste, "stockout": stockout, "sales": sales,
        "profit_q_star": profit, "vopi": vopi, "policies": policies,
        "demand_samples": demand_samples,
    }
    fig_demand_and_q_star(demand_samples, Q_star, BREAD_PARAMS, costs,
                           f"{fig_dir}/m3_demand_and_qstar.png")
    fig_profit_curve(demand_samples, BREAD_PARAMS, costs, Q_star,
                     f"{fig_dir}/m3_profit_curve.png")
    fig_waste_stockout_tradeoff(demand_samples, Q_star,
                                 f"{fig_dir}/m3_waste_stockout.png")
    fig_critical_ratio_sensitivity(demand_samples, BREAD_PARAMS,
                                    f"{fig_dir}/m3_cr_sensitivity.png")
    fig_policy_comparison_table(full_results,
                                 f"{fig_dir}/m3_policy_comparison.png")

    print(f"\n✓ Module 3 complete.")
    return full_results


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    run_module3(
        fig_dir=str(root / "outputs" / "figures"),
        data_dir=str(root / "data"),
    )
