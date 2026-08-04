"""
Module 2: (Q, r) Inventory Policy Optimization
=================================================
This module derives the optimal inventory policy for each SKU at Nordpack GmbH.
The policy is defined by two numbers: (Q*, r*)
    Q* = how many units to order each time you order
    r* = when to place the order (at what inventory level)

The math below is from Silver, Pyke & Thomas (2017) "Inventory and Production
Management in Supply Chains" — the standard graduate-level reference for this.

THREE THINGS MOST PEOPLE GET WRONG (and what you'll get right):
------------------------------------------------------------------
1. EOQ is NOT the correct Q when stockouts have penalties.
   The standard EOQ = √(2DS/H) minimises holding + ordering cost only.
   It ignores shortage cost completely. When there are contractual SLA
   penalties (which Nordpack has), the correct Q is larger than EOQ.

2. Type 1 ≠ Type 2 service level. They sound the same. They are not.
   Type 1 (CSL): P(no stockout per ORDER CYCLE) = α
   Type 2 (FR):  fraction of UNITS demanded filled from stock = β
   At the same target percentage, Type 2 requires LESS safety stock.
   Companies that confuse them systematically overstock.

3. Solving Q* and r* independently is wrong.
   Q* depends on E[shortage per cycle] which depends on r*.
   r* (via its optimal CSL) depends on Q*.
   They must be solved JOINTLY via iteration.
   The independent solution can overestimate total annual cost by 8–15%.

NOTATION:
----------
D   = annual demand (units/year)
S   = ordering cost (€ per order placed)
H   = holding cost (€/unit/year) = holding rate × unit cost
p   = shortage penalty (€ per unit short)
L   = lead time (weeks) — random variable ~ Normal(μ_L, σ_L)
LTD = Lead Time Demand — total demand during lead time (random variable)
μ_LTD, σ_LTD = mean and std of LTD
r   = reorder point (units)
SS  = safety stock = r - μ_LTD (the buffer above expected LTD)
Q   = order quantity (units)
TC  = total annual cost = ordering + holding + shortage cost
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

# ── SKU cost and lead-time parameters ────────────────────────────────────────
# These are the operational inputs — in a real project you'd pull these
# from the ERP system (SAP MM, Oracle SCM). Here they're calibrated to
# plausible FMCG values.

SKU_PARAMS = {
    "SKU_A_Yoghurt": {
        "unit_cost":        2.50,    # € per unit
        "order_cost":       120.0,   # € fixed cost per purchase order
        "holding_rate":     0.28,    # 28% of unit value per year (refrigerated!)
        "shortage_penalty": 8.00,    # € per unit short (SLA breach penalty)
        "lead_time_mean":   2.0,     # weeks
        "lead_time_std":    0.4,
    },
    "SKU_B_Detergent": {
        "unit_cost":        4.00,
        "order_cost":       150.0,
        "holding_rate":     0.22,
        "shortage_penalty": 12.00,
        "lead_time_mean":   2.0,
        "lead_time_std":    0.5,
    },
    "SKU_C_XmasCookies": {
        "unit_cost":        3.50,
        "order_cost":       100.0,
        "holding_rate":     0.30,
        "shortage_penalty": 10.00,
        "lead_time_mean":   3.0,
        "lead_time_std":    1.0,    # higher variability: seasonal supplier
    },
    "SKU_D_Coffee": {
        "unit_cost":        8.00,
        "order_cost":       200.0,
        "holding_rate":     0.20,
        "shortage_penalty": 22.00,
        "lead_time_mean":   3.0,
        "lead_time_std":    0.5,
    },
    "SKU_E_Bread": {
        "unit_cost":        1.20,
        "order_cost":       80.0,
        "holding_rate":     0.60,   # very high: perishable, short shelf life
        "shortage_penalty": 4.00,
        "lead_time_mean":   1.0,
        "lead_time_std":    0.2,
    },
    "SKU_F_Cleaning": {
        "unit_cost":        12.00,
        "order_cost":       250.0,
        "holding_rate":     0.18,
        "shortage_penalty": 35.00,
        "lead_time_mean":   4.0,
        "lead_time_std":    1.2,
    },
}

SERVICE_LEVEL_TARGET = 0.975   # 97.5% — Nordpack's contractual SLA with retailers
N_SIM = 20_000                 # Monte Carlo simulations for LTD distribution


# ── Step 1: Simulate Lead Time Demand (LTD) distribution ─────────────────────

def simulate_ltd(weekly_demand: np.ndarray, lead_time_mean: float,
                  lead_time_std: float, n_sim: int = N_SIM,
                  seed: int = 99) -> np.ndarray:
    """
    Monte Carlo simulation of Lead Time Demand.

    Why Monte Carlo instead of the analytical formula σ_LTD = √(L×σ²)?
    The analytical formula assumes demand is i.i.d. Normal and lead time
    is independent of demand. Both are violated here:
      - Demand residuals are leptokurtic (found in Module 1)
      - Lead time is itself random (variable σ_L)

    The compound variance formula for variable lead time is:
        σ²_LTD = μ_L × σ²_demand + μ²_demand × σ²_L
    This is only exact under independence + stationarity. Monte Carlo
    makes no such assumption and is valid for any residual distribution.

    Process:
    1. Draw a lead time L from TruncatedNormal(μ_L, σ_L, min=1 week)
    2. Round L to nearest integer (demand is weekly)
    3. Sample L weeks of demand from the empirical historical demand
    4. Sum to get one LTD realisation
    5. Repeat n_sim times
    """
    rng = np.random.default_rng(seed)
    ltd_samples = np.zeros(n_sim)
    for i in range(n_sim):
        # Truncated normal lead time — cannot be less than 1 week
        lt = max(1, int(round(rng.normal(lead_time_mean, lead_time_std))))
        # Sample `lt` weeks of demand from the empirical demand history
        sampled_weeks = rng.choice(weekly_demand, size=lt, replace=True)
        ltd_samples[i] = sampled_weeks.sum()
    return ltd_samples


# ── Step 2: EOQ ───────────────────────────────────────────────────────────────

def eoq(D: float, S: float, H: float) -> float:
    """
    Economic Order Quantity — minimises holding + ordering cost only.
    Q* = √(2DS/H)

    This is the BASELINE. The joint optimization in Step 6 will give
    a DIFFERENT (larger) Q when shortage costs are non-zero.
    """
    return np.sqrt(2 * D * S / H)


# ── Step 3: Safety stock — Type 1 (Cycle Service Level) ──────────────────────

def safety_stock_type1(ltd_samples: np.ndarray, service_level: float) -> dict:
    """
    Type 1: P(no stockout during lead time) = service_level

    Interpretation: in `service_level` fraction of replenishment cycles,
    we never hit zero inventory before the new order arrives.

    Using empirical quantile (not z × σ) because Module 1 showed leptokurtic
    residuals — the Normal quantile would underestimate required SS.

    r_type1 = F^{-1}(service_level) from empirical LTD distribution
    SS_type1 = r_type1 - μ_LTD
    """
    mu_ltd = ltd_samples.mean()
    r = np.quantile(ltd_samples, service_level)
    ss = r - mu_ltd
    return {
        "reorder_point": r,
        "safety_stock": ss,
        "mu_ltd": mu_ltd,
        "sigma_ltd": ltd_samples.std(),
        "service_level_achieved": service_level,
    }


# ── Step 4: Safety stock — Type 2 (Fill Rate) ────────────────────────────────

def expected_shortage(r: float, ltd_samples: np.ndarray) -> float:
    """
    E[n(r)] = expected units short per replenishment cycle.
    = E[max(LTD - r, 0)]
    Computed empirically from Monte Carlo LTD samples.
    """
    return float(np.mean(np.maximum(ltd_samples - r, 0)))


def safety_stock_type2(ltd_samples: np.ndarray, Q: float,
                        service_level: float) -> dict:
    """
    Type 2 Fill Rate: fraction of demand units filled from stock = service_level.

    FR = 1 - E[n(r)] / Q

    Solving for r: find r such that E[n(r)] = Q × (1 - FR)
    E[n(r)] is monotonically decreasing in r → use bisection search.

    KEY INSIGHT: Type 2 at 97.5% requires LESS safety stock than Type 1 at 97.5%
    because a stockout event in one cycle might only affect a few units (out of Q),
    not the entire cycle. Type 1 counts the cycle as a "failure" regardless.
    """
    mu_ltd = ltd_samples.mean()
    target_shortage = Q * (1 - service_level)

    # Bisection search for r
    r_lo = mu_ltd * 0.5
    r_hi = np.quantile(ltd_samples, 0.9999)
    for _ in range(60):
        r_mid = (r_lo + r_hi) / 2
        if expected_shortage(r_mid, ltd_samples) > target_shortage:
            r_lo = r_mid
        else:
            r_hi = r_mid
        if (r_hi - r_lo) < 0.01:
            break

    r = (r_lo + r_hi) / 2
    ss = r - mu_ltd
    en = expected_shortage(r, ltd_samples)
    fr_achieved = 1 - en / Q

    return {
        "reorder_point": r,
        "safety_stock": ss,
        "mu_ltd": mu_ltd,
        "sigma_ltd": ltd_samples.std(),
        "expected_shortage_per_cycle": en,
        "fill_rate_achieved": fr_achieved,
    }


# ── Step 5: Total Annual Cost ─────────────────────────────────────────────────

def total_annual_cost(Q: float, r: float, D: float, S: float, H: float,
                       p: float, mu_ltd: float, ltd_samples: np.ndarray) -> float:
    """
    TC(Q, r) = ordering cost + holding cost + shortage cost

    Ordering cost:   (D/Q) × S           [orders/year × €/order]
    Holding cost:    (Q/2 + SS) × H      [cycle stock + safety stock, per year]
    Shortage cost:   (D/Q) × p × E[n(r)] [cycles/year × €/unit × units short/cycle]

    Note: (Q/2 + SS) × H rather than just Q/2 × H because safety stock
    is held ALL year round, not just during the cycle.
    """
    SS = r - mu_ltd
    En = expected_shortage(r, ltd_samples)
    order_cost    = (D / Q) * S
    holding_cost  = (Q / 2 + SS) * H
    shortage_cost = (D / Q) * p * En
    return order_cost + holding_cost + shortage_cost


# ── Step 6: Joint (Q*, r*) iterative optimization ────────────────────────────

def joint_optimize(D: float, S: float, H: float, p: float,
                    ltd_samples: np.ndarray, max_iter: int = 30,
                    tol: float = 0.5) -> dict:
    """
    Joint iterative optimization of (Q*, r*).

    WHY ITERATIVE? Q* depends on E[n(r*)] which depends on r*.
    r* (optimal CSL) = 1 - Q*H/(Dp) which depends on Q*.
    They are coupled — you cannot solve one without the other.

    Algorithm (Silver, Pyke & Thomas §5.4):
    1. Start with Q_0 = EOQ (ignores shortage cost)
    2. Compute optimal CSL: α_k = 1 - Q_{k-1} × H / (D × p)
       (derived from ∂TC/∂r = 0 condition)
    3. Set r_k = F^{-1}(α_k) [empirical quantile]
    4. Compute E[n(r_k)] from Monte Carlo
    5. Update Q_k = √(2D(S + p×E[n(r_k)])/H)
       (derived from ∂TC/∂Q = 0 condition)
    6. Repeat until |Q_k - Q_{k-1}| < tol

    The independent (wrong) solution uses Q=EOQ and ignores steps 2-6.
    """
    mu_ltd = ltd_samples.mean()
    Q = eoq(D, S, H)   # step 1: start with EOQ

    history = []
    for k in range(max_iter):
        # Step 2: optimal cycle service level given Q
        csl = 1 - (Q * H) / (D * p)
        csl = np.clip(csl, 0.50, 0.9999)   # keep between 50% and 99.99%

        # Step 3: reorder point from empirical quantile
        r = np.quantile(ltd_samples, csl)

        # Step 4: expected shortage per cycle
        En = expected_shortage(r, ltd_samples)

        # Step 5: updated Q
        Q_new = np.sqrt(2 * D * (S + p * En) / H)

        history.append({"iter": k, "Q": Q, "r": r, "csl": csl, "En": En})

        if abs(Q_new - Q) < tol:
            Q = Q_new
            break
        Q = Q_new

    SS = r - mu_ltd
    TC = total_annual_cost(Q, r, D, S, H, p, mu_ltd, ltd_samples)

    return {
        "Q_star": Q, "r_star": r, "SS_star": SS, "csl_achieved": csl,
        "En_star": En, "TC_joint": TC, "iterations": len(history),
        "convergence_history": history,
    }


def independent_solution(D: float, S: float, H: float, p: float,
                           ltd_samples: np.ndarray,
                           service_level: float = SERVICE_LEVEL_TARGET) -> dict:
    """
    The WRONG (independent) approach:
    Step 1: Q_ind = EOQ (ignores shortage cost)
    Step 2: r_ind = F^{-1}(service_level) (ignores that optimal CSL depends on Q)
    These are NOT jointly optimal — TC is higher than joint solution.
    """
    mu_ltd = ltd_samples.mean()
    Q_ind = eoq(D, S, H)
    r_ind = np.quantile(ltd_samples, service_level)
    SS_ind = r_ind - mu_ltd
    En_ind = expected_shortage(r_ind, ltd_samples)
    TC_ind = total_annual_cost(Q_ind, r_ind, D, S, H, p, mu_ltd, ltd_samples)
    return {
        "Q_ind": Q_ind, "r_ind": r_ind, "SS_ind": SS_ind,
        "En_ind": En_ind, "TC_ind": TC_ind,
    }


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_type1_vs_type2(results: dict, out_path: str):
    """
    THE KEY INSIGHT CHART:
    Side-by-side safety stock required under Type 1 vs Type 2
    at the same 97.5% service level target.
    """
    skus = list(results.keys())
    ss_t1 = [results[s]["ss_type1"]["safety_stock"] for s in skus]
    ss_t2 = [results[s]["ss_type2"]["safety_stock"] for s in skus]
    overstock = [t1 - t2 for t1, t2 in zip(ss_t1, ss_t2)]
    holding_rates = [SKU_PARAMS[s]["unit_cost"] * SKU_PARAMS[s]["holding_rate"] for s in skus]
    wasted_holding = [ov * h for ov, h in zip(overstock, holding_rates)]

    x = np.arange(len(skus))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    bars1 = ax.bar(x - width/2, ss_t1, width, label="Type 1 (CSL)", color="#264653")
    bars2 = ax.bar(x + width/2, ss_t2, width, label="Type 2 (Fill Rate)", color="#2a9d8f")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in skus], fontsize=8)
    ax.set_ylabel("Safety Stock (units)")
    ax.set_title("Safety Stock Required at 97.5% Service Level\n"
                 "Type 1 (CSL) vs Type 2 (Fill Rate) — same target, DIFFERENT answer",
                 fontweight="bold")
    ax.legend()
    for b1, b2, ov in zip(bars1, bars2, overstock):
        ax.annotate(f"Δ={ov:.0f}", xy=((b1.get_x()+b2.get_x()+b2.get_width())/2,
                    max(b1.get_height(), b2.get_height()) + 2),
                    ha="center", fontsize=7, color="#c1121f", fontweight="bold")

    ax2 = axes[1]
    colors_waste = ["#e76f51" if w > 0 else "#2a9d8f" for w in wasted_holding]
    ax2.bar(x, wasted_holding, color=colors_waste, width=0.5)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.replace("_", "\n") for s in skus], fontsize=8)
    ax2.set_ylabel("Excess Annual Holding Cost (€/year)")
    ax2.set_title("Annual Holding Cost Wasted by Using Type 1\nwhen the SLA contract specifies Type 2",
                  fontweight="bold")
    total_waste = sum(w for w in wasted_holding if w > 0)
    ax2.annotate(f"Total unnecessary holding cost: €{total_waste:,.0f}/year",
                 xy=(0.5, 0.92), xycoords="axes fraction", ha="center",
                 fontsize=9, color="#c1121f", fontweight="bold",
                 bbox=dict(boxstyle="round", fc="#fff3cd", alpha=0.9))

    fig.suptitle("The Type 1 vs Type 2 Service Level Problem\n"
                 "Companies that confuse them carry excess safety stock they don't need",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_joint_vs_independent(results: dict, out_path: str):
    """
    Shows the total annual cost gap between joint and independent optimization.
    """
    skus = list(results.keys())
    tc_joint = [results[s]["joint"]["TC_joint"] for s in skus]
    tc_ind   = [results[s]["independent"]["TC_ind"] for s in skus]
    savings  = [ti - tj for ti, tj in zip(tc_ind, tc_joint)]
    pct_gap  = [(ti - tj) / ti * 100 for ti, tj in zip(tc_ind, tc_joint)]

    x = np.arange(len(skus))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.bar(x - width/2, tc_ind,   width, label="Independent (EOQ + fixed CSL)", color="#e76f51", alpha=0.85)
    ax.bar(x + width/2, tc_joint, width, label="Joint iterative optimum",        color="#2a9d8f", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in skus], fontsize=8)
    ax.set_ylabel("Total Annual Cost (€)")
    ax.set_title("Total Annual Cost: Independent vs Joint Optimization",
                 fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"€{v:,.0f}"))
    ax.legend()

    ax2 = axes[1]
    bars = ax2.bar(x, pct_gap, color="#264653", width=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.replace("_", "\n") for s in skus], fontsize=8)
    ax2.set_ylabel("Cost Reduction from Joint Optimization (%)")
    ax2.set_title("% Savings from Solving (Q*, r*) Jointly\nvs. Independent EOQ + Safety Stock",
                  fontweight="bold")
    for bar, pct, sav in zip(bars, pct_gap, savings):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.1,
                 f"{pct:.1f}%\n(€{sav:,.0f}/yr)",
                 ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    total_savings = sum(savings)
    ax2.annotate(f"Total savings across all SKUs: €{total_savings:,.0f}/year",
                 xy=(0.5, 0.88), xycoords="axes fraction", ha="center",
                 fontsize=9, color="#2a9d8f", fontweight="bold",
                 bbox=dict(boxstyle="round", fc="#d4edda", alpha=0.9))

    fig.suptitle("The Joint Optimization Gap\n"
                 "Solving Q* and r* independently ignores their coupling through shortage cost",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_total_cost_curves(results: dict, out_path: str):
    """
    TC(Q) curve for two representative SKUs showing:
    - EOQ minimum
    - Joint Q* (shifted right due to shortage cost)
    - The flat region around the optimum (insensitivity zone)
    """
    showcase = ["SKU_A_Yoghurt", "SKU_D_Coffee"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, sku in zip(axes, showcase):
        r = results[sku]
        D = r["D"]; S = r["S"]; H = r["H"]; p = r["p"]
        mu_ltd = r["ltd_samples"].mean()
        ltd = r["ltd_samples"]
        r_star = r["joint"]["r_star"]

        Q_range = np.linspace(0.2 * r["joint"]["Q_star"],
                               2.5 * r["joint"]["Q_star"], 200)
        TC_vals = [total_annual_cost(q, r_star, D, S, H, p, mu_ltd, ltd)
                   for q in Q_range]

        ax.plot(Q_range, TC_vals, color="#264653", lw=2)
        ax.axvline(eoq(D, S, H), color="#e76f51", ls="--", lw=1.5,
                   label=f"EOQ = {eoq(D,S,H):.0f} units")
        ax.axvline(r["joint"]["Q_star"], color="#2a9d8f", ls="-", lw=2,
                   label=f"Q* joint = {r['joint']['Q_star']:.0f} units")
        ax.set_xlabel("Order Quantity Q (units)")
        ax.set_ylabel("Total Annual Cost (€)")
        ax.set_title(f"{sku.replace('_',' ')}\nTC(Q) — note Q* > EOQ when shortage cost ≠ 0",
                     fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"€{v:,.0f}"))
        ax.legend(fontsize=8)

        # Shade ±10% region around Q*
        Q_lo = r["joint"]["Q_star"] * 0.90
        Q_hi = r["joint"]["Q_star"] * 1.10
        ax.axvspan(Q_lo, Q_hi, alpha=0.12, color="#2a9d8f",
                   label="±10% insensitivity zone")

    fig.suptitle("Total Cost Curve TC(Q) — why EOQ is NOT the right answer under SLA penalties",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_ltd_distributions(results: dict, out_path: str):
    skus = list(results.keys())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, sku in enumerate(skus):
        ax = axes[i]
        ltd = results[sku]["ltd_samples"]
        r_t1 = results[sku]["ss_type1"]["reorder_point"]
        r_t2 = results[sku]["ss_type2"]["reorder_point"]
        mu   = ltd.mean()

        ax.hist(ltd, bins=60, density=True, color="#264653", alpha=0.55,
                label="LTD distribution (MC)")
        ax.axvline(mu,  color="black",   lw=1.2, ls=":",  label=f"μ_LTD={mu:.0f}")
        ax.axvline(r_t1, color="#e76f51", lw=1.8, ls="--",
                   label=f"r (Type 1)={r_t1:.0f}")
        ax.axvline(r_t2, color="#2a9d8f", lw=1.8, ls="-",
                   label=f"r (Type 2)={r_t2:.0f}")
        ax.set_title(sku.replace("_", " "), fontsize=9, fontweight="bold")
        ax.set_xlabel("Lead Time Demand (units)")
        ax.legend(fontsize=6.5)

    fig.suptitle("Lead Time Demand Distributions (Monte Carlo, n=20,000)\n"
                 "Vertical lines show reorder point r under Type 1 vs Type 2 — "
                 "Type 2 reorder point is always lower (less safety stock needed)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_policy_summary(results: dict, out_path: str):
    """One-page summary table chart of the recommended policy for all SKUs."""
    rows = []
    for sku, r in results.items():
        params = SKU_PARAMS[sku]
        H = params["unit_cost"] * params["holding_rate"]
        rows.append({
            "SKU": sku.replace("SKU_", "").replace("_", " "),
            "Annual\nDemand": f"{r['D']:,.0f}",
            "EOQ\n(units)": f"{eoq(r['D'], r['S'], H):.0f}",
            "Q* Joint\n(units)": f"{r['joint']['Q_star']:.0f}",
            "SS Type1\n(units)": f"{r['ss_type1']['safety_stock']:.0f}",
            "SS Type2\n(units)": f"{r['ss_type2']['safety_stock']:.0f}",
            "r* (reorder\npoint, units)": f"{r['joint']['r_star']:.0f}",
            "TC Joint\n(€/yr)": f"€{r['joint']['TC_joint']:,.0f}",
            "TC Indep\n(€/yr)": f"€{r['independent']['TC_ind']:,.0f}",
            "Savings\n(€/yr)": f"€{r['independent']['TC_ind'] - r['joint']['TC_joint']:,.0f}",
        })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 2.2)

    # Header styling
    for j in range(len(df.columns)):
        tbl[0, j].set_facecolor("#264653")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # Alternating row colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            tbl[i, j].set_facecolor("#f0f4f8" if i % 2 == 0 else "white")
        # Highlight savings column
        tbl[i, len(df.columns) - 1].set_facecolor("#d4edda")

    ax.set_title("Nordpack GmbH — Recommended (Q*, r*) Inventory Policy\n"
                 "All SKUs at 97.5% SLA target | Joint optimization vs. independent baseline",
                 fontsize=11, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_module2(demand_df: pd.DataFrame = None,
                fig_dir: str = "outputs/figures",
                data_dir: str = "data") -> dict:
    import os
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    if demand_df is None:
        from src.module1_demand import simulate_demand
        demand_df = simulate_demand()

    print("\n" + "="*60)
    print("MODULE 2: (Q, r) INVENTORY POLICY OPTIMIZATION")
    print("="*60)

    skus = [c for c in demand_df.columns if c != "date"]
    results = {}

    for sku in skus:
        print(f"\n── {sku} ──")
        weekly = demand_df[sku].values
        params = SKU_PARAMS[sku]
        D = weekly.mean() * 52              # annual demand
        H = params["unit_cost"] * params["holding_rate"]
        S = params["order_cost"]
        p = params["shortage_penalty"]

        print(f"  Annual demand D={D:,.0f}  H=€{H:.2f}/unit/yr  S=€{S:.0f}/order  p=€{p:.0f}/unit")

        print(f"  Simulating LTD distribution (n={N_SIM:,})...")
        ltd = simulate_ltd(weekly, params["lead_time_mean"],
                            params["lead_time_std"])

        Q_eoq = eoq(D, S, H)
        print(f"  EOQ (baseline) = {Q_eoq:.0f} units")

        t1 = safety_stock_type1(ltd, SERVICE_LEVEL_TARGET)
        t2 = safety_stock_type2(ltd, Q_eoq, SERVICE_LEVEL_TARGET)
        print(f"  Type 1 SS (97.5% CSL) = {t1['safety_stock']:.0f} units  →  r={t1['reorder_point']:.0f}")
        print(f"  Type 2 SS (97.5% FR)  = {t2['safety_stock']:.0f} units  →  r={t2['reorder_point']:.0f}")
        print(f"  Excess SS from Type1 vs Type2: {t1['safety_stock']-t2['safety_stock']:.0f} units")

        joint = joint_optimize(D, S, H, p, ltd)
        ind   = independent_solution(D, S, H, p, ltd)

        cost_gap_pct = (ind["TC_ind"] - joint["TC_joint"]) / ind["TC_ind"] * 100
        print(f"  Joint Q*={joint['Q_star']:.0f}  r*={joint['r_star']:.0f}  TC=€{joint['TC_joint']:,.0f}/yr  (converged in {joint['iterations']} iterations)")
        print(f"  Indep  Q*={ind['Q_ind']:.0f}    r*={ind['r_ind']:.0f}    TC=€{ind['TC_ind']:,.0f}/yr")
        print(f"  Cost gap: {cost_gap_pct:.1f}%  → joint saves €{ind['TC_ind']-joint['TC_joint']:,.0f}/yr")

        results[sku] = {
            "D": D, "S": S, "H": H, "p": p,
            "ltd_samples": ltd,
            "ss_type1": t1, "ss_type2": t2,
            "joint": joint, "independent": ind,
        }

    # Aggregate savings
    total_joint = sum(r["joint"]["TC_joint"] for r in results.values())
    total_ind   = sum(r["independent"]["TC_ind"] for r in results.values())
    total_ss_waste_eur = sum(
        (results[s]["ss_type1"]["safety_stock"] - results[s]["ss_type2"]["safety_stock"])
        * SKU_PARAMS[s]["unit_cost"] * SKU_PARAMS[s]["holding_rate"]
        for s in skus
    )

    print("\n" + "="*60)
    print("AGGREGATE FINDINGS")
    print("="*60)
    print(f"  Total TC (joint):       €{total_joint:>10,.0f}/year")
    print(f"  Total TC (independent): €{total_ind:>10,.0f}/year")
    print(f"  Savings from joint opt: €{total_ind-total_joint:>10,.0f}/year  ({(total_ind-total_joint)/total_ind*100:.1f}%)")
    print(f"  Wasted holding cost from Type1 vs Type2 confusion: €{total_ss_waste_eur:,.0f}/year")

    print("\nGenerating figures...")
    fig_type1_vs_type2(results,        f"{fig_dir}/m2_type1_vs_type2.png")
    fig_joint_vs_independent(results,  f"{fig_dir}/m2_joint_vs_independent.png")
    fig_total_cost_curves(results,     f"{fig_dir}/m2_cost_curves.png")
    fig_ltd_distributions(results,     f"{fig_dir}/m2_ltd_distributions.png")
    fig_policy_summary(results,        f"{fig_dir}/m2_policy_summary.png")

    # Save policy table
    policy_rows = []
    for sku in skus:
        r = results[sku]
        policy_rows.append({
            "sku": sku,
            "annual_demand": r["D"],
            "eoq": eoq(r["D"], r["S"], r["H"]),
            "Q_star_joint": r["joint"]["Q_star"],
            "r_star_joint": r["joint"]["r_star"],
            "SS_type1": r["ss_type1"]["safety_stock"],
            "SS_type2": r["ss_type2"]["safety_stock"],
            "TC_joint": r["joint"]["TC_joint"],
            "TC_independent": r["independent"]["TC_ind"],
            "annual_savings": r["independent"]["TC_ind"] - r["joint"]["TC_joint"],
        })
    pd.DataFrame(policy_rows).to_csv(f"{data_dir}/policy_recommendations.csv", index=False)
    print(f"  Saved: {data_dir}/policy_recommendations.csv")

    print("\n✓ Module 2 complete.")
    return results


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    run_module2(
        fig_dir=str(root / "outputs" / "figures"),
        data_dir=str(root / "data"),
    )
