"""
Module 6: Recommendations & Business Case
==========================================
This module compiles every finding from Modules 1-5 into a single,
actionable operations management report. It answers the question
a hiring manager or operations director would actually ask:

  "You ran all this analysis. What should we DO, and what does it cost
   us if we don't do it?"

THREE DELIVERABLES:
1. Executive recommendation table — one row per SKU, one row for dock.
   Specific action, specific number, specific annual saving.
2. Savings waterfall — visual breakdown of where the total annual saving
   comes from across all interventions.
3. The counterintuitive finding — documented with the math, not just
   asserted, because an interviewer WILL ask you to prove it.

THE COUNTERINTUITIVE FINDING:
"Reducing order frequency on fast-movers by accepting larger batch sizes
 (increasing Q beyond Q*) actually INCREASES total annual cost when
 shortage penalties are properly accounted for — even though it reduces
 the number of purchase orders placed per year."

Most operations managers believe "fewer orders = lower cost" because they
see ordering cost as the dominant cost and assume safety stock is a fixed
given. This is only true if p=0 (no shortage cost). When shortage penalties
exist and Q grows above Q*, the safety stock required to maintain the same
service level grows SLOWER than the increase in average cycle stock, so
the TC curve rises — meaning large batch sizes are not "efficient" when
stockouts are penalised.

The math: TC(Q) = (D/Q)S + (Q/2 + SS)H + (D/Q)p*E[n(r*(Q))]
As Q increases beyond Q*:
  - (D/Q)S decreases (fewer orders, lower ordering cost) ← what managers see
  - (Q/2)H increases (more cycle stock, higher holding) ← they accept this
  - The r*(Q) = F^{-1}(1 - QH/(Dp)) DECREASES as Q grows (the optimal CSL
    falls, so the reorder point is lower) ← they don't account for this
  - E[n(r*(Q))] INCREASES as r falls (more expected shortages per cycle)
  - (D/Q)p*E[n(r*)] net effect is U-shaped — it first falls then rises
  - Combined TC curve has its minimum at Q* and rises on both sides
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import textwrap
import warnings
warnings.filterwarnings("ignore")

# ── Compiled results from all modules ────────────────────────────────────────
# These are the actual outputs produced by Modules 1-5.

FINAL_RESULTS = {
    # FORMAT: {sku: {policy_item: value}}
    "SKU_A_Yoghurt": {
        "strata": "Fast-mover, seasonal",
        "annual_demand": 45836,
        "cv": 0.171,
        "q_naive": 3964,     "q_star": 4062,
        "ss_naive": 1764,    "ss_type2": 158,    "ss_type1": 1073,
        "r_star": 3019,
        "tc_naive": 3853,    "tc_star": 3723,
        "annual_saving": 130,
        "action": "Switch from 2-wk SS rule to Type-2 SS (158 units). Order Q*=4,062.",
        "risk": "Medium",
    },
    "SKU_B_Detergent": {
        "strata": "Fast-mover, promo spikes",
        "annual_demand": 34568,
        "cv": 0.175,
        "q_naive": 3433,     "q_star": 3591,
        "ss_naive": 1330,    "ss_type2": 200,    "ss_type1": 840,
        "r_star": 2448,
        "tc_naive": 4385,    "tc_star": 4143,
        "annual_saving": 242,
        "action": "Switch to Type-2 SS (200 units). Promotional spikes: add a separate "
                  "promo override of +300 units 2 weeks before planned promotions.",
        "risk": "Medium",
    },
    "SKU_C_XmasCookies": {
        "strata": "Slow-mover, extreme seasonal",
        "annual_demand": 5762,
        "cv": 1.467,
        "q_naive": 1048,     "q_star": 1236,
        "ss_naive": 222,     "ss_type2": 387,    "ss_type1": 725,
        "r_star": 1064,
        "tc_naive": 2089,    "tc_star": 2069,
        "annual_saving": 20,
        "action": "Increase Q to 1,236 (current naive Q too small for high-CV demand). "
                  "Pre-position stock by week 40 each year — use the STL seasonal curve "
                  "from Module 1 as the ordering trigger, not a fixed calendar date.",
        "risk": "High — CV=1.47 means even the optimal policy has high cost variance.",
    },
    "SKU_D_Coffee": {
        "strata": "Slow-mover, stable",
        "annual_demand": 4883,
        "cv": 0.156,
        "q_naive": 1105,     "q_star": 1121,
        "ss_naive": 188,     "ss_type2": -11,    "ss_type1": 123,
        "r_star": 414,
        "tc_naive": 2010,    "tc_star": 2005,
        "annual_saving": 5,
        "action": "Minimal change required. Reduce safety stock to near-zero (Type-2 SS=-11 "
                  "means zero buffer needed). Current over-stocking by ~188 units unnecessarily.",
        "risk": "Low — stable demand, low CV.",
    },
    "SKU_E_Bread": {
        "strata": "Perishable, newsvendor",
        "annual_demand": 60895,
        "cv": 0.125,
        "q_naive": 1456,     "q_star": 1187,     # newsvendor Q*
        "ss_naive": "N/A",   "ss_type2": "N/A",  "ss_type1": "N/A",
        "r_star": "N/A",
        "tc_naive": 32387,   "tc_star": 41600,   # these are PROFIT not cost
        "annual_saving": 9213,   # profit recovered vs Type-1 policy
        "action": "Switch from 97.5th-pct ordering (Q=1,456) to newsvendor Q*=1,187. "
                  "Critical ratio = 0.54. Weekly profit gain: €177/week = €9,213/year.",
        "risk": "Low operational risk. High CFO scrutiny risk: reducing order quantity "
                "requires explaining to management why ordering LESS bread is better.",
    },
    "SKU_F_Cleaning": {
        "strata": "Intermittent/lumpy",
        "annual_demand": 2131,
        "cv": 1.107,
        "q_naive": 702,      "q_star": 760,
        "ss_naive": 82,      "ss_type2": 70,     "ss_type1": 230,
        "r_star": 400,
        "tc_naive": 2159,    "tc_star": 2152,
        "annual_saving": 7,
        "action": "Minimal Q change. The real action: switch to a periodic-review "
                  "policy with review period R=4 weeks (Croston's method for intermittent "
                  "demand is the production-grade upgrade beyond this project's scope).",
        "risk": "Medium — intermittent demand makes all metrics noisier.",
    },
}

DOCK_RESULT = {
    "annual_saving_trucks": 274906,
    "annual_saving_net":    249906,
    "capex":                180000,
    "payback_months":       8.6,
    "npv_10yr":            1496890,
    "action": "Add third dock door (c=2→c=3). ρ drops 0.75→0.50. "
              "Wq drops 38.6→4.7 min. Payback 8.6 months, 10-yr NPV €1.5M.",
    "risk": "Low financial risk given 8.6-month payback. "
            "Operational risk: construction disruption for ~3 weeks.",
}

TYPE1_TYPE2_WASTE = 2355   # from Module 2

# ── Savings waterfall data ────────────────────────────────────────────────────

def build_waterfall():
    items = [
        ("Dock expansion\n(Module 4)",      DOCK_RESULT["annual_saving_net"],       "#264653"),
        ("Bread newsvendor\n(Module 3)",     FINAL_RESULTS["SKU_E_Bread"]["annual_saving"], "#2a9d8f"),
        ("Type1→Type2 SS\n(Module 2)",       TYPE1_TYPE2_WASTE,                     "#2a9d8f"),
        ("SKU-B joint opt.\n(Module 2)",     FINAL_RESULTS["SKU_B_Detergent"]["annual_saving"], "#2a9d8f"),
        ("SKU-A joint opt.\n(Module 2)",     FINAL_RESULTS["SKU_A_Yoghurt"]["annual_saving"],   "#2a9d8f"),
        ("SKU-C adjustment\n(Module 2)",     FINAL_RESULTS["SKU_C_XmasCookies"]["annual_saving"],"#e9c46a"),
        ("SKU-D + F minor\n(Module 2)",
         FINAL_RESULTS["SKU_D_Coffee"]["annual_saving"] +
         FINAL_RESULTS["SKU_F_Cleaning"]["annual_saving"], "#e9c46a"),
    ]
    return items


# ── Counterintuitive finding math ─────────────────────────────────────────────

def counterintuitive_tc_curve(D, S, H, p, ltd_samples):
    """
    Compute TC(Q) sweeping Q around Q* to show TC RISES when Q grows above Q*.
    The 'fewer orders = lower cost' intuition breaks down here.
    """
    mu_ltd = ltd_samples.mean()
    Q_star_base = np.sqrt(2 * D * S / H)  # EOQ as starting point

    Q_range = np.linspace(Q_star_base * 0.4, Q_star_base * 2.5, 200)

    TC_full, TC_ordering, TC_holding, TC_shortage, r_vals, csl_vals = [], [], [], [], [], []

    for Q in Q_range:
        csl = np.clip(1 - Q * H / (D * p), 0.50, 0.9999)
        r   = np.quantile(ltd_samples, csl)
        SS  = r - mu_ltd
        En  = np.mean(np.maximum(ltd_samples - r, 0))

        ord_c  = (D / Q) * S
        hld_c  = (Q / 2 + SS) * H
        shr_c  = (D / Q) * p * En

        TC_ordering.append(ord_c)
        TC_holding.append(hld_c)
        TC_shortage.append(shr_c)
        TC_full.append(ord_c + hld_c + shr_c)
        r_vals.append(r)
        csl_vals.append(csl)

    return (Q_range, np.array(TC_full), np.array(TC_ordering),
            np.array(TC_holding), np.array(TC_shortage),
            np.array(r_vals), np.array(csl_vals))


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_recommendations_table(out_path: str):
    skus_order = ["SKU_A_Yoghurt","SKU_B_Detergent","SKU_C_XmasCookies",
                  "SKU_D_Coffee","SKU_E_Bread","SKU_F_Cleaning"]
    
    # Exactly 8 columns
    cols = ["SKU", "Strata", "CV", "Q* (units)", "SS Recommended",
            "Annual Saving (€)", "Risk", "Key Action"]
    rows = []
    
    # Text wrap limits for the wider columns
    risk_width = 30
    action_width = 45

    # 1. Populate SKU rows (8 elements per row)
    for sku in skus_order:
        r = FINAL_RESULTS[sku]
        ss_rec = r["ss_type2"] if isinstance(r["ss_type2"], str) else \
                 f"{r['ss_type2']:.0f}" if r["ss_type2"] >= 0 else "0 (none needed)"
        
        rows.append([
            sku.replace("SKU_","").replace("_"," "),
            r["strata"], 
            f"{r['cv']:.3f}", 
            f"{r['q_star']:,}",
            ss_rec, 
            f"€{r['annual_saving']:,.0f}",
            textwrap.fill(r["risk"], width=risk_width),
            textwrap.fill(r["action"], width=action_width),
        ])
        
    # 2. Populate DOCK row (8 elements)
    rows.append([
        "DOCK", 
        "Receiving dock", 
        "–", 
        "3 doors", 
        "–",
        f"€{DOCK_RESULT['annual_saving_net']:,.0f}",
        textwrap.fill("Low (8.6-mo payback)", width=risk_width), 
        textwrap.fill(DOCK_RESULT["action"], width=action_width)
    ])

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(18, 5.5))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 4.0)

    # Exactly 8 width values
    col_widths = [0.09, 0.12, 0.04, 0.07, 0.10, 0.09, 0.18, 0.31]
    for i in range(len(rows) + 1):
        for j in range(len(cols)):
            tbl[i, j].set_width(col_widths[j])

    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#264653")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
        
    highlight_colors = {
        0: "#d4edda", 1: "#d4edda", 2: "#fff3cd", 3: "#fff3cd",
        4: "#d4edda", 5: "#fff3cd", 6: "#d4f5e9"
    }
    
    for i in range(1, len(rows) + 1):
        bg = highlight_colors.get(i - 1, "white")
        for j in range(len(cols)):
            tbl[i, j].set_facecolor(bg)
            
    tbl[len(rows), len(cols)-3].set_facecolor("#d4edda")
    
    ax.set_title("Nordpack GmbH — Operations Analyst Recommendations Summary\n"
                 "Source: DistroSense Analytics | Modules 1–5",
                 fontsize=11, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_savings_waterfall(out_path: str):
    items = build_waterfall()
    labels  = [i[0] for i in items]
    values  = [i[1] for i in items]
    colors  = [i[2] for i in items]
    total   = sum(values)

    running = 0
    lefts   = []
    for v in values:
        lefts.append(running)
        running += v

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (label, val, color, left) in enumerate(zip(labels, values, colors, lefts)):
        ax.bar(i, val, bottom=left, color=color, width=0.55, alpha=0.88)
        ax.text(i, left + val / 2, f"€{val:,.0f}",
                ha="center", va="center", fontsize=8, fontweight="bold", color="white")

    # Total bar
    ax.bar(len(items), total, color="#264653", width=0.55, alpha=0.88)
    ax.text(len(items), total / 2, f"€{total:,.0f}\nTotal",
            ha="center", va="center", fontsize=9, fontweight="bold", color="white")

    ax.set_xticks(list(range(len(items))) + [len(items)])
    ax.set_xticklabels(labels + ["TOTAL\nAnnual\nSaving"], fontsize=8)
    ax.set_ylabel("Annual Saving (€)")
    ax.set_title("Annual Savings Waterfall — All Recommendations Combined\n"
                 "Dock expansion alone justifies the engagement | Bread newsvendor is the "
                 "highest single-SKU finding",
                 fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"€{v:,.0f}"))

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#264653", label="Large impact (>€100k)"),
        Patch(facecolor="#2a9d8f", label="Medium impact (€5k–€100k)"),
        Patch(facecolor="#e9c46a", label="Small impact (<€5k)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_counterintuitive_finding(ltd_samples, out_path: str):
    """
    The TC(Q) curve showing Q* as the true minimum, and proving that
    Q > Q* raises TC even though ordering cost falls — because shortage
    cost rises faster.
    """
    b = {"D": 45836, "S": 120, "H": 0.28 * 2.50, "p": 8.00}
    Q_range, TC, ord_c, hld_c, shr_c, r_vals, csl_vals = \
        counterintuitive_tc_curve(b["D"], b["S"], b["H"], b["p"], ltd_samples)

    Q_star_idx = np.argmin(TC)
    Q_star     = Q_range[Q_star_idx]
    Q_eoq      = np.sqrt(2 * b["D"] * b["S"] / b["H"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: stacked cost decomposition
    ax = axes[0]
    ax.stackplot(Q_range, ord_c, hld_c, shr_c,
                 labels=["Ordering cost (D/Q)×S",
                          "Holding cost (Q/2+SS)×H",
                          "Shortage cost (D/Q)×p×E[n(r*(Q))]"],
                 colors=["#e9c46a", "#264653", "#e76f51"], alpha=0.78)
    ax.plot(Q_range, TC, color="black", lw=2.5, label="Total Cost TC(Q)")
    ax.axvline(Q_star, color="#2a9d8f", lw=2, ls="-",
               label=f"Q* (joint optimum) = {Q_star:.0f}")
    ax.axvline(Q_eoq, color="#e76f51",  lw=2, ls="--",
               label=f"EOQ (no shortage cost) = {Q_eoq:.0f}")
    ax.scatter([Q_star], [TC[Q_star_idx]], s=100, color="#2a9d8f", zorder=10)
    ax.set_xlabel("Order Quantity Q (units)")
    ax.set_ylabel("Annual Cost (€)")
    ax.set_title("Cost Decomposition — TC(Q) with jointly optimal r*(Q)\n"
                 "Shortage cost RISES past Q* because optimal CSL falls",
                 fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"€{v:,.0f}"))
    ax.set_ylim(0, TC.max() * 1.15)
    ax.legend(fontsize=7.5, loc="upper right")

    # Right: optimal CSL and reorder point as Q grows
    ax2 = axes[1]
    ax2_r = ax2.twinx()
    ax2.plot(Q_range, csl_vals * 100, color="#264653", lw=2,
             label="Optimal CSL (%) = 1 - QH/(Dp)")
    ax2_r.plot(Q_range, r_vals, color="#e76f51", lw=2, ls="--",
               label="Reorder point r*(Q) (units)")
    ax2.axvline(Q_star, color="#2a9d8f", lw=1.5, ls="-", label=f"Q* = {Q_star:.0f}")
    ax2.set_xlabel("Order Quantity Q (units)")
    ax2.set_ylabel("Optimal Cycle Service Level (%)", color="#264653")
    ax2_r.set_ylabel("Reorder Point r* (units)", color="#e76f51")
    ax2.set_title("Why 'Bigger Orders = Lower Cost' is Wrong Here\n"
                  "As Q grows, optimal CSL falls → r* falls → more expected shortages",
                  fontweight="bold")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    fig.suptitle("THE COUNTERINTUITIVE FINDING — SKU-A Yoghurt\n"
                 "Ordering MORE than Q* increases total cost despite lower ordering cost,\n"
                 "because the jointly optimal reorder point falls, creating more stockouts",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_implementation_roadmap(out_path: str):
    """Implementation timeline — what to do in which order."""
    phases = [
        ("QUICK WIN\n(Week 1–4)", [
            "Update SKU-E Bread order qty: 1,187 units/week  [€9,213/yr]",
            "Zero SKU-D Coffee safety stock (free up warehouse space now)",
            "Update SKU-A, B, F safety stock to Type-2 values  [€2,355/yr]",
        ], "#2a9d8f"),
        ("MEDIUM TERM\n(Month 2–4)", [
            "Re-optimize Q* for SKU-A, B, C, D, F using joint (Q,r) method  [€432/yr]",
            "Set up promotional override process for SKU-B Detergent",
            "Apply STL seasonal trigger for SKU-C Xmas Cookies pre-positioning",
        ], "#264653"),
        ("CAPITAL DECISION\n(Month 3–6)", [
            "Board approval: Dock expansion capex €180,000  [NPV €1.5M, 8.6-mo payback]",
            "Construction planning: 3-week operational disruption window",
            "Commission third dock door; re-calibrate M/M/c model with new μ",
        ], "#e76f51"),
        ("ONGOING\n(Quarterly)", [
            "Re-run STL decomposition on rolling 3-year demand window",
            "Re-estimate LTD distribution as lead-time variance evolves",
            "Monte Carlo re-run: update parameter distributions from actuals",
        ], "#e9c46a"),
    ]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")
    col_w = 1.0 / len(phases)
    for i, (phase_label, actions, color) in enumerate(phases):
        x = i * col_w + 0.01
        ax.add_patch(plt.Rectangle((x, 0.72), col_w - 0.02, 0.25,
                                    facecolor=color, alpha=0.85, transform=ax.transAxes))
        ax.text(x + (col_w - 0.02) / 2, 0.845, phase_label, ha="center",
                va="center", fontsize=9, fontweight="bold", color="white",
                transform=ax.transAxes)
        for j, action in enumerate(actions):
            y_pos = 0.62 - j * 0.18
            ax.add_patch(plt.Rectangle((x, y_pos), col_w - 0.02, 0.15,
                                        facecolor=color, alpha=0.18,
                                        transform=ax.transAxes))
            
            # Pre-wrap the text to fit within the box width, and indent subsequent lines
            wrapped_text = textwrap.fill(f"• {action}", width=40, subsequent_indent="  ")
            
            # Removed wrap=True, as textwrap handles it now
            ax.text(x + 0.01, y_pos + 0.075, wrapped_text, ha="left",
                    va="center", fontsize=7.5,
                    transform=ax.transAxes)

    ax.set_title("DistroSense Implementation Roadmap — Nordpack GmbH\n"
                 f"Total identified annual benefit: "
                 f"€{sum(r['annual_saving'] for r in FINAL_RESULTS.values()) + DOCK_RESULT['annual_saving_net']:,.0f}/year",
                 fontsize=11, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_module6(ltd_samples=None, fig_dir="outputs/figures",
                data_dir="data") -> dict:
    import os
    os.makedirs(fig_dir, exist_ok=True)

    print("\n" + "="*60)
    print("MODULE 6: FINAL RECOMMENDATIONS & BUSINESS CASE")
    print("="*60)

    # Aggregate financials
    sku_savings = sum(r["annual_saving"] for r in FINAL_RESULTS.values())
    total_savings = sku_savings + DOCK_RESULT["annual_saving_net"]

    print(f"\n  SKU-level annual savings:    €{sku_savings:,.0f}")
    print(f"  Dock expansion net savings:  €{DOCK_RESULT['annual_saving_net']:,.0f}")
    print(f"  Type 1→2 SS waste recovered: €{TYPE1_TYPE2_WASTE:,.0f}")
    print(f"  ─────────────────────────────────────")
    print(f"  TOTAL annual benefit:        €{total_savings:,.0f}")
    print(f"  Dock payback:                {DOCK_RESULT['payback_months']:.1f} months")
    print(f"  Dock 10-yr NPV:              €{DOCK_RESULT['npv_10yr']:,.0f}")

    print("\n  Per-SKU recommendation summary:")
    for sku, r in FINAL_RESULTS.items():
        print(f"  {sku:22s}: Q*={r['q_star']:,}  "
              f"SS_rec={r['ss_type2'] if isinstance(r['ss_type2'],str) else int(r['ss_type2']):>5}  "
              f"saving=€{r['annual_saving']:,}")

    print("\n  Generating figures...")

    # Need LTD samples for SKU-A to draw the counterintuitive finding
    if ltd_samples is None:
        # Generate them fresh if not passed in
        from src.module5_montecarlo import simulate_ltd_fast, BASELINE
        b = BASELINE["SKU_A_Yoghurt"]
        ltd_samples = simulate_ltd_fast(b["mu_w"], b["sig_resid"],
                                         b["mu_L"], b["sig_L"], n_sim=10000)

    fig_recommendations_table(f"{fig_dir}/m6_recommendations.png")
    fig_savings_waterfall(f"{fig_dir}/m6_savings_waterfall.png")
    fig_counterintuitive_finding(ltd_samples, f"{fig_dir}/m6_counterintuitive.png")
    fig_implementation_roadmap(f"{fig_dir}/m6_roadmap.png")

    print(f"\n✓ Module 6 complete.")
    return {"total_savings": total_savings, "sku_savings": sku_savings}


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    run_module6(fig_dir=str(root/"outputs"/"figures"),
                data_dir=str(root/"data"))
