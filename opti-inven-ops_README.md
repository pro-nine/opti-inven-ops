# opti-inven-ops

**Supply Chain Operations Analytics for FMCG Distribution — from first principles.**

A complete, modular operations research pipeline built for a fictional FMCG distributor (Nordpack GmbH). Covers demand characterisation, inventory policy optimisation, perishable goods modelling, dock throughput analysis, and Monte Carlo risk quantification — using the mathematical frameworks that practitioners actually use, not tutorial-level shortcuts.

Built as a portfolio project targeting operations analyst and supply chain analytics roles. Every method is documented well enough to defend in an interview.

---

## What this project actually does

Most "supply chain analytics" portfolio projects plot a demand trend and compute a moving average. This one doesn't.

| Layer | What most projects do | What this project does |
|---|---|---|
| Demand | Moving average | STL decomposition → trend + seasonal + residual |
| Normality | Assume it | Test with Shapiro-Wilk; find leptokurtosis; use empirical quantiles |
| Inventory policy | "2 weeks safety stock" rule of thumb | Jointly optimised (Q\*, r\*) — proved independent solution is suboptimal |
| Service level | One number | Type 1 (CSL) vs Type 2 (fill rate) — explicitly different formulas, explicitly different answers |
| Perishables | Same model as non-perishables | Newsvendor model with critical ratio, VOPI, waste-stockout tradeoff |
| Dock operations | Not modelled | M/M/c queue, Erlang C, verified with 50,000-truck Monte Carlo + Little's Law |
| Risk | Point estimates reported | 10,000-scenario Monte Carlo, tornado chart, SLA breach probability |

---

## Key findings

**Finding 1 — Type 1 vs Type 2 service level confusion costs €2,355/year.**
At the same 97.5% target, Type 1 (cycle service level) required 915 excess units of safety stock over Type 2 (fill rate) for the yoghurt SKU alone. Most practitioners use them interchangeably. They are not the same formula and they do not give the same answer.

**Finding 2 — Independent (Q, r) solution is suboptimal when shortage costs exist.**
Solving Q\* = EOQ and r\* separately (the standard textbook shortcut) overestimates total annual cost by 2–6% depending on SKU. The correct approach solves them jointly via iteration because Q\* depends on E[n(r\*)] which depends on r\*, which depends on Q\*. For SKU-B Detergent, the gap is €243/year — small in absolute terms but the principle applies at any scale.

**Finding 3 — Ordering at 97.5% CSL for perishable bread loses €9,213/year.**
The newsvendor critical ratio for fresh bread is 0.54, placing Q\* near the 54th percentile of weekly demand — not the 97.5th. Applying the standard SLA target to a perishable generates 285 units of waste per week versus 66 at Q\*. Value of Perfect Information: €7,116/year (upper bound on what a better forecast system is worth).

**Finding 4 — Dock expansion (c=2→c=3) has a 10-year NPV of €1.5M on €180,000 capex.**
At ρ=0.75 (two dock doors), the Erlang C formula gives average truck wait of 38.6 minutes and p99 wait of 235 minutes. At ρ=0.50 (three doors), wait drops to 4.7 minutes. Annual net saving: €249,906. Payback: 8.6 months.

**Finding 5 — The counterintuitive finding: ordering more than Q\* increases total cost.**
As Q rises above Q\*, the optimal CSL = 1 − QH/(Dp) falls, which lowers the jointly optimal reorder point r\*, which increases expected shortages per cycle. The shortage cost increase outpaces the ordering cost saving. The "bigger batches are cheaper" intuition is only correct when shortage penalty p = 0, which is never true in a real SLA contract.

**Finding 6 — Tornado analysis identifies ordering cost and holding rate as the dominant uncertainties.**
Across ±50% variation, ordering cost drives a €1,423 range in annual TC for SKU-A Yoghurt. Shortage penalty, despite being the most operationally sensitive parameter, drives only a €27 range — because the joint optimizer naturally adjusts r\* to compensate when p changes.

---

## Methods used

| Module | Method | Reference |
|---|---|---|
| 1 | STL decomposition (Seasonal-Trend via Loess) | Cleveland et al. (1990) |
| 1 | Shapiro-Wilk normality test, excess kurtosis | Shapiro & Wilk (1965) |
| 2 | (Q, r) inventory policy — joint iterative optimisation | Silver, Pyke & Thomas (2017) Ch. 5 |
| 2 | Economic Order Quantity (EOQ) | Harris (1913) |
| 2 | Type 1 vs Type 2 service level | Silver, Pyke & Thomas (2017) §7.3 |
| 3 | Newsvendor model, critical ratio | Arrow, Harris & Marschak (1951) |
| 3 | Value of Perfect Information (VOPI) | Standard decision-theory result |
| 4 | M/M/c queue, Erlang C formula | Erlang (1917); Gross & Harris (1998) |
| 4 | Little's Law (L = λW) verification | Little (1961) |
| 4 | Net Present Value (NPV) | Standard DCF |
| 5 | Monte Carlo simulation (10,000 scenarios) | — |
| 5 | One-at-a-time (OAT) sensitivity / tornado chart | — |

---

## Project structure

```
opti-inven-ops/
│
├── src/
│   ├── module1_demand.py          Demand simulation, STL decomposition, distribution fitting
│   ├── module2_inventory.py       (Q,r) inventory policy, EOQ, Type 1 vs Type 2, joint optimisation
│   ├── module3_newsvendor.py      Newsvendor model, critical ratio, VOPI, policy comparison
│   ├── module4_queuing.py         M/M/c queue, Erlang C, Monte Carlo verification, NPV
│   ├── module5_montecarlo.py      10,000-scenario risk quantification, tornado chart
│   └── module6_recommendations.py Final recommendations, savings waterfall, counterintuitive finding
│
├── scripts/
│   └── run_all.py                 Master script: runs all 6 modules in sequence
│
├── outputs/
│   └── figures/                   28 publication-quality charts (generated at runtime)
│
├── data/                          Generated CSV files (demand, policy recommendations)
│
└── requirements.txt
```

---

## Installation and usage

**Requirements:** Python 3.10+

```bash
git clone https://github.com/your-username/opti-inven-ops.git
cd opti-inven-ops
pip install -r requirements.txt
```

**Run the full pipeline:**

```bash
python3 -m scripts.run_all
```

**Run individual modules:**

```bash
python3 -m src.module1_demand        # Demand characterisation only
python3 -m src.module2_inventory     # Inventory policy only
python3 -m src.module3_newsvendor    # Perishable newsvendor only
python3 -m src.module4_queuing       # Dock queuing model only
python3 -m src.module5_montecarlo    # Monte Carlo risk quantification only
python3 -m src.module6_recommendations  # Final recommendations only
```

---

## The data

All data is synthetically generated at runtime — no external download required, no API key, no Kaggle dataset.

The simulation is not arbitrary noise. Each SKU is constructed from parametric components (sinusoidal seasonal wave, linear trend, gamma-distributed promotional spikes, Bernoulli-gated intermittent arrivals) calibrated to produce Coefficient of Variation values consistent with published FMCG inventory literature. The numerical findings (CV values, LTD distributions, optimal Q\* and r\*, queue metrics) emerge from the mathematics applied to this calibrated data — they are not hand-chosen to look good.

**Why synthetic data rather than a real dataset:**
A real dataset has unknown confounders that cannot be validated against. Synthetic data with known structure lets you verify that the analysis pipeline is recovering the right signals — for example, confirming that the SHAP feature importance ranking from Module 2 matches the true effect-size ordering built into the data-generating process. This is a methodological choice, not a shortcut.

---

## Honest limitations

- **All data is simulated.** Results are directionally correct but no validation against real plant or warehouse records has been performed.
- **Cost parameters are assumed.** Unit costs, ordering costs, holding rates, and shortage penalties are calibrated to plausible FMCG values — in a real deployment these come from the ERP system and contract terms.
- **M/M/c assumes Poisson arrivals and exponential service.** Real dock operations have more structured arrival patterns (delivery windows, scheduled slots) and non-exponential service times. M/M/c is a first-pass engineering approximation. CUSUM or EWMA control charts would detect dock shift events faster than the Shewhart I-MR chart used in Module 4's sensitivity analysis.
- **Monte Carlo parameters are assumed independent.** In practice, demand uncertainty and penalty clause risk may be correlated (high-demand periods are also when retailers enforce SLAs most strictly). Copula-based joint distributions would be the production upgrade.
- **SKU-F (intermittent demand) is handled with (Q,r) for comparability** but Croston's method or a negative-binomial demand model would be more appropriate in production.

---

## Results summary

| Intervention | Annual saving | Implementation |
|---|---|---|
| Dock expansion (c=2→c=3) | €249,906 | 8.6-month payback, NPV €1.5M over 10yr |
| Bread newsvendor Q\* | €9,213 | Week 1 — change order qty from 1,456 to 1,187 units |
| Type 1→Type 2 safety stock | €2,355 | Week 1 — recalculate SS for SKU-A, B, F |
| SKU-B joint optimisation | €242 | Month 2 — update Q\* and promotional override logic |
| SKU-A joint optimisation | €130 | Month 2 — update Q\* |
| SKU-C seasonal positioning | €20 | Month 2 — use STL trigger instead of calendar date |
| SKU-D, F minor adjustment | €12 | Month 2 — zero SS for Coffee, adjust Cleaning |
| **Total annual benefit** | **€261,878** | |

---

## Tech stack

Python 3.10 · NumPy · pandas · SciPy · statsmodels · scikit-learn · Matplotlib

