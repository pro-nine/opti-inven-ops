"""Out-of-sample demand and service-distribution robustness benchmarks.

Inventory tests lead-time-demand coverage, not executed (Q,r) cycle service.
Training always precedes evaluation; no synthetic-generator re-use. Queue
comparisons share arrivals and gamma base draws across capacity alternatives.
"""
from pathlib import Path
import argparse
import heapq
import numpy as np
import pandas as pd
from scipy import stats
from src.module2_inventory import SKU_PARAMS
from src.module3_newsvendor import BREAD_PARAMS, compute_costs


def protection_forecast(history, horizon, method, target=0.975):
    history = np.asarray(history, float)
    if len(history) < 52 or horizon < 1:
        raise ValueError("Need 52 historical weeks and a positive horizon")
    if method == "seasonal":
        phase = np.arange(len(history)) % 52
        profile = np.array([history[phase == k].mean() for k in range(52)])
        residuals = history - profile[phase]
        point = profile[(len(history) + np.arange(horizon)) % 52].sum()
        errors = np.convolve(residuals, np.ones(horizon), mode="valid")
        return max(0.0, point + np.quantile(errors, target))
    if method == "normal":
        return max(0.0, history.mean()*horizon + stats.norm.ppf(target)*history.std(ddof=1)*np.sqrt(horizon))
    if method == "empirical":
        totals = np.convolve(history, np.ones(horizon), mode="valid")
        return float(np.quantile(totals, target))
    raise ValueError(method)


def demand_backtest(frame):
    rows = []
    for sku in SKU_PARAMS:
        demand = frame[sku].to_numpy(float)
        if sku == "SKU_E_Bread":
            continue
        lead = max(1, int(round(SKU_PARAMS[sku]["lead_time_mean"])))
        for review in [0, 1]:
            horizon = lead + review
            for method in ["normal", "empirical", "seasonal"]:
                pred, actual = [], []
                for t in range(52, len(demand)-horizon+1):
                    pred.append(protection_forecast(demand[:t], horizon, method))
                    actual.append(demand[t:t+horizon].sum())
                pred, actual = np.array(pred), np.array(actual)
                rows.append({"sku": sku, "method": method, "review_weeks": review,
                    "protection_weeks": horizon, "windows": len(actual), "coverage": float((actual<=pred).mean()),
                    "mean_buffer_units": float(pred.mean()), "mean_shortfall_units": float(np.maximum(actual-pred,0).mean())})
    return pd.DataFrame(rows)


def bread_backtest(frame):
    demand = frame["SKU_E_Bread"].to_numpy(float)
    costs = compute_costs(BREAD_PARAMS)
    cr, cu, co = costs["CR"], costs["Cu"], costs["Co"]
    rows=[]
    for method in ["mean", "normal_quantile", "empirical_quantile", "seasonal_residual"]:
        orders=[]
        for t in range(52,len(demand)):
            hist=demand[:t]
            if method=="mean": q=hist.mean()
            elif method=="normal_quantile": q=hist.mean()+stats.norm.ppf(cr)*hist.std(ddof=1)
            elif method=="empirical_quantile": q=np.quantile(hist,cr,method="inverted_cdf")
            else: q=protection_forecast(hist,1,"seasonal",cr)
            orders.append(max(0,float(q)))
        actual=demand[52:]; q=np.asarray(orders); waste=np.maximum(q-actual,0); short=np.maximum(actual-q,0)
        loss=co*waste+cu*short
        rows.append({"method":method,"weeks":len(actual),"critical_ratio":cr,
            "mean_waste":float(waste.mean()),"mean_shortfall":float(short.mean()),
            "mean_loss_eur":float(loss.mean()),"annualised_loss_eur":float(loss.mean()*52)})
    return pd.DataFrame(rows)


def queue_waits(arrivals,service,doors):
    if doors<1: raise ValueError("doors must be positive")
    free=[0.0]*doors; heapq.heapify(free); waits=[]
    for arrival,duration in zip(arrivals,service):
        available=heapq.heappop(free); start=max(available,float(arrival))
        waits.append(start-arrival); heapq.heappush(free,start+duration)
    return np.asarray(waits)


def queue_benchmark(n=30000,seeds=(11,12,13,14,15),warmup=2000):
    if warmup>=n: raise ValueError("warmup must be less than arrivals")
    rows=[]
    # Stress-test fixture, not a claim about the repository's operational arrival rate.
    lam,mu=3.0,2.0
    for cv in [0.0,1.0,2.0]:
        for seed in seeds:
            rng=np.random.default_rng(seed)
            arrivals=np.cumsum(rng.exponential(1/lam,n))
            service=np.full(n,1/mu) if cv==0 else rng.gamma(1/cv**2,cv**2/mu,n)
            for doors in [2,3]:
                waits=queue_waits(arrivals,service,doors)[warmup:]
                rows.append({"service_cv":cv,"seed":seed,"doors":doors,
                    "arrival_rate":lam,"service_rate":mu,"mean_wait_minutes":float(waits.mean()*60),
                    "p95_wait_minutes":float(np.quantile(waits,.95)*60)})
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--fast",action="store_true")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]; out=root/"outputs"/"validation";out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(root/"data"/"demand_weekly.csv")
    tables={"inventory_coverage":demand_backtest(frame),"bread_loss":bread_backtest(frame),
            "queue_robustness":queue_benchmark(n=5000 if args.fast else 30000,seeds=(11,) if args.fast else (11,12,13,14,15),warmup=500 if args.fast else 2000)}
    for name,table in tables.items():table.to_csv(out/(name+".csv"),index=False)
    report=["# Robustness benchmarks", "", "Chronological demand holdout begins after week 52. Inventory coverage is overlapping protection-window coverage, not realised cycle service or fill rate. Lead times here are fixed rounded means; random lead-time validation remains separate.", "", "Bread loss uses overage/underage costs and true observed weekly demand; it equals regret against perfect foresight under the stated single-period model. Annualised values are projections, not realised savings.", "", "Queue sensitivity uses an explicit fixture (arrival rate 3/hour, service rate 2/hour), 2 versus 3 doors, service CV 0, 1 and 2, and a discarded warm-up. No capex recommendation is inferred."]
    for name,table in tables.items():report.extend(["", "## "+name, "", table.round(4).to_csv(index=False)])
    (out/"REPORT.md").write_text("\n".join(report),encoding="utf-8")
    print("Wrote benchmarks to",out)


if __name__=="__main__":main()
