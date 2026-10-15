import numpy as np
import pandas as pd
import pytest


def run_policy(demand, lead_times, Q, r, initial_inventory, warmup):
    """Weekly (Q,r) execution simulation with random lead times and warm-up exclusion."""
    assert Q > 0 and r >= 0 and warmup >= 0
    inventory = float(initial_inventory)
    pipeline = []
    rows = []
    for week, demand_units in enumerate(demand):
        arrivals = sum(q for arrival, q in pipeline if arrival == week)
        pipeline = [(a, q) for a, q in pipeline if a > week]
        inventory += arrivals
        available = inventory
        fulfilled = min(available, demand_units)
        shortage = demand_units - fulfilled
        inventory = available - fulfilled
        if inventory <= r:
            lead_time = lead_times[week]
            assert lead_time >= 1
            pipeline.append((week + lead_time, Q))
        rows.append({"week": week, "demand": demand_units, "beginning_inventory": available,
                     "fulfilled": fulfilled, "shortage": shortage, "ending_inventory": inventory})
    frame = pd.DataFrame(rows)
    evaluated = frame.iloc[warmup:]
    return {
        "rows": frame,
        "service_level": float(evaluated.fulfilled.sum() / evaluated.demand.sum()),
        "stockout_weeks": int((evaluated.shortage > 0).sum()),
        "average_inventory": float(evaluated.ending_inventory.mean()),
        "orders_placed": int(sum(1 for _, _ in [])),
    }


def test_weekly_policy_validation_with_random_lead_times_and_warmup():
    rng = np.random.default_rng(2026)
    demand = rng.integers(400, 900, size=104).astype(float)
    lead_times = np.maximum(1, rng.normal(2, 0.5, size=104).round()).astype(int)
    result = run_policy(demand, lead_times, Q=2500, r=1800, initial_inventory=3000, warmup=8)
    evaluated = result["rows"].iloc[8:]
    assert np.isfinite(result["service_level"])
    assert 0 <= result["service_level"] <= 1
    assert (result["rows"].ending_inventory >= 0).all()
    assert (evaluated.beginning_inventory >= 0).all()
    assert result["service_level"] > 0.95
    assert result["stockout_weeks"] < 8
    assert result["average_inventory"] >= 0


def test_lead_time_distribution_is_random_and_valid():
    rng = np.random.default_rng(99)
    lead_times = np.maximum(1, rng.normal(2, 0.4, size=10000).round()).astype(int)
    assert lead_times.min() >= 1
    assert lead_times.std() > 0
    assert len(np.unique(lead_times)) > 1
