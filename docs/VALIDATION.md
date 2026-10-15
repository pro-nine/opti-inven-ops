# Validation scope and reproducible commands

The normal data-platform workflow runs unit tests on Python 3.11/3.12/3.13 and the integrated warehouse/Power BI CSV pipeline only on 3.12. The manual `validation` workflow additionally runs the model benchmarks and builds and executes the Docker image; its success is required before claiming Docker runtime verification.

## Commands

```bash
python -m pip install -r Requirements.txt pytest
python -m pytest -q
python -m src.robustness
python -m data_platform.pipeline
docker build -t opti-inven-ops .
docker run --rm opti-inven-ops
```

Benchmark outputs: outputs/validation/inventory_coverage.csv, bread_loss.csv, queue_robustness.csv and REPORT.md. The manual workflow publishes them as an artifact. Generated CSVs can be imported in Power BI; no Power BI dashboard/service deployment is included.

## Experimental definitions

- Demand benchmarks use expanding chronological history beginning at 52 weeks and evaluate only subsequent observations. No future observations fit the seasonal profile.
- Inventory: normal, contiguous empirical totals and seasonal-residual protection forecasts; compare mean rounded lead time L against L+1 for weekly review. Overlapping-window coverage is NOT executed cycle-service level or fill rate, and fixed lead-time experiments do not replace random lead-time execution validation.
- Bread: compare mean, normal quantile, exact empirical quantile and seasonal-residual orders; measure waste, shortfall and overage/underage loss on held-out weeks. Bread is not treated as carry-forward inventory.
- Queue: common arrivals and service times for 2 versus 3 doors, Poisson arrivals at 3/hour, service rate 2/hour, service CV 0/1/2. Use five seeds and discard 2,000 arrivals per run. Parameters are a stress fixture, not verified operational telemetry; no financial claim follows from it.

## Local baseline on the attached 156-week synthetic demand snapshot

Held-out mean bread loss per week: mean 156.94 EUR; normal quantile 146.90 EUR; empirical quantile 143.30 EUR; seasonal residual 157.11 EUR. Seasonality is not automatically better.

Mean waiting minutes, averaged over five seeds: CV=0, 19.96 (2 doors) / 2.60 (3); CV=1, 38.43 / 4.70; CV=2, 95.99 / 11.10. These are Monte Carlo estimates, not savings or service guarantees.

Four new mathematical/benchmark tests passed locally on Python 3.12. CI and Docker execution results must be checked after committing.

## Remaining work explicitly not covered by this increment

Standalone module1 still needs its direct source-level frozen-CDF repair; the integrated runner already uses the compatibility adapter. Persistent failure/success run history is not implemented. Weekly executed (Q,r) simulations with random lead times and start-up controls remain separate from the protection-window benchmark. Do not describe this increment as completion of those tasks.
