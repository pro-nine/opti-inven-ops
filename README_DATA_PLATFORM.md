# Data Platform Upgrade

This branch adds a portfolio-oriented data platform layer without changing the existing six analytical modules.

## Flow

```text
data/demand_weekly.csv
        -> ingestion and wide-to-long transformation
        -> quality checks
        -> DuckDB raw and analytical tables
        -> Power BI CSV exports
```

## Tables

- `raw_demand`: original wide-format source data.
- `demand`: long-format table with `date`, `sku`, and `demand`.
- `demand_summary`: SKU-level profiling table.

## Quality checks

The pipeline checks for an empty table, null dates, duplicate date-SKU pairs, null demand, and negative demand.

## Run locally

```bash
pip install -r requirements-data-platform.txt
python -m data_platform.pipeline
pytest -q
```

The existing inventory, newsvendor, queueing, Monte Carlo, and recommendation modules remain unchanged. The next upgrade should connect their outputs to warehouse tables such as `inventory_policy`, `newsvendor_policy`, `dock_capacity`, `risk_summary`, and `recommendations`.
