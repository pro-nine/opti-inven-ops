from pathlib import Path
import duckdb
import pandas as pd


def build_warehouse(raw: pd.DataFrame, demand_long: pd.DataFrame, db_path: str | Path) -> None:
    con = duckdb.connect(str(db_path))
    con.register("raw_df", raw)
    con.register("demand_df", demand_long)
    con.execute("CREATE OR REPLACE TABLE raw_demand AS SELECT * FROM raw_df")
    con.execute("CREATE OR REPLACE TABLE demand AS SELECT date, sku, demand FROM demand_df")
    con.execute("""
        CREATE OR REPLACE TABLE demand_summary AS
        SELECT sku, MIN(date) AS first_date, MAX(date) AS last_date,
               COUNT(*) AS observations, AVG(demand) AS average_demand,
               STDDEV_SAMP(demand) AS demand_stddev,
               MIN(demand) AS minimum_demand, MAX(demand) AS maximum_demand
        FROM demand GROUP BY sku ORDER BY sku
    """)
    con.close()
