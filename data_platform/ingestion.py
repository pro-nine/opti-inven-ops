from pathlib import Path
import pandas as pd


def load_demand_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "SKU_A_Yoghurt", "SKU_B_Detergent", "SKU_C_XmasCookies", "SKU_D_Coffee", "SKU_E_Bread", "SKU_F_Cleaning"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    return df


def to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    long_df = df.melt(id_vars=["date"], var_name="sku", value_name="demand")
    long_df["sku"] = long_df["sku"].str.replace("SKU_", "", regex=False)
    long_df["demand"] = pd.to_numeric(long_df["demand"], errors="raise")
    return long_df.sort_values(["date", "sku"]).reset_index(drop=True)
