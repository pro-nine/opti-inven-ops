import pandas as pd


def validate_demand(df: pd.DataFrame) -> dict:
    errors = []
    if df.empty:
        errors.append("demand table is empty")
    if df["date"].isna().any():
        errors.append("date contains nulls")
    if df[["date", "sku"]].duplicated().any():
        errors.append("duplicate date-SKU rows found")
    if df["demand"].isna().any():
        errors.append("demand contains nulls")
    if (df["demand"] < 0).any():
        errors.append("demand contains negative values")
    result = {"passed": not errors, "errors": errors, "row_count": int(len(df)), "sku_count": int(df["sku"].nunique())}
    if errors:
        raise ValueError("; ".join(errors))
    return result
