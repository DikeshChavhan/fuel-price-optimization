import pandas as pd

def read_csv_data(path: str) -> pd.DataFrame:
    """
    Reads CSV data and ensures date is parsed and sorted.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def validate_columns(df: pd.DataFrame, required_cols: list):
    """
    Ensures mandatory columns exist in the dataset.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
