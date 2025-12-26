import pandas as pd
from pathlib import Path
from utils import read_csv_data, validate_columns


REQUIRED_COLS = [
    "date", "price", "cost",
    "comp1_price", "comp2_price", "comp3_price",
    "volume"
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates engineered features for ML training.
    """
    validate_columns(df, REQUIRED_COLS)

    # competitor price signals
    df["avg_comp_price"] = df[["comp1_price", "comp2_price", "comp3_price"]].mean(axis=1)
    df["price_spread_vs_comp"] = df["price"] - df["avg_comp_price"]

    # lag effects
    df["lag_price_1"] = df["price"].shift(1)
    df["lag_volume_1"] = df["volume"].shift(1)

    # moving averages
    df["ma_volume_7"] = df["volume"].rolling(7).mean()
    df["ma_volume_14"] = df["volume"].rolling(14).mean()

    # seasonality features
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    df = df.dropna().reset_index(drop=True)
    return df


def run_pipeline(input_path: str, output_path: str):
    df = read_csv_data(input_path)
    processed = compute_features(df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(output_path, index=False)

    print(f"Pipeline completed. Saved processed dataset to: {output_path}")


if __name__ == "__main__":
    run_pipeline(
        "data/oil_retail_history.csv",
        "data/processed_dataset.parquet"
    )
