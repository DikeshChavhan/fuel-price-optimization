import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


FEATURE_COLS = [
    "price", "cost",
    "comp1_price", "comp2_price", "comp3_price",
    "avg_comp_price", "price_spread_vs_comp",
    "lag_price_1", "lag_volume_1",
    "ma_volume_7", "ma_volume_14",
    "dayofweek", "month"
]


def train_model(processed_path: str, model_path: str):
    df = pd.read_parquet(processed_path)

    X = df[FEATURE_COLS]
    y = df["volume"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    print(f"Model trained. MAE = {mae:.2f}")
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    train_model(
        "data/processed_dataset.parquet",
        "models/volume_model.pkl"
    )
