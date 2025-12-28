import json
import numpy as np
import joblib
import pandas as pd
from datetime import datetime


FEATURE_COLS = [
    "price", "cost",
    "comp1_price", "comp2_price", "comp3_price",
    "avg_comp_price", "price_spread_vs_comp",
    "lag_price_1", "lag_volume_1",
    "ma_volume_7", "ma_volume_14",
    "dayofweek", "month"
]


def apply_business_rules(candidate_prices, base_price, avg_comp_price):
    """Applies realistic pricing constraints."""

    filtered = [p for p in candidate_prices if abs(p - base_price) <= 1.5]
    filtered = [p for p in filtered if p <= avg_comp_price + 1.0]

    return filtered if len(filtered) > 0 else list(candidate_prices)


def recommend_price(today_input, model_path: str = "models/volume_model.pkl"):
    """
    Supports:
    • JSON path (CLI mode)
    • DataFrame (Streamlit mode)
    """

    # -------- Input Handling --------
    if isinstance(today_input, str):
        with open(today_input, "r") as f:
            today = json.load(f)
        row = pd.DataFrame([today])

    elif isinstance(today_input, pd.DataFrame):
        row = today_input.copy()

    else:
        raise TypeError("today_input must be a JSON filepath or a DataFrame")

    # -------- Load Model --------
    model = joblib.load(model_path)

    # -------- Price Handling --------
    if "price" not in row.columns:
        if "last_price" in row.columns:
            row["price"] = row["last_price"]
        else:
            raise KeyError("Input must contain 'price' or 'last_price'")

    # -------- Competitor Fallback --------
    for col in ["comp1_price", "comp2_price", "comp3_price"]:
        if col not in row.columns:
            row[col] = row.get("competitor_price", row.get("comp_price", 0))

    row["avg_comp_price"] = row[["comp1_price", "comp2_price", "comp3_price"]].mean(axis=1)
    row["price_spread_vs_comp"] = row["price"] - row["avg_comp_price"]

    # -------- Lag & MA --------
    row["lag_price_1"] = row["price"]
    row["lag_volume_1"] = row.get("est_volume_yesterday", 15000)
    row["ma_volume_7"] = row["lag_volume_1"]
    row["ma_volume_14"] = row["lag_volume_1"]

    # -------- Date Handling (Fallback to today) --------
    if "date" not in row.columns:
        row["date"] = datetime.today().strftime("%Y-%m-%d")

    row["dayofweek"] = pd.to_datetime(row["date"]).dt.dayofweek
    row["month"] = pd.to_datetime(row["date"]).dt.month

    base_price = float(row["price"].iloc[0])
    avg_comp_price = float(row["avg_comp_price"].iloc[0])
    cost_today = float(row["cost"].iloc[0])

    # ---- Candidate price grid ----
    candidates = np.round(np.arange(base_price - 2, base_price + 2.01, 0.1), 2)
    candidates = apply_business_rules(candidates, base_price, avg_comp_price)

    best_price = None
    best_profit = -1e18
    best_volume = None

    for p in candidates:
        tmp = row.copy()
        tmp["price"] = p
        tmp["price_spread_vs_comp"] = p - avg_comp_price

        volume_pred = model.predict(tmp[FEATURE_COLS])[0]
        profit = (p - cost_today) * volume_pred

        if profit > best_profit:
            best_profit = profit
            best_price = p
            best_volume = volume_pred

    return {
        "recommended_price": round(best_price, 2),
        "expected_volume": round(best_volume, 2),
        "expected_profit": round(best_profit, 2)
    }


# -------- CLI Mode Test --------
if __name__ == "__main__":
    result = recommend_price(
        "data/today_example.json",
        "models/volume_model.pkl"
    )

    print("\nFinal Recommendation\n--------------------")
    print(result)
