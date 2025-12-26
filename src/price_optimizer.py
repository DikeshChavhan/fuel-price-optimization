import json
import numpy as np
import joblib
import pandas as pd


FEATURE_COLS = [
    "price", "cost",
    "comp1_price", "comp2_price", "comp3_price",
    "avg_comp_price", "price_spread_vs_comp",
    "lag_price_1", "lag_volume_1",
    "ma_volume_7", "ma_volume_14",
    "dayofweek", "month"
]


def apply_business_rules(candidate_prices, base_price, avg_comp_price):
    """
    Applies realistic pricing constraints.
    """

    # Rule 1: limit price move to ±1.5 INR
    filtered = [p for p in candidate_prices if abs(p - base_price) <= 1.5]

    # Rule 2: not more than 1 INR above competitors
    filtered = [p for p in filtered if p <= avg_comp_price + 1.0]

    # if everything got filtered out -> fall back to original grid
    if len(filtered) == 0:
        return list(candidate_prices)

    return filtered


def recommend_price(today_json: str, model_path: str):
    with open(today_json, "r") as f:
        today = json.load(f)

    model = joblib.load(model_path)

    row = pd.DataFrame([today])

    # --- feature engineering consistent with training ---
    row["avg_comp_price"] = row[["comp1_price", "comp2_price", "comp3_price"]].mean(axis=1)
    row["price_spread_vs_comp"] = row["price"] - row["avg_comp_price"]

    # lag + moving averages fallback assumptions
    row["lag_price_1"] = row["price"]
    row["lag_volume_1"] = today.get("est_volume_yesterday", 15000)
    row["ma_volume_7"] = row["lag_volume_1"]
    row["ma_volume_14"] = row["lag_volume_1"]

    row["dayofweek"] = pd.to_datetime(row["date"]).dt.dayofweek
    row["month"] = pd.to_datetime(row["date"]).dt.month

    base_price = float(row["price"].iloc[0])
    avg_comp_price = float(row["avg_comp_price"].iloc[0])
    cost_today = float(row["cost"].iloc[0])

    # ---- candidate price grid ----
    candidates = np.round(np.arange(base_price - 2, base_price + 2.01, 0.1), 2)
    candidates = apply_business_rules(candidates, base_price, avg_comp_price)

    # tracking variables instead of dict to avoid KeyError
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


if __name__ == "__main__":
    result = recommend_price(
        "data/today_example.json",
        "models/volume_model.pkl"
    )

    print("\nFinal Recommendation\n--------------------")
    print(result)
