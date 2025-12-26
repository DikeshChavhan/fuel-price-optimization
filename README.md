# Fuel Price Optimization — ML Engineering Assignment

This project implements a small end-to-end ML pipeline to support daily retail
fuel price recommendation using historical demand and competitor price signals.

The system reads historical data, builds engineered features, trains a
Random Forest regression model to estimate expected volume, and then evaluates
candidate price options to select the price that maximizes daily profit while
respecting practical business guardrails.

### How to Run

python src/pipeline.py
python src/train_model.py
python src/price_optimizer.py


### Output Example

{
"recommended_price": 101.2,
"expected_volume": 18342.4,
"expected_profit": 92113.7
}


The approach is intentionally simple and extensible so it can be adapted to a
production data pipeline or deployed as an API service in future iterations.
