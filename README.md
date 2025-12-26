# Fuel Price Optimization — ML Engineering Assignment

This project implements an end-to-end Machine Learning pipeline to support
daily retail fuel price recommendation. The goal is to recommend a price that
maximizes daily profit while considering demand behaviour and competitor price
movements.

The solution follows a practical engineering-oriented approach focusing on
pipeline design, feature computation, model training, and a clean price
optimization workflow.

---

## 📁 Project Structure
```
fuel-price-optimization
│
├── data/
│ ├── oil_retail_history.csv # historical dataset
│ ├── today_example.json # daily market input example
│
├── src/
│ ├── pipeline.py # data ingestion + feature engineering
│ ├── train_model.py # model training script
│ ├── price_optimizer.py # price recommendation logic
│ ├── utils.py # helper functions
│
├── models/ # trained models (ignored in git)
├── README.md
├── .gitignore
```

---

## 🧩 Approach Summary

### 🔹 Data Engineering
- Reads historical transaction data  
- Validates and cleans records  
- Generates engineered features:
  - competitor price spread  
  - lag & moving-average demand indicators  
  - basic seasonality features  
- Stores processed data for training  

### 🔹 Machine Learning
- Random Forest Regression model  
- Predicts expected sales volume for a given price  
- Evaluated using hold-out validation (MAE metric)  

### 🔹 Price Optimization Logic
For the current day:

1. Generate candidate price range around last price  
2. Predict expected volume for each price  
3. Compute profit: **(price − cost) × volume**  
4. Apply realistic business guardrails:
   - daily price movement limits  
   - competitor alignment checks  
   - minimum margin protection  
5. Return:
   - recommended price  
   - expected volume  
   - expected profit  

---
## ▶️ How to Run

Install dependencies:

```bash
pip install pandas numpy scikit-learn joblib pyarrow
```

Run pipeline:

```bash
python src/pipeline.py
```

Train model:

```bash
python src/train_model.py
```

Generate price recommendation:

```bash
python src/price_optimizer.py
```

## 📝 Example Output

```json
{
  "recommended_price": 101.3,
  "expected_volume": 18224.55,
  "expected_profit": 91833.27
}
```




🚀 Possible Enhancements

Add FastAPI service endpoint

Model retraining scheduler / batch processing

Price elasticity modeling

XGBoost comparison

Monitoring & drift checks

🙋 Author

Prepared by Dikesh Chavhan
Submitted as part of an ML Engineering hiring assignment.