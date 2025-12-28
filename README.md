# ⛽ Fuel Price Optimization — Machine Learning & Business Analytics Project

A complete end-to-end **Fuel Price Optimization System** that predicts demand, simulates pricing scenarios, and recommends the most profitable daily fuel price while respecting **business constraints and competitor movements**.

This project demonstrates **real-world ML engineering**, including:
- data processing & feature engineering  
- demand prediction using ML  
- profit-based price optimization  
- business rule enforcement  
- interactive Streamlit dashboard  
- reporting, history logs & export options  

---

## 🎯 Problem Statement

Fuel stations face a daily pricing challenge:

- Higher price → higher margin but lower demand  
- Lower price → higher demand but lower margin  

The objective is to **maximize daily profit** while:
- remaining competitive in the market  
- avoiding price shocks  
- maintaining safe margins  
- following pricing regulations  

This system predicts demand for different price levels and chooses the **optimal price that gives the highest safe profit**.

---

## 🧠 Solution Overview

Data → Feature Engineering → ML Model → Price Simulation → Business Rules → Recommendation

---


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
├── requirements.txt
├── .gitignore
```

---


---

## 🧩 Feature Engineering Highlights

- Competitor price spread  
- Average competitor price index  
- Lag price & lag demand features  
- Moving-average demand indicators  
- Day-of-week & seasonal signals  
- Smart fallback handling for missing values  

---

## 🤖 Machine Learning Model

- Algorithm → **Random Forest Regressor**  
- Target → **Predicted daily fuel volume**
- Evaluation → **MAE (Hold-out validation)**  
- Model exported to `models/volume_model.pkl`

Prediction feeds into **profit optimization logic**.

---

## 💰 Price Optimization Logic

For each candidate price, the engine:

1️⃣ Predicts expected volume  
2️⃣ Computes profit → `(price − cost) × volume`  
3️⃣ Applies business rules:

- maximum daily price change
- minimum margin safety threshold
- competitor alignment tolerance

4️⃣ Selects **highest-profit safe price**

Returns:

- 🟢 Recommended Price  
- 📦 Expected Volume  
- 💵 Expected Profit  
- ⚠ Risk & Strategy Messages  

---

## 🖥️ Streamlit App Features

✔ Fuel type selection (Petrol / Diesel)  
✔ Business constraints panel  
✔ Competitor price controls  
✔ Risk alerts & pricing insights  
✔ Demand & profit visualization  
✔ Prediction history table  
✔ Export results (CSV / PDF)  
✔ Multilingual UI (English / Hindi / Marathi)

Designed to simulate a **real pricing decision tool** used by fuel retailers.

---

## ▶️ How to Run (CLI Mode)

Install dependencies:

```bash
pip install -r requirements.txt

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

---
🚀 Run the Streamlit Dashboard (UI Mode)

To launch the interactive Fuel Price Optimization app:
```bash
streamlit run app.py
```

This will open the application in your browser, where you can:

select fuel type

enter competitor and cost inputs

apply business constraints

view demand & profit predictions

see strategy insights and risk alerts

export results and history logs

---

## 📝 Example Output

```json
{
  "recommended_price": 101.3,
  "expected_volume": 18224.55,
  "expected_profit": 91833.27
}
```

---
🚀 Possible Enhancements

Add FastAPI service endpoint

Model retraining scheduler / batch processing

Price elasticity modeling

XGBoost comparison

Monitoring & drift checks

---
👤 Author

Dikesh Chavhan
Machine Learning & Data Engineering Enthusiast

🔗 LinkedIn — https://www.linkedin.com/in/dikeshchavhan18
---
