
import time
import io
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from src.price_optimizer import recommend_price
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import plotly.express as px


# ================== LANGUAGE PACK ==================
LANG = {
    "English": {
        "title": "Fuel Price Optimization Dashboard",
        "about": "The model predicts demand at different price levels and recommends the price that maximizes profit while staying within business rules.",
        "fuel_type": "Select Fuel Type",
        "market_inputs": "Market Inputs",
        "competitor_price": "Competitor Price",
        "last_price": "Last Selling Price",
        "purchase_cost": "Purchase Cost",
        "demand_index": "Demand Index",
        "constraints": "Business Constraints",
        "max_up": "Max Price Increase (₹)",
        "max_down": "Max Price Decrease (₹)",
        "min_margin": "Minimum Margin Safety (₹)",
        "gap_limit": "Max Gap Above Competitors (₹)",
        "stock": "Minimum Tank Stock (Litres)",
        "boost": "Weekend / Festival Demand Boost",
        "run": "Recommend Best Price",
        "safe": "Price is safe and within business rules",
        "history": "Prediction History",
        "no_history": "No predictions made yet.",
        "download_pdf": "Download PDF Report",
        "export_csv": "Export History as CSV"
    },

    "हिंदी": {
        "title": "ईंधन मूल्य अनुकूलन डैशबोर्ड",
        "about": "मॉडल विभिन्न कीमतों पर मांग का अनुमान लगाता है और वह कीमत सुझाता है जो लाभ को अधिकतम करती है और व्यापारिक नियमों के अंदर रहती है।",
        "fuel_type": "ईंधन प्रकार चुनें",
        "market_inputs": "बाज़ार इनपुट",
        "competitor_price": "प्रतिद्वंदी मूल्य",
        "last_price": "पिछला विक्रय मूल्य",
        "purchase_cost": "खरीद कीमत",
        "demand_index": "डिमांड इंडेक्स",
        "constraints": "व्यावसायिक नियम",
        "max_up": "अधिकतम मूल्य वृद्धि (₹)",
        "max_down": "अधिकतम मूल्य घटाव (₹)",
        "min_margin": "न्यूनतम मार्जिन सुरक्षा (₹)",
        "gap_limit": "प्रतिद्वंदी से अधिकतम अंतर (₹)",
        "stock": "न्यूनतम टैंक स्टॉक (लीटर)",
        "boost": "वीकेंड / त्योहार मांग वृद्धि",
        "run": "सर्वश्रेष्ठ मूल्य सुझाएँ",
        "safe": "मूल्य सुरक्षित है और नियमों के अनुरूप है",
        "history": "पूर्वानुमान इतिहास",
        "no_history": "अभी तक कोई भविष्यवाणी नहीं की गई।",
        "download_pdf": "पीडीएफ रिपोर्ट डाउनलोड करें",
        "export_csv": "इतिहास CSV के रूप में डाउनलोड करें"
    },

    "मराठी": {
        "title": "इंधन किंमत ऑप्टिमायझेशन डॅशबोर्ड",
        "about": "मॉडेल वेगवेगळ्या किंमतींवर मागणीचा अंदाज लावते आणि नफा जास्तीत जास्त होईल अशी किंमत सुचवते.",
        "fuel_type": "इंधन प्रकार निवडा",
        "market_inputs": "बाजार माहिती",
        "competitor_price": "स्पर्धकाची किंमत",
        "last_price": "मागील विक्री किंमत",
        "purchase_cost": "खरेदी खर्च",
        "demand_index": "मागणी निर्देशांक",
        "constraints": "व्यवसायिक अटी",
        "max_up": "कमाल किंमत वाढ (₹)",
        "max_down": "कमाल किंमत घट (₹)",
        "min_margin": "किमान नफा मर्यादा (₹)",
        "gap_limit": "स्पर्धकांपेक्षा कमाल अंतर (₹)",
        "stock": "किमान टँक साठा (लिटर)",
        "boost": "सुट्टी / सण मागणी वाढ",
        "run": "सर्वोत्तम किंमत सुचवा",
        "safe": "किंमत सुरक्षित आणि नियमांप्रमाणे आहे",
        "history": "भविष्यवाणी इतिहास",
        "no_history": "अजून कोणतीही भविष्यवाणी नाही.",
        "download_pdf": "पीडीएफ अहवाल डाउनलोड करा",
        "export_csv": "इतिहास CSV म्हणून जतन करा"
    }
}


# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Fuel Price Optimization", page_icon="⛽", layout="centered")

if "history" not in st.session_state:
    st.session_state.history = []


# ================== LANGUAGE SELECTOR ==================
lang_choice = st.selectbox("🌐 Language / भाषा / भाषा निवडा",
                           ["English", "हिंदी", "मराठी"])
T = LANG[lang_choice]


# ================== HEADER ==================
st.markdown(f"<h2>⛽ {T['title']}</h2>", unsafe_allow_html=True)
st.write(T["about"])


# ================== FUEL TYPE ==================
fuel_type = st.selectbox(T["fuel_type"], ["Petrol", "Diesel", "CNG", "Premium (XP95)"])

fuel_config = {
    "Petrol": {"default_cost": 84.5},
    "Diesel": {"default_cost": 78.2},
    "CNG": {"default_cost": 67.3},
    "Premium (XP95)": {"default_cost": 89.6},
}
base_cost = fuel_config[fuel_type]["default_cost"]


# ================== INPUTS ==================
st.subheader(T["market_inputs"])

c1, c2 = st.columns(2)

last_price = c1.number_input(T["last_price"], value=95.50, step=0.1)
comp_price = c2.number_input(T["competitor_price"], value=96.00, step=0.1)
cost = c1.number_input(T["purchase_cost"], value=base_cost, step=0.1)
demand_index = c2.slider(T["demand_index"], 0.2, 1.2, value=0.75)


# ================== BUSINESS RULES ==================
st.subheader(T["constraints"])

colA, colB, colC = st.columns(3)
max_up = colA.number_input(T["max_up"], value=1.5, step=0.1)
max_down = colB.number_input(T["max_down"], value=1.5, step=0.1)
min_margin = colC.number_input(T["min_margin"], value=3.0, step=0.1)

allow_gap = colA.number_input(T["gap_limit"], value=1.0, step=0.1)
min_stock = colB.number_input(T["stock"], value=2000)
boost = colC.checkbox(T["boost"])


# ================== RUN MODEL ==================
df_today = pd.DataFrame([{
    "fuel_type": fuel_type,
    "price": last_price,
    "cost": cost,
    "comp1_price": comp_price,
    "comp2_price": comp_price,
    "comp3_price": comp_price,
    "demand_index": demand_index
}])

run = st.button(f"🚀 {T['run']}")

if run:
    result = recommend_price(df_today, "models/volume_model.pkl")

    st.success(f"✔ {T['safe']}")

    rec = result["recommended_price"]

    st.metric("💰 Price", rec)
    st.metric("📦 Volume", result["expected_volume"])
    st.metric("🏦 Profit", result["expected_profit"])

    # save history
    st.session_state.history.append({
        "date": datetime.now().strftime("%d-%m-%Y %H:%M"),
        "fuel_type": fuel_type,
        "price": rec
    })


# ================== HISTORY ==================
st.subheader(f"🗂 {T['history']}")

if len(st.session_state.history):
    dfh = pd.DataFrame(st.session_state.history)
    st.dataframe(dfh)
    st.download_button(T["export_csv"], dfh.to_csv(index=False), "history.csv")
else:
    st.info(T["no_history"])
