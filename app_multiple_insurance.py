import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Multiple Linear Regression - Insurance", layout="centered")

# Load CSS
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# Title
st.markdown("""
<div class="card">
    <h1>Multiple Linear Regression</h1>
    <p>Predict <b>Insurance Charges</b> using multiple health factors</p>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")

df = load_data()

# Encode smoker column
df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})

# Preview
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df[["age", "bmi", "children", "smoker", "charges"]].head())
st.markdown('</div>', unsafe_allow_html=True)

# Prepare data
x = df[["age", "bmi", "children", "smoker"]]
y = df["charges"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train model
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test)-1) / (len(y_test)-x_test.shape[1]-1)

# Performance
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")

c3, c4 = st.columns(2)
c3.metric("R²", f"{r2:.3f}")
c4.metric("Adj R²", f"{adj_r2:.3f}")

st.markdown('</div>', unsafe_allow_html=True)

# Model Interpretation
st.markdown(f"""
<div class="card">
<h3>Model Interpretation</h3>
<p>
<b>Age Coefficient:</b> {model.coef_[0]:.2f}<br>
<b>BMI Coefficient:</b> {model.coef_[1]:.2f}<br>
<b>Children Coefficient:</b> {model.coef_[2]:.2f}<br>
<b>Smoker Coefficient:</b> {model.coef_[3]:.2f}<br>
<b>Intercept:</b> {model.intercept_:.2f}
</p>
</div>
""", unsafe_allow_html=True)

# Prediction
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Insurance Charges")

age = st.slider("Age", 18, 64, 30)
bmi = st.slider("BMI", 15.0, 45.0, 25.0)
children = st.slider("Children", 0, 5, 1)
smoker = st.selectbox("Smoker", ["No", "Yes"])

smoker_val = 1 if smoker == "Yes" else 0

input_scaled = scaler.transform([[age, bmi, children, smoker_val]])
prediction = model.predict(input_scaled)[0]

st.markdown(
    f'<div class="prediction-box">Predicted Charges: ₹ {prediction:,.2f}</div>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)
