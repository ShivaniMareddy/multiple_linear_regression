import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Multiple Linear Regression - Iris", layout="centered")

# Load CSS
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# Title
st.markdown("""
<div class="card">
    <h1>Multiple Linear Regression (Iris Dataset)</h1>
    <p>Predict <b>Petal Length</b> using <b>Sepal Length</b> and <b>Sepal Width</b></p>
</div>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    return sns.load_dataset("iris")

df = load_data()

# Dataset Preview
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df[["sepal_length", "sepal_width", "petal_length"]].head())
st.markdown('</div>', unsafe_allow_html=True)

# Prepare Data
x = df[["sepal_length", "sepal_width"]]
y = df["petal_length"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train Model
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - x_test.shape[1] - 1)

# Visualization
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Sepal Length vs Petal Length (Multiple Regression)")

fig, ax = plt.subplots()
ax.scatter(df["sepal_length"], df["petal_length"], alpha=0.6)
ax.plot(
    df["sepal_length"],
    model.predict(scaler.transform(x)),
    color="red"
)
ax.set_xlabel("Sepal Length (cm)")
ax.set_ylabel("Petal Length (cm)")
st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

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
        <b>Coefficient (Sepal Length):</b> {model.coef_[0]:.3f}<br>
        <b>Coefficient (Sepal Width):</b> {model.coef_[1]:.3f}<br>
        <b>Intercept:</b> {model.intercept_:.3f}
    </p>
    <p>
        Petal length depends on both <b>sepal length</b> and <b>sepal width</b>.
    </p>
</div>
""", unsafe_allow_html=True)

# Prediction
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Petal Length")

sepal_length = st.slider(
    "Sepal Length (cm)",
    float(df.sepal_length.min()),
    float(df.sepal_length.max()),
    float(df.sepal_length.mean())
)

sepal_width = st.slider(
    "Sepal Width (cm)",
    float(df.sepal_width.min()),
    float(df.sepal_width.max()),
    float(df.sepal_width.mean())
)

input_scaled = scaler.transform([[sepal_length, sepal_width]])
prediction = model.predict(input_scaled)[0]

st.markdown(
    f'<div class="prediction-box">Predicted Petal Length: {prediction:.2f} cm</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
