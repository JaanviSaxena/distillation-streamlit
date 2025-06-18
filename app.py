
import streamlit as st
import numpy as np
import joblib

# Load saved models
models = {
    "Ethanol concentration": joblib.load("model_Ethanol_concentration.pkl"),
    "Yield": joblib.load("model_Yield.pkl"),
    "Waste_Generated": joblib.load("model_Waste_Generated.pkl"),
    "Energy_Consumption": joblib.load("model_Energy_Consumption.pkl"),
}

# Streamlit App UI
st.set_page_config(page_title="Distillation Process Predictor", layout="centered")
st.title("🧪 Distillation Process Predictor")

st.markdown("""
Enter the values for the operating parameters below, then select which output to predict.
""")

# Sidebar for output selection
target = st.sidebar.selectbox("Select Output to Predict", list(models.keys()))

# Input fields
with st.form("input_form"):
    F = st.number_input("Feed (F)", value=100.0)
    D = st.number_input("Distillate (D)", value=50.0)
    B = st.number_input("Bottoms (B)", value=50.0)
    L = st.number_input("Reflux (L)", value=30.0)
    V = st.number_input("Vapor (V)", value=30.0)
    T1 = st.number_input("Temperature T1 (°C)", value=80.0)
    T14 = st.number_input("Temperature T14 (°C)", value=120.0)
    submit = st.form_submit_button("Predict")

if submit:
    model = models[target]
    input_data = np.array([[F, D, B, L, V, T1, T14]])
    prediction = model.predict(input_data)[0]
    st.success(f"**Predicted {target}: {prediction:.4f}**")

