import streamlit as st
from joblib import load
import pandas as pd
import os

# =========================
# Page Config (MUST be first Streamlit command)
# =========================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# =========================
# Load Model Safely
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "desics.pkl")

@st.cache_resource
def load_model():
    return load(MODEL_PATH)

model = load_model()

# =========================
# App Title
# =========================
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict 10-year heart disease risk")

# =========================
# Input Section
# =========================
st.subheader("📥 Patient Details")

col1, col2 = st.columns(2)

with col1:
    male = st.selectbox("Gender (0 = Female, 1 = Male)", [0, 1])
    age = st.number_input("Age", min_value=1, max_value=120, value=40)
    currentSmoker = st.selectbox("Current Smoker (0 = No, 1 = Yes)", [0, 1])
    cigsPerDay = st.number_input("Cigarettes Per Day", min_value=0, max_value=100, value=0)
    BPMeds = st.selectbox("BP Medication (0 = No, 1 = Yes)", [0, 1])
    prevalentStroke = st.selectbox("Previous Stroke (0 = No, 1 = Yes)", [0, 1])
    prevalentHyp = st.selectbox("Hypertension (0 = No, 1 = Yes)", [0, 1])

with col2:
    diabetes = st.selectbox("Diabetes (0 = No, 1 = Yes)", [0, 1])
    totChol = st.number_input("Total Cholesterol", min_value=100, max_value=500, value=200)
    sysBP = st.number_input("Systolic BP", min_value=80, max_value=250, value=120)
    diaBP = st.number_input("Diastolic BP", min_value=50, max_value=150, value=80)
    BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0)
    heartRate = st.number_input("Heart Rate", min_value=40, max_value=200, value=75)
    glucose = st.number_input("Glucose Level", min_value=50, max_value=400, value=100)

# =========================
# Create Input DataFrame
# =========================
input_data = pd.DataFrame(
    [[
        male, age, currentSmoker, cigsPerDay, BPMeds,
        prevalentStroke, prevalentHyp, diabetes, totChol,
        sysBP, diaBP, BMI, heartRate, glucose
    ]],
    columns=[
        "male", "age", "currentSmoker", "cigsPerDay", "BPMeds",
        "prevalentStroke", "prevalentHyp", "diabetes", "totChol",
        "sysBP", "diaBP", "BMI", "heartRate", "glucose"
    ]
)

input_data = input_data.astype(float)

# =========================
# Prediction
# =========================
if st.button("🚀 Predict Risk"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.write(f"📊 Risk Probability: **{probability[0][1] * 100:.2f}%**")

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown("Heart Disease ML Predictor")