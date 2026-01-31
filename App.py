import streamlit as st
import joblib
import pandas as pd
import os

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction App")
st.write("Predict 10-Year Risk of Heart Disease using Machine Learning")

# --------------------------------------------------
# Load Model Safely
# --------------------------------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "desics.pkl")
    return joblib.load(model_path)

model = load_model()

# --------------------------------------------------
# Input Section
# --------------------------------------------------
st.subheader("📥 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    male = st.selectbox("Gender (0 = Female, 1 = Male)", [0, 1])
    age = st.number_input("Age", min_value=1, max_value=120, value=40)
    currentSmoker = st.selectbox("Current Smoker?", [0, 1])
    cigsPerDay = st.number_input("Cigarettes Per Day", min_value=0, max_value=100, value=0)
    BPMeds = st.selectbox("BP Medication?", [0, 1])
    prevalentStroke = st.selectbox("Previous Stroke?", [0, 1])
    prevalentHyp = st.selectbox("Hypertension?", [0, 1])

with col2:
    diabetes = st.selectbox("Diabetes?", [0, 1])
    totChol = st.number_input("Total Cholesterol", min_value=100, max_value=500, value=200)
    sysBP = st.number_input("Systolic BP", min_value=80, max_value=250, value=120)
    diaBP = st.number_input("Diastolic BP", min_value=50, max_value=150, value=80)
    BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0)
    heartRate = st.number_input("Heart Rate", min_value=40, max_value=200, value=75)
    glucose = st.number_input("Glucose Level", min_value=50, max_value=400, value=100)

# --------------------------------------------------
# Create Input DataFrame (MODEL STYLE)
# --------------------------------------------------
input_data = pd.DataFrame([{
    "male": male,
    "age": age,
    "currentSmoker": currentSmoker,
    "cigsPerDay": cigsPerDay,
    "BPMeds": BPMeds,
    "prevalentStroke": prevalentStroke,
    "prevalentHyp": prevalentHyp,
    "diabetes": diabetes,
    "totChol": totChol,
    "sysBP": sysBP,
    "diaBP": diaBP,
    "BMI": BMI,
    "heartRate": heartRate,
    "glucose": glucose
}])

# Safety: ensure numeric + no NaN
input_data = input_data.astype(float)
input_data.fillna(0, inplace=True)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🚀 Predict Risk"):
    try:
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        if prediction[0] == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

        st.write(f"📊 Risk Probability: **{probability[0][1]*100:.2f}%**")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Heart Disease ML Predictor")