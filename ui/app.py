import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

MODEL_PATH = Path("reports") / "sleep_quality_model.joblib"
THRESHOLD = 0.5  # operating point

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH.resolve()}")
    return joblib.load(MODEL_PATH)

def predict(payload: dict):
    model = load_model()
    # Add sleep_disorder_missing column (1 if sleep_disorder is missing/None, 0 otherwise)
    payload_with_missing = payload.copy()
    payload_with_missing["sleep_disorder_missing"] = 1 if payload["sleep_disorder"] == "None" else 0
    
    df = pd.DataFrame([payload_with_missing])
    prob_good = float(model.predict_proba(df)[0, 1])
    label = "Good" if prob_good >= THRESHOLD else "Poor"
    score = round(prob_good * 100.0, 1)
    return {"sleep_score": score, "prob_good": prob_good, "predicted_label": label}

st.set_page_config(page_title="Sleep Quality Predictor", page_icon="😴", layout="centered")
st.title("😴 Sleep Quality Predictor")
st.caption(f"Using model: `{MODEL_PATH}` · Threshold={THRESHOLD}")

with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=25, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        bmi_category = st.selectbox("BMI Category", ["Underweight","Normal","Overweight","Obese"])
        blood_pressure = st.selectbox("Blood Pressure", ["117/76","120/80","130/85","140/90"])
        heart_rate = st.number_input("Resting Heart Rate (bpm)", min_value=40, max_value=120, value=72, step=1)

    with col2:
        daily_steps = st.number_input("Daily Steps", min_value=0, max_value=40000, value=8000, step=100)
        sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=24.0, value=7.5, step=0.1)
        physical_activity_level = st.number_input("Physical Activity Level (0-100)", min_value=0, max_value=100, value=60, step=1)
        stress_level = st.number_input("Stress Level (0-10)", min_value=0, max_value=10, value=3, step=1)
        occupation = st.selectbox("Occupation", [
            "Engineer","Accountant","Salesperson","Lawyer","Nurse","Doctor","Teacher","Student","Other"
        ])
    sleep_disorder = st.selectbox("Sleep Disorder", ["None","Insomnia","Sleep Apnea","Other"])

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "age": int(age),
        "gender": gender,
        "bmi_category": bmi_category,
        "blood_pressure": blood_pressure,
        "heart_rate": int(heart_rate),
        "daily_steps": int(daily_steps),
        "sleep_duration": float(sleep_duration),
        "physical_activity_level": int(physical_activity_level),
        "stress_level": int(stress_level),
        "occupation": occupation,
        "sleep_disorder": sleep_disorder
    }
    try:
        result = predict(payload)
        colA, colB, colC = st.columns(3)
        colA.metric("Sleep Score", f"{result['sleep_score']} / 100")
        colB.metric("Label", result["predicted_label"])
        colC.metric("P(good)", f"{result['prob_good']:.3f}")

        st.progress(min(1.0, result["sleep_score"]/100.0))

        with st.expander("Request payload (JSON)"):
            st.code(json.dumps(payload, indent=2), language="json")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)
