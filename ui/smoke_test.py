import joblib, pandas as pd
from pathlib import Path

MODEL_PATH = Path("models") / "sleep_quality_model.joblib"

def main():
    model = joblib.load(MODEL_PATH)
    sample = pd.DataFrame([{
        "age": 25,
        "gender": "Male",
        "bmi_category": "Normal",
        "blood_pressure": "117/76",
        "heart_rate": 72,
        "daily_steps": 8000,
        "sleep_duration": 7.5,
        "physical_activity_level": 60,
        "stress_level": 3,
        "occupation": "Engineer",
        "sleep_disorder": "None",
        "sleep_disorder_missing": 1  # 1 because sleep_disorder is "None"
    }])
    prob = float(model.predict_proba(sample)[0,1])
    label = "Good" if prob >= 0.5 else "Poor"
    print({"sleep_score": round(prob*100,1), "prob_good": prob, "predicted_label": label})

if __name__ == "__main__":
    main()
