import json, argparse, joblib, pandas as pd, pathlib

def predict_from_json(input_json, model_path="reports/sleep_quality_model.joblib"):
    model = joblib.load(model_path)
    data = pd.DataFrame([input_json])
    
    if "sleep_disorder_missing" not in data.columns:
        data["sleep_disorder_missing"] = 0 if data.get("sleep_disorder", [None])[0] else 1
    
    y_prob = model.predict_proba(data)[0, 1]
    y_pred = int(y_prob >= 0.5)
    sleep_score = round(y_prob * 100, 1)
    return { "sleep_score": sleep_score, "predicted_label": "Good" if y_pred == 1 else "Poor"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True, help="Path to JSON file with input data")
    args = ap.parse_args()
    with open(args.input_json, "r") as f: input_data = json.load(f)
    output = predict_from_json(input_data)
    print("Prediction:", output)
