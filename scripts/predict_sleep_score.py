import json, argparse, joblib, pandas as pd, pathlib, sys

def predict_from_json(input_json, model_path=None, threshold=0.5):
    # Ensure custom transformers are importable when unpickling
    scripts_dir = pathlib.Path(__file__).parent
    if str(scripts_dir) not in sys.path:
        sys.path.append(str(scripts_dir))
    # default to the same path the UI expects
    if model_path is None:
        pref = pathlib.Path("models/model_augmented_latest.joblib")
        model_path = pref if pref.exists() else pathlib.Path("models/sleep_quality_model.joblib")
    model = joblib.load(model_path)
    row = dict(input_json)
    # Pre-split blood pressure
    bp = str(row.get("blood_pressure", ""))
    import re
    m = re.match(r"^(\d{2,3})/(\d{2,3})$", bp)
    if m:
        row["bp_sys"] = float(m.group(1))
        row["bp_dia"] = float(m.group(2))
    data = pd.DataFrame([row])
    
    if "sleep_disorder_missing" not in data.columns:
        v = data.get("sleep_disorder", [None])[0]
        is_missing = (v is None) or (isinstance(v, str) and str(v).strip() == "")
        data["sleep_disorder_missing"] = 1 if is_missing else 0
    
    y_prob = model.predict_proba(data)[0, 1]
    y_pred = int(y_prob >= threshold)
    sleep_score = round(y_prob * 100, 1)
    return { "sleep_score": sleep_score, "predicted_label": "Good" if y_pred == 1 else "Poor"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", required=True, help="Path to JSON file with input data")
    ap.add_argument("--model_path", default="models/model_augmented_latest.joblib")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    with open(args.input_json, "r") as f: input_data = json.load(f)
    output = predict_from_json(input_data, model_path=args.model_path, threshold=args.threshold)
    print("Prediction:", output)
