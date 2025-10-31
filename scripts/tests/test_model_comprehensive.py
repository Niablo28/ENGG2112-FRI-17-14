"""Edge cases for the sleep-quality predictor."""
import json
import joblib
import pandas as pd
import sys
from pathlib import Path

# Allow pipeline helpers to import at runtime
scripts_dir = Path(__file__).parent.parent
if str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))

from shared_transforms import _split_bp

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "sleep_quality_model.joblib"
THRESHOLD = 0.5

def load_model():
    """Load the trained model"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def predict(model, payload: dict):
    """Make prediction with proper preprocessing"""
    payload_with_missing = payload.copy()
    payload_with_missing["sleep_disorder_missing"] = 1 if payload["sleep_disorder"] == "None" else 0
    df = pd.DataFrame([payload_with_missing])
    df = _split_bp(df)
    prob_good = float(model.predict_proba(df)[0, 1])
    label = "Good" if prob_good >= THRESHOLD else "Poor"
    score = round(prob_good * 100.0, 1)
    return {"sleep_score": score, "prob_good": prob_good, "predicted_label": label}

# Test scenarios
test_cases = [
    {
        "name": "Test 1: Perfect Health Profile",
        "description": "Ideal sleep conditions - should score highest",
        "data": {
            "age": 30,
            "gender": "Male",
            "occupation": "Accountant",
            "bmi_category": "Normal",
            "blood_pressure": "120/80",
            "heart_rate": 65,
            "daily_steps": 10000,
            "sleep_duration": 8.0,
            "physical_activity_level": 80,
            "stress_level": 2,
            "sleep_disorder": "None"
        },
        "expected": {"label": "Good", "min_score": 80}
    },
    {
        "name": "Test 2: Poor Sleep Profile",
        "description": "Multiple risk factors - should score low",
        "data": {
            "age": 45,
            "gender": "Female",
            "occupation": "Nurse",
            "bmi_category": "Overweight",
            "blood_pressure": "140/95",
            "heart_rate": 90,
            "daily_steps": 3000,
            "sleep_duration": 5.0,
            "physical_activity_level": 30,
            "stress_level": 8,
            "sleep_disorder": "Insomnia"
        },
        "expected": {"label": "Poor", "max_score": 50}
    },
    {
        "name": "Test 3: Minimal Sleep (4 hours)",
        "description": "Critical sleep deprivation - should flag",
        "data": {
            "age": 25,
            "gender": "Male",
            "occupation": "Software Engineer",
            "bmi_category": "Normal",
            "blood_pressure": "125/80",
            "heart_rate": 75,
            "daily_steps": 5000,
            "sleep_duration": 4.0,
            "physical_activity_level": 40,
            "stress_level": 7,
            "sleep_disorder": "None"
        },
        "expected": {"label": "Poor", "max_score": 50}
    },
    {
        "name": "Test 4: Excessive Sleep (10 hours)",
        "description": "Oversleeping - model should detect as non-optimal",
        "data": {
            "age": 28,
            "gender": "Female",
            "occupation": "Teacher",
            "bmi_category": "Normal",
            "blood_pressure": "120/80",
            "heart_rate": 70,
            "daily_steps": 8000,
            "sleep_duration": 10.0,
            "physical_activity_level": 60,
            "stress_level": 3,
            "sleep_disorder": "None"
        },
        "expected": {"label": "Good", "min_score": 60}
    },
    {
        "name": "Test 5: Very Low Stress (0/10)",
        "description": "Minimal stress - should boost score",
        "data": {
            "age": 35,
            "gender": "Male",
            "occupation": "Engineer",
            "bmi_category": "Normal",
            "blood_pressure": "117/76",
            "heart_rate": 65,
            "daily_steps": 9000,
            "sleep_duration": 7.5,
            "physical_activity_level": 70,
            "stress_level": 0,
            "sleep_disorder": "None"
        },
        "expected": {"label": "Good", "min_score": 80}
    },
    {
        "name": "Test 6: Very High Stress (10/10)",
        "description": "Maximum stress - should lower score significantly",
        "data": {
            "age": 40,
            "gender": "Female",
            "occupation": "Salesperson",
            "bmi_category": "Normal",
            "blood_pressure": "130/85",
            "heart_rate": 85,
            "daily_steps": 6000,
            "sleep_duration": 6.0,
            "physical_activity_level": 35,
            "stress_level": 10,
            "sleep_disorder": "None"
        },
        "expected": {"label": "Poor", "max_score": 60}
    },
    {
        "name": "Test 7: Normal Moderate Profile",
        "description": "Average healthy person - should score moderate",
        "data": {
            "age": 40,
            "gender": "Male",
            "occupation": "Doctor",
            "bmi_category": "Normal",
            "blood_pressure": "125/80",
            "heart_rate": 72,
            "daily_steps": 7000,
            "sleep_duration": 7.0,
            "physical_activity_level": 60,
            "stress_level": 5,
            "sleep_disorder": "None"
        },
        "expected": {"label": "Good", "min_score": 60}
    },
    {
        "name": "Test 8: Borderline Score",
        "description": "Near threshold (65-75 range) - test binary classification",
        "data": {
            "age": 38,
            "gender": "Female",
            "occupation": "Scientist",
            "bmi_category": "Normal",
            "blood_pressure": "120/80",
            "heart_rate": 70,
            "daily_steps": 7500,
            "sleep_duration": 6.5,
            "physical_activity_level": 55,
            "stress_level": 6,
            "sleep_disorder": "None"
        },
        "expected": {"label": None, "score_range": [60, 75]}
    },
    {
        "name": "Test 9: High Activity, Low Sleep",
        "description": "Active lifestyle but insufficient sleep",
        "data": {
            "age": 32,
            "gender": "Male",
            "occupation": "Lawyer",
            "bmi_category": "Normal",
            "blood_pressure": "122/78",
            "heart_rate": 68,
            "daily_steps": 15000,
            "sleep_duration": 5.5,
            "physical_activity_level": 85,
            "stress_level": 7,
            "sleep_disorder": "None"
        },
        "expected": {"label": "Poor", "max_score": 65}
    },
    {
        "name": "Test 10: Sleep Apnea Disorder",
        "description": "Has sleep disorder - should impact score",
        "data": {
            "age": 50,
            "gender": "Male",
            "occupation": "Other",
            "bmi_category": "Overweight",
            "blood_pressure": "135/90",
            "heart_rate": 80,
            "daily_steps": 5000,
            "sleep_duration": 7.0,
            "physical_activity_level": 40,
            "stress_level": 4,
            "sleep_disorder": "Sleep Apnea"
        },
        "expected": {"label": "Poor", "max_score": 65}
    }
]

def print_analysis(result, test_case):
    """Print detailed analysis of prediction"""
    print(f"\n{'='*80}")
    print(f"{test_case['name']}")
    print(f"{'-'*80}")
    print(f"Description: {test_case['description']}")
    print(f"\nInput Values:")
    for key, value in test_case['data'].items():
        print(f"  {key}: {value}")
    
    print(f"\nPrediction Results:")
    print(f"  Sleep Score: {result['sleep_score']}/100")
    print(f"  Label: {result['predicted_label']}")
    print(f"  Confidence (P(Good)): {result['prob_good']:.2%}")
    
    # Checks
    expected = test_case['expected']
    issues = []
    
    if expected.get('label'):
        if result['predicted_label'] != expected['label']:
            issues.append(f"[X] Label mismatch: Expected '{expected['label']}', got '{result['predicted_label']}'")
        else:
            print(f"  [OK] Label correct: {result['predicted_label']}")
    
    if expected.get('min_score'):
        if result['sleep_score'] < expected['min_score']:
            issues.append(f"[X] Score too low: Expected >= {expected['min_score']}, got {result['sleep_score']}")
        else:
            print(f"  [OK] Score meets minimum: {result['sleep_score']} >= {expected['min_score']}")
    
    if expected.get('max_score'):
        if result['sleep_score'] > expected['max_score']:
            issues.append(f"[X] Score too high: Expected <= {expected['max_score']}, got {result['sleep_score']}")
        else:
            print(f"  [OK] Score within maximum: {result['sleep_score']} <= {expected['max_score']}")
    
    if expected.get('score_range'):
        if not (expected['score_range'][0] <= result['sleep_score'] <= expected['score_range'][1]):
            issues.append(f"[X] Score outside range: Expected {expected['score_range']}, got {result['sleep_score']}")
        else:
            print(f"  [OK] Score in expected range: {result['sleep_score']} in {expected['score_range']}")
    
    if issues:
        print(f"\n[!] Issues Found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"\n[OK] All expectations met!")
    
    return issues

def main():
    print("\n" + "="*80)
    print("SLEEP QUALITY PREDICTOR - COMPREHENSIVE MODEL TESTING")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    model = load_model()
    print(f"[OK] Model loaded from: {MODEL_PATH}")
    print(f"   Model type: {type(model).__name__}")
    
    # Run tests
    all_issues = []
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n\n[TEST] Running {test_case['name']}...")
        result = predict(model, test_case['data'])
        issues = print_analysis(result, test_case)
        
        if issues:
            failed += 1
            all_issues.extend(issues)
        else:
            passed += 1
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total Tests: {len(test_cases)}")
    print(f"[OK] Passed: {passed}")
    print(f"[X] Failed: {failed}")
    print(f"Success Rate: {passed/len(test_cases)*100:.1f}%")
    
    if all_issues:
        print(f"\n[!] Issues Found:")
        for issue in all_issues:
            print(f"  {issue}")
    else:
        print("\n[SUCCESS] All tests passed!")
    
    # Model insights
    print("\n" + "="*80)
    print("MODEL INSIGHTS")
    print("="*80)
    
    # Feature importance
    if hasattr(model, 'named_steps'):
        if 'preprocessor' in model.named_steps and 'classifier' in model.named_steps:
            try:
                feature_names = model.named_steps['preprocessor'].get_feature_names_out()
                if hasattr(model.named_steps['classifier'], 'coef_'):
                    coefs = model.named_steps['classifier'].coef_[0]
                    
                    # Top 10 features
                    top_features = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)[:10]
                    
                    print("\nTop 10 Most Important Features:")
                    for name, coef in top_features:
                        direction = "↑ Increases" if coef > 0 else "↓ Decreases"
                        print(f"  {direction:20s} sleep quality | Coefficient: {coef:+.3f} | Feature: {name}")
            except Exception as e:
                print(f"Could not extract feature importance: {e}")

if __name__ == "__main__":
    main()

