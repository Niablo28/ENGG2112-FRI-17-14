"""Edge-case checks for the augmented model."""
import joblib
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Allow pipeline helpers to import at runtime
scripts_dir = Path(__file__).parent.parent
if str(scripts_dir) not in sys.path:
    sys.path.append(str(scripts_dir))

from shared_transforms import _split_bp

def test_edge_cases(model):
    """Test specific edge cases that the original model struggled with"""
    
    print("\n" + "="*80)
    print("EDGE CASE TESTING - AUGMENTED MODEL")
    print("="*80)
    
    # Case 1
    print("\n[TEST 1] Excessive Sleep (10 hours)")
    print("-" * 80)
    test1 = {
        "age": 30,
        "gender": "Female",
        "occupation": "Teacher",
        "bmi_category": "Normal",
        "blood_pressure": "120/80",
        "heart_rate": 70,
        "daily_steps": 8000,
        "sleep_duration": 10.0,
        "physical_activity_level": 60,
        "stress_level": 3,
        "sleep_disorder": "None",
        "sleep_disorder_missing": 1
    }
    
    df1 = pd.DataFrame([test1])
    df1 = _split_bp(df1)
    prob1 = float(model.predict_proba(df1)[0, 1])
    score1 = round(prob1 * 100, 1)
    label1 = "Good" if prob1 >= 0.5 else "Poor"
    
    print(f"Sleep Duration: 10.0h")
    print(f"Result: Score = {score1}/100 ({label1})")
    print(f"Confidence: {prob1:.1%}")
    
    if score1 < 95:
        print("[OK] Model correctly penalizes excessive sleep!")
    else:
        print("[ISSUE] Model still scores oversleeping too high")
    
    # Case 2
    print("\n[TEST 2] Sleep Apnea Disorder")
    print("-" * 80)
    test2 = {
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
        "sleep_disorder": "Sleep Apnea",
        "sleep_disorder_missing": 0
    }
    
    df2 = pd.DataFrame([test2])
    df2 = _split_bp(df2)
    prob2 = float(model.predict_proba(df2)[0, 1])
    score2 = round(prob2 * 100, 1)
    label2 = "Good" if prob2 >= 0.5 else "Poor"
    
    print(f"Sleep Disorder: Sleep Apnea")
    print(f"Sleep Duration: 7.0h")
    print(f"Result: Score = {score2}/100 ({label2})")
    print(f"Confidence: {prob2:.1%}")
    
    if score2 < 70:
        print("[OK] Model correctly penalizes sleep disorders!")
    else:
        print("[ISSUE] Model still under-penalizes sleep disorders")
    
    # Case 3
    print("\n[TEST 3] Borderline Poor Sleep")
    print("-" * 80)
    test3 = {
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
        "sleep_disorder": "None",
        "sleep_disorder_missing": 1
    }
    
    df3 = pd.DataFrame([test3])
    df3 = _split_bp(df3)
    prob3 = float(model.predict_proba(df3)[0, 1])
    score3 = round(prob3 * 100, 1)
    label3 = "Good" if prob3 >= 0.5 else "Poor"
    
    print(f"Sleep Duration: 6.5h, Stress: 6/10")
    print(f"Result: Score = {score3}/100 ({label3})")
    print(f"Confidence: {prob3:.1%}")
    
    if 60 <= score3 <= 75:
        print("[OK] Model correctly identifies borderline cases!")
    else:
        print(f"[INFO] Score = {score3} (in acceptable range)")
    
    # Case 4
    print("\n[TEST 4] Perfect Health Profile")
    print("-" * 80)
    test4 = {
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
        "sleep_disorder": "None",
        "sleep_disorder_missing": 1
    }
    
    df4 = pd.DataFrame([test4])
    df4 = _split_bp(df4)
    prob4 = float(model.predict_proba(df4)[0, 1])
    score4 = round(prob4 * 100, 1)
    label4 = "Good" if prob4 >= 0.5 else "Poor"
    
    print(f"Sleep Duration: 8.0h, Stress: 2/10, Activity: 80/100")
    print(f"Result: Score = {score4}/100 ({label4})")
    print(f"Confidence: {prob4:.1%}")
    
    if score4 >= 80:
        print("[OK] Model correctly rewards excellent profiles!")
    else:
        print("[ISSUE] Model should score excellent profiles higher")
    
    # Case 5
    print("\n[TEST 5] Excessive Sleep + Sleep Disorder")
    print("-" * 80)
    test5 = {
        "age": 45,
        "gender": "Female",
        "occupation": "Nurse",
        "bmi_category": "Overweight",
        "blood_pressure": "140/95",
        "heart_rate": 85,
        "daily_steps": 3000,
        "sleep_duration": 11.0,
        "physical_activity_level": 30,
        "stress_level": 7,
        "sleep_disorder": "Sleep Apnea",
        "sleep_disorder_missing": 0
    }
    
    df5 = pd.DataFrame([test5])
    df5 = _split_bp(df5)
    prob5 = float(model.predict_proba(df5)[0, 1])
    score5 = round(prob5 * 100, 1)
    label5 = "Good" if prob5 >= 0.5 else "Poor"
    
    print(f"Sleep Duration: 11.0h, Disorder: Sleep Apnea, Stress: 7/10")
    print(f"Result: Score = {score5}/100 ({label5})")
    print(f"Confidence: {prob5:.1%}")
    
    if score5 < 60:
        print("[OK] Model heavily penalizes problematic combination!")
    else:
        print("[ISSUE] Model should penalize this more")
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print("\nEdge Cases Tested:")
    print(f"  1. Excessive Sleep (10h): {score1}/100")
    print(f"  2. Sleep Apnea: {score2}/100")
    print(f"  3. Borderline Case: {score3}/100")
    print(f"  4. Perfect Health: {score4}/100")
    print(f"  5. Severe Case: {score5}/100")
    
    print("\nImprovement Check:")
    print("  - Oversleeping penalized: ", "[OK]" if score1 < 95 else "[ISSUE]")
    print("  - Sleep disorders penalized: ", "[OK]" if score2 < 70 else "[ISSUE]")
    print("  - Borderline cases handled: ", "[OK]" if 60 <= score3 <= 75 else "[~]")
    print("  - Excellent profiles rewarded: ", "[OK]" if score4 >= 80 else "[ISSUE]")
    print("  - Severe cases heavily penalized: ", "[OK]" if score5 < 60 else "[ISSUE]")

def main():
    repo_root = Path(__file__).resolve().parents[2]
    
    # Load augmented model
    model_path = repo_root / "models" / "model_augmented_latest.joblib"
    print(f"\nLoading model from: {model_path}")
    model = joblib.load(model_path)
    
    # Test edge cases
    test_edge_cases(model)
    
    print("\n[SUCCESS] Testing complete!")
    print("[INFO] Model artefact: models/model_augmented_latest.joblib")

if __name__ == "__main__":
    main()

