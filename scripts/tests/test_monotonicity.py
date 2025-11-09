"""Sanity checks ensuring monotonic model behaviour for key features."""

from pathlib import Path

import joblib
import pandas as pd
import pytest

from scripts.shared_transforms import _split_bp


MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "sleep_quality_model.joblib"


@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        pytest.skip(f"Model artifact missing at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def _predict_good_prob(model, payload):
    df = pd.DataFrame([payload])
    df = _split_bp(df)
    return float(model.predict_proba(df)[0, 1])


def _baseline_payload(**overrides):
    base = {
        "age": 30,
        "gender": "Male",
        "occupation": "Engineer",
        "bmi_category": "Normal",
        "blood_pressure": "120/80",
        "heart_rate": 72,
        "daily_steps": 8000,
        "sleep_duration": 7.5,
        "physical_activity_level": 60,
        "stress_level": 3,
        "sleep_disorder": "None",
        "sleep_disorder_missing": 1,
    }
    base.update(overrides)
    return base


def test_sleep_duration_monotonic_up_to_optimal(model):
    payload_short = _baseline_payload(sleep_duration=6.0)
    payload_optimal = _baseline_payload(sleep_duration=8.0)

    p_short = _predict_good_prob(model, payload_short)
    p_optimal = _predict_good_prob(model, payload_optimal)

    assert p_optimal >= p_short - 1e-6, "8h sleep should not reduce probability vs 6h"


def test_extreme_oversleep_penalised(model):
    payload_optimal = _baseline_payload(sleep_duration=9.0)
    payload_oversleep = _baseline_payload(sleep_duration=11.0, physical_activity_level=30, stress_level=7)

    p_optimal = _predict_good_prob(model, payload_optimal)
    p_oversleep = _predict_good_prob(model, payload_oversleep)

    assert p_oversleep <= p_optimal + 1e-6, "11h oversleep scenario should not score above optimal"


def test_stress_and_activity_joint_effect(model):
    payload_low_stress = _baseline_payload(stress_level=2, physical_activity_level=70)
    payload_high_stress = _baseline_payload(stress_level=8, physical_activity_level=20)

    p_low = _predict_good_prob(model, payload_low_stress)
    p_high = _predict_good_prob(model, payload_high_stress)

    assert p_low > p_high, "High stress with low activity should reduce good-sleep probability"

