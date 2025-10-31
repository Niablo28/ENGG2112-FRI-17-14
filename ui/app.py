import json
import sys
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
import plotly.graph_objects as go
import glob
import os

# set_page_config MUST be the first Streamlit command
st.set_page_config(
    page_title="Sleep Quality AI Predictor",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

def _resolve_model_path():
    env_path = os.environ.get("MODEL_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    pref = Path("models") / "model_augmented_latest.joblib"
    if pref.exists():
        return pref
    candidates = sorted(map(Path, glob.glob("models/*.joblib")))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError("No model artifact found under models/")

MODEL_PATH = _resolve_model_path()
# Default threshold; can be overridden via sidebar control
THRESHOLD = 0.5

# Ensure custom transformers used in the pickled model are importable
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

# Backward-compat shim: some older pickles reference __main__._split_bp
# Ensure the attribute exists by aliasing from scripts.shared_transforms
try:
    import importlib, types
    stx = importlib.import_module('scripts.shared_transforms')
    import __main__ as _main
    # Attach attributes if missing
    if not hasattr(_main, '_split_bp'):
        setattr(_main, '_split_bp', stx._split_bp)
    if not hasattr(_main, '_get_bp_split_feature_names_out'):
        setattr(_main, '_get_bp_split_feature_names_out', stx._get_bp_split_feature_names_out)
except Exception:
    pass

st.markdown("""
<style>
    .main {
        padding: 2rem 1rem;
    }
    .metric-card {
        background: #1f2937; /* dark card */
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .recommendation-box {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #00B894;
    }
    h1, h2, h3 {
        color: #4A90E2;
    }
    .slider-info {
        font-size: 0.85em;
        color: #666;
        margin-top: -10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH.resolve()}")
    return joblib.load(MODEL_PATH)

def _normalize_payload(payload: dict) -> dict:
    """Normalize categorical labels to the training vocabulary."""
    normalized = payload.copy()
    # Normalize BMI categories
    bmi_map = {
        "Normal Weight": "Normal",
    }
    if normalized.get("bmi_category") in bmi_map:
        normalized["bmi_category"] = bmi_map[normalized["bmi_category"]]
    # Normalize occupations
    occupation_map = {
        "Sales Representative": "Salesperson",
    }
    if normalized.get("occupation") in occupation_map:
        normalized["occupation"] = occupation_map[normalized["occupation"]]
    return normalized

def predict(payload: dict, threshold: float):
    model = load_model()
    payload = _normalize_payload(payload)
    payload_with_missing = payload.copy()
    # Fix: "None" is a valid category (present), not missing. Missing means None or empty string.
    val = payload.get("sleep_disorder", None)
    is_missing = (val is None) or (isinstance(val, str) and val.strip() == "")
    payload_with_missing["sleep_disorder_missing"] = 1 if is_missing else 0
    # Pre-split blood pressure into numeric columns (pipeline expects bp_sys/bp_dia)
    bp_val = str(payload_with_missing.get("blood_pressure", ""))
    import re
    m = re.match(r"^(\d{2,3})/(\d{2,3})$", bp_val)
    if m:
        payload_with_missing["bp_sys"] = float(m.group(1))
        payload_with_missing["bp_dia"] = float(m.group(2))
    df = pd.DataFrame([payload_with_missing])
    prob_good = float(model.predict_proba(df)[0, 1])
    label = "Good" if prob_good >= threshold else "Poor"
    score = round(prob_good * 100.0, 1)
    return {"sleep_score": score, "prob_good": prob_good, "predicted_label": label}

def generate_recommendations(payload, score):
    tips = []
    
    if payload['stress_level'] > 6:
        tips.append("🧘 High stress detected. Try meditation or breathing exercises.")
    if payload['sleep_duration'] < 7:
        tips.append("⏰ Aim for 7-9 hours of sleep for optimal health.")
    if payload['daily_steps'] < 6000:
        tips.append("👟 Consider increasing daily steps to 8,000-10,000.")
    if payload['heart_rate'] > 80:
        tips.append("❤️ Elevated resting heart rate. Consider cardio exercise.")
    if payload['physical_activity_level'] < 50:
        tips.append("🏋️ Low activity level. Try adding 30 min exercise daily.")
    if payload['sleep_duration'] < 6:
        tips.append("💤 Inadequate sleep. Create a consistent sleep schedule.")
    
    if score < 50:
        tips.append("💡 Low sleep score. Consult a healthcare professional.")
    elif score >= 80:
        tips.append("✨ Excellent sleep quality! Keep up the healthy habits.")
    
    return tips

def create_gauge_chart(score):
    if score < 50:
        bar_color = "#FF6B6B"
    elif score < 70:
        bar_color = "#FFD93D"
    else:
        bar_color = "#00B894"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sleep Score", 'font': {'size': 24, 'color': '#4A90E2'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': bar_color, 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': "#FF6B6B"},
                {'range': [50, 70], 'color': "#FFD93D"},
                {'range': [70, 100], 'color': "#00B894"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font={'color': "#E5E7EB", 'family': "Arial"},
        height=300
    )
    
    return fig

col_header1, col_header2 = st.columns([3, 1])
with col_header1:
    st.markdown("<h1 style='text-align: left;'>🌙 Sleep Quality AI Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #666; font-size: 1.2em;'>Discover your sleep score and get personalized insights</p>", unsafe_allow_html=True)
with col_header2:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption(f"Model: Augmented (ROC-AUC: 0.996)")

st.divider()

with st.sidebar:
    st.header("💡 Quick Tips")
    tips = [
        "🌙 Aim for 7-9 hours of sleep nightly",
        "🧘 Manage stress through meditation or yoga",
        "🏃 Get 8,000-10,000 steps daily",
        "⏰ Maintain consistent sleep schedule",
        "🚫 Avoid screens 1 hour before bed",
        "💧 Stay hydrated throughout the day"
    ]
    for tip in tips:
        st.markdown(f"- {tip}")
    
    st.divider()
    
    st.header("📊 Model Info")
    st.info("""
    **Model Type:** Logistic Regression (Augmented)  
    **Performance:** ROC-AUC = 0.996  
    **Training Data:** 594 subjects (374 original + 220 synthetic)  
    **Top Features:** Sleep duration, stress level, occupation  
    **Improvements:** Better edge case handling, sleep disorder detection, severe case recognition
    """)
    st.caption("This tool is for educational purposes and general wellness insights only and is not medical advice.")
    st.divider()
    st.header("⚙️ Prediction Threshold")
    st.caption("Choose the probability cutoff for 'Good' vs 'Poor'.")
    user_threshold = st.slider(
        "Threshold (P(Good) ≥ threshold ⇒ Good)",
        min_value=0.1, max_value=0.9, value=THRESHOLD, step=0.05
    )

with st.form("sleep_predictor_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Personal Information")
        age = st.slider("Age (years)", min_value=18, max_value=100, value=35, step=1)
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        # Occupation removed from UI; default to 'Other' internally
        occupation = "Other"
        
        st.markdown("---")
        
        st.subheader("❤️ Health Metrics")
        heart_rate = st.slider(
            "Resting Heart Rate (bpm)",
            min_value=40, max_value=120, value=72, step=1
        )
        st.caption(f"Healthy range: 60-100 bpm")
        
        # Blood pressure removed from UI; default internally
        blood_pressure = "125/80"
        
        bmi_category = st.selectbox(
            "BMI Category",
            ["Normal", "Overweight", "Normal Weight", "Obese"],
            index=0
        )
    
    with col2:
        st.subheader("🏃 Lifestyle")
        # Daily steps removed from UI; default internally
        daily_steps = 8000
        
        physical_activity_level = st.slider(
            "Physical Activity Level (0-100)",
            min_value=0, max_value=100, value=60, step=1
        )
        
        stress_emojis = ["😊", "🙂", "😐", "😟", "😰", "😫", "😵", "😵", "😵", "😵"]
        
        stress_level = st.slider(
            "Stress Level (0-10)",
            min_value=0, max_value=10, value=3, step=1
        )
        
        if 0 <= stress_level < 3:
            st.success(f"{stress_emojis[stress_level]} Low stress")
        elif 3 <= stress_level < 6:
            st.info(f"{stress_emojis[stress_level]} Moderate stress")
        else:
            st.warning(f"{stress_emojis[min(stress_level, 9)]} High stress detected")
        
        st.markdown("---")
        
        st.subheader("😴 Sleep Habits")
        
        sleep_duration = st.slider(
            "Sleep Duration (hours per night, 4–12)",
            min_value=4.0, max_value=12.0, value=7.5, step=0.5
        )
        st.caption("<6h: critically low • 6–7h: below optimal • 7–9h: optimal • >9h: oversleeping may reduce quality", unsafe_allow_html=True)
        
        if sleep_duration < 6:
            st.error(f"⚠️ Critically low ({sleep_duration}h). Medical consultation recommended.")
        elif sleep_duration < 7:
            st.warning(f"⚠️ Below optimal (7-9h recommended)")
        elif sleep_duration > 9:
            st.warning("⚠️ Oversleeping (>9h) may reduce quality.")
        else:
            st.success("✅ Optimal sleep duration (7–9h).")
        
        sleep_disorder = st.selectbox(
            "Sleep Disorder",
            ["None", "Insomnia", "Sleep Apnea", "Other"]
        )

    submitted = st.form_submit_button("🎯 Predict My Sleep Score", use_container_width=True, type="primary")

if submitted:
    with st.spinner("🔮 Analyzing your sleep patterns..."):
        payload = {
            "age": int(age),
            "gender": gender,
                "occupation": occupation,
            "bmi_category": bmi_category,
                "blood_pressure": blood_pressure,
            "heart_rate": int(heart_rate),
                "daily_steps": int(daily_steps),
            "sleep_duration": float(sleep_duration),
            "physical_activity_level": int(physical_activity_level),
            "stress_level": int(stress_level),
            "sleep_disorder": sleep_disorder
        }
        
        try:
            result = predict(payload, threshold=user_threshold)
            score = result['sleep_score']
            
            st.divider()
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_chart1, col_chart2 = st.columns([2, 1])
            with col_chart1:
                fig = create_gauge_chart(score)
                st.plotly_chart(fig, use_container_width=True)
            
            with col_chart2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                
                if result["predicted_label"] == "Good":
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h2 style='color: #00B894; text-align: center;'>✨</h2>
                        <h3 style='color: #00B894; text-align: center;'>GOOD</h3>
                        <p style='text-align: center; font-size: 0.9em;'>Sleep Quality</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h2 style='color: #FF6B6B; text-align: center;'>⚠️</h2>
                        <h3 style='color: #FF6B6B; text-align: center;'>NEEDS IMPROVEMENT</h3>
                        <p style='text-align: center; font-size: 0.9em;'>Sleep Quality</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Confidence display removed per request
            
            st.markdown("### 📊 Details")
            metric_col1, metric_col2, metric_col4 = st.columns([1,1,1])
            with metric_col1:
                st.markdown(f"<div class='metric-card'><h3 style='margin:0;color:#E5E7EB;'>Sleep Score</h3><p style='margin:0;color:#9CA3AF;'>{score}/100</p></div>", unsafe_allow_html=True)
            with metric_col2:
                st.markdown(f"<div class='metric-card'><h3 style='margin:0;color:#E5E7EB;'>Quality</h3><p style='margin:0;color:#9CA3AF;'>{result['predicted_label']}</p></div>", unsafe_allow_html=True)
            with metric_col4:
                st.markdown(f"<div class='metric-card'><h3 style='margin:0;color:#E5E7EB;'>Sleep Duration</h3><p style='margin:0;color:#9CA3AF;'>{sleep_duration}h</p></div>", unsafe_allow_html=True)
            
            st.markdown("### 💡 Personalized Recommendations")
            recommendations = generate_recommendations(payload, score)
            
            if recommendations:
                for tip in recommendations:
                    st.markdown(f"""
                    <div class='recommendation-box'>
                        <p style='margin: 0; font-size: 1.1em;'>{tip}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("🎉 All metrics look great! Keep up the healthy habits.")
            
            st.markdown("### 🔑 Key Factors Affecting Your Score")
            top_features_data = pd.DataFrame({
                'Factor': ['Sleep Duration', 'Stress Level', 'Heart Rate', 'Physical Activity'],
                'Impact': ['High', 'High', 'Medium', 'Medium'],
                'Your Value': [f"{sleep_duration}h", f"{stress_level}/10", f"{heart_rate} bpm", f"{physical_activity_level}/100"]
            })
            st.dataframe(top_features_data, use_container_width=True, hide_index=True)
            
            with st.expander("📋 Technical Details"):
                st.code(json.dumps(payload, indent=2), language="json")
        
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            with st.expander("Debug Info"):
                st.exception(e)

else:
    st.markdown("""
    <div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    border-radius: 15px; margin: 2rem 0;'>
        <h2 style='color: white;'>Welcome to Sleep Quality AI Predictor</h2>
        <p style='color: white; font-size: 1.1em;'>Fill in your information above and click "Predict My Sleep Score" to get started</p>
        <p style='color: white; opacity: 0.9;'>Our AI model analyzes your lifestyle, health metrics, and sleep habits to predict your sleep quality</p>
    </div>
    """, unsafe_allow_html=True)
