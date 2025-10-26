import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
import plotly.graph_objects as go

MODEL_PATH = Path("models") / "sleep_quality_model.joblib"
THRESHOLD = 0.5

st.markdown("""
<style>
    .main {
        padding: 2rem 1rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .recommendation-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
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

def predict(payload: dict):
    model = load_model()
    payload_with_missing = payload.copy()
    payload_with_missing["sleep_disorder_missing"] = 1 if payload["sleep_disorder"] == "None" else 0
    df = pd.DataFrame([payload_with_missing])
    prob_good = float(model.predict_proba(df)[0, 1])
    label = "Good" if prob_good >= THRESHOLD else "Poor"
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
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sleep Score", 'font': {'size': 24, 'color': '#4A90E2'}},
        delta={'reference': 70},
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
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"},
        height=300
    )
    
    return fig

st.set_page_config(
    page_title="Sleep Quality AI Predictor",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

col_header1, col_header2 = st.columns([3, 1])
with col_header1:
    st.markdown("<h1 style='text-align: left;'>🌙 Sleep Quality AI Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #666; font-size: 1.2em;'>Discover your sleep score and get personalized insights</p>", unsafe_allow_html=True)
with col_header2:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption(f"Model: Baseline (ROC-AUC: 0.999)")

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
    **Model Type:** Logistic Regression  
    **Performance:** ROC-AUC = 0.999  
    **Training Data:** 374 subjects  
    **Top Features:** Sleep duration, stress level, occupation
    """)

with st.form("sleep_predictor_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Personal Information")
        age = st.slider("Age (years)", min_value=18, max_value=100, value=35, step=1)
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        occupation = st.selectbox(
            "Occupation",
            ["Software Engineer", "Doctor", "Nurse", "Teacher", "Engineer", 
             "Accountant", "Lawyer", "Salesperson", "Scientist", "Other"]
        )
        
        st.markdown("---")
        
        st.subheader("❤️ Health Metrics")
        heart_rate = st.slider(
            "Resting Heart Rate (bpm)",
            min_value=40, max_value=120, value=72, step=1
        )
        st.caption(f"Healthy range: 60-100 bpm")
        
        blood_pressure = st.selectbox(
            "Blood Pressure",
            ["115/75", "117/76", "120/80", "125/80", "130/85", "140/90", "140/95"],
            index=3
        )
        
        bmi_category = st.selectbox(
            "BMI Category",
            ["Normal", "Overweight", "Normal Weight", "Obese"],
            index=0
        )
    
    with col2:
        st.subheader("🏃 Lifestyle")
        
        daily_steps = st.slider(
            "Daily Steps",
            min_value=0, max_value=20000, value=8000, step=100
        )
        
        steps_progress = daily_steps / 10000
        if daily_steps < 6000:
            st.warning("⚠️ Below recommended 8,000 steps")
            st.progress(min(steps_progress, 1.0))
        elif daily_steps >= 10000:
            st.success("✅ Excellent activity level!")
            st.progress(min(steps_progress, 1.0))
        else:
            st.info("📊 Good progress toward 10,000 steps")
            st.progress(min(steps_progress, 1.0))
        
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
            "Sleep Duration (hours per night)",
            min_value=4.0, max_value=12.0, value=7.5, step=0.5
        )
        
        if sleep_duration < 6:
            st.error(f"⚠️ Critically low ({sleep_duration}h). Medical consultation recommended.")
        elif sleep_duration < 7:
            st.warning(f"⚠️ Below optimal (7-9h recommended)")
        elif sleep_duration >= 9:
            st.info(f"ℹ️ Good duration (7-9h is optimal range)")
        else:
            st.success("✅ Optimal sleep duration!")
        
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
            result = predict(payload)
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
                
                st.markdown(f"""
                <div class='metric-card'>
                    <p style='text-align: center; margin: 0.5em 0;'>Confidence</p>
                    <h3 style='text-align: center; color: #4A90E2;'>{result['prob_good']:.1%}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 📊 Detailed Metrics")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                delta_sleep = sleep_duration - 7.5
                st.metric("Sleep Score", f"{score}/100", f"{score-70:.1f}")
            with metric_col2:
                st.metric("Quality Label", result["predicted_label"])
            with metric_col3:
                st.metric("Confidence", f"{result['prob_good']:.1%}")
            with metric_col4:
                st.metric("Sleep Duration", f"{sleep_duration}h", f"{delta_sleep:+.1f}h")
            
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
                'Factor': ['Sleep Duration', 'Stress Level', 'Occupation Type', 'Heart Rate', 'Daily Steps'],
                'Impact': ['High', 'High', 'Medium', 'Medium', 'Medium'],
                'Your Value': [f"{sleep_duration}h", f"{stress_level}/10", occupation, f"{heart_rate} bpm", f"{daily_steps:,}"]
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
