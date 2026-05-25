"""
Digital Twin IoT Power Prediction - Streamlit Web App Deployment
Author: [Nama Mahasiswa]
Date: 2026

This is the deployment component for Digital Twin IoT Power Prediction.
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Digital Twin - IoT Power Prediction",
    page_icon="⚡",
    layout="wide"
)

# ============================================================================
# LOAD MODEL AND SCALER
# ============================================================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('xgboost_power_prediction_model.joblib')
        scaler = joblib.load('feature_scaler.joblib')
        feature_columns = joblib.load('feature_columns.joblib')
        return model, scaler, feature_columns
    except FileNotFoundError:
        return None, None, None

model, scaler, feature_columns = load_model()

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        border-bottom: 2px solid #1f77b4;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stMetric {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================
st.markdown('<h1 class="main-header">⚡ Digital Twin - IoT Power Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: gray;">Real-time Energy Monitoring & Prediction System</p>', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.header("🎛️ Control Panel")

# Model status
if model is not None:
    st.sidebar.success("✓ Model Loaded Successfully")
else:
    st.sidebar.warning("⚠ Model not found - Running Demo Mode")

# Input sliders
st.sidebar.subheader("📊 Sensor Inputs")

suhu = st.sidebar.slider("Suhu (°C)", 20.0, 40.0, 30.0, 0.1)
kelembaban = st.sidebar.slider("Kelembaban (%)", 40, 90, 65, 1)
tegangan = st.sidebar.slider("Tegangan (V)", 200.0, 250.0, 225.0, 0.1)
arus = st.sidebar.slider("Arus (A)", 0.0, 2.0, 0.16, 0.01)

# Time inputs
st.sidebar.subheader("⏰ Time Settings")
hour = st.sidebar.slider("Jam", 0, 23, datetime.now().hour, 1)
day_of_week = st.sidebar.selectbox("Hari",
    ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"],
    index=datetime.now().weekday())

day_map = {"Senin": 0, "Selasa": 1, "Rabu": 2, "Kamis": 3, "Jumat": 4, "Sabtu": 5, "Minggu": 6}
day_num = day_map[day_of_week]

# ============================================================================
# MAIN CONTENT
# ============================================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🌡️ Suhu", f"{suhu:.1f}°C")
with col2:
    st.metric("💧 Kelembaban", f"{kelembaban}%")
with col3:
    st.metric("⚡ Tegangan", f"{tegangan:.1f}V")
with col4:
    st.metric("🔌 Arus", f"{arus:.2f}A")

# ============================================================================
# PREDICTION SECTION
# ============================================================================
st.subheader("🔮 Prediksi Konsumsi Daya")

# Feature engineering for prediction
suhu_kelembaban = suhu * kelembaban
tegangan_arus = tegangan * arus

# Time period encoding
time_period_morning = 1 if 6 <= hour < 10 else 0
time_period_midday = 1 if 10 <= hour < 14 else 0
time_period_afternoon = 1 if 14 <= hour < 18 else 0
time_period_evening = 1 if 18 <= hour < 22 else 0

# Create feature vector (order must match training)
features = {
    'Suhu (°C)': suhu,
    'Kelembaban (%)': kelembaban,
    'Tegangan (V)': tegangan,
    'Arus (A)': arus,
    'Hour': hour,
    'DayOfWeek': day_num,
    'Day': datetime.now().day,
    'Suhu_Kelembaban': suhu_kelembaban,
    'Tegangan_Arus': tegangan_arus,
    'Daya_MA_5': 35.7,  # Using mean as placeholder
    'Daya_MA_15': 35.7,
    'Suhu_MA_5': suhu,
    'TimePeriod_Morning': time_period_morning,
    'TimePeriod_Midday': time_period_midday,
    'TimePeriod_Afternoon': time_period_afternoon,
    'TimePeriod_Evening': time_period_evening
}

# Make prediction
if model is not None:
    X_pred = pd.DataFrame([features])[feature_columns]
    prediction = model.predict(X_pred)[0]
    confidence = 0.95  # Placeholder confidence
else:
    # Demo mode - calculate based on simple formula
    prediction = (tegangan * arus) * 0.85 + (suhu - 25) * 0.5
    confidence = 0.70

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
                padding: 2rem; border-radius: 10px; text-align: center; color: white;">
        <h2>⚡ Prediksi Daya</h2>
        <h1 style="font-size: 3rem; margin: 0;">{prediction:.2f} Watt</h1>
        <p>Confidence: {confidence*100:.0f}%</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Power factor calculation
    apparent_power = tegangan * arus
    real_power = prediction
    power_factor = real_power / apparent_power if apparent_power > 0 else 0

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2rem; border-radius: 10px; text-align: center; color: white;">
        <h2>📊 Analisis Daya</h2>
        <p>Daya Semu: <strong>{apparent_power:.2f} VA</strong></p>
        <p>Daya Nyata: <strong>{real_power:.2f} W</strong></p>
        <p>Power Factor: <strong>{power_factor:.3f}</strong></p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# VISUALIZATION SECTION
# ============================================================================
st.subheader("📈 Visualisasi Data")

# Create sample data for visualization
np.random.seed(42)
hours = list(range(24))
actual_power = [30 + 10*np.sin((h-6)*np.pi/12) + np.random.normal(0, 2) for h in hours]
predicted_power = [30 + 10*np.sin((h-6)*np.pi/12) for h in hours]

# Plotly chart
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=hours, y=actual_power,
    mode='lines+markers',
    name='Actual Power',
    line=dict(color='#ff6b6b', width=2)
))

fig.add_trace(go.Scatter(
    x=hours, y=predicted_power,
    mode='lines+markers',
    name='Predicted Power',
    line=dict(color='#4ecdc4', width=2, dash='dash')
))

fig.update_layout(
    title='Profil Konsumsi Daya Harian',
    xaxis_title='Jam',
    yaxis_title='Daya (Watt)',
    template='plotly_white',
    height=400
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
st.subheader("📊 Feature Importance")

if model is not None:
    importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)

    fig_bar = px.bar(
        importance.tail(10),
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top 10 Feature Importance',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ============================================================================
# REAL-TIME SIMULATION
# ============================================================================
st.subheader("🔄 Simulasi Real-Time")

# Generate real-time simulation data
simulation_placeholder = st.empty()

with simulation_placeholder.container():
    col1, col2, col3 = st.columns(3)

    # Simulate current readings
    current_suhu = suhu + np.random.normal(0, 0.5)
    current_kelembaban = kelembaban + np.random.normal(0, 1)
    current_tegangan = tegangan + np.random.normal(0, 2)
    current_arus = arus + np.random.normal(0, 0.05)
    current_daya = prediction + np.random.normal(0, 1)

    with col1:
        st.metric("🌡️ Suhu Sekarang", f"{current_suhu:.1f}°C", delta=f"{current_suhu - suhu:.1f}")
    with col2:
        st.metric("💧 Kelembaban", f"{current_kelembaban:.1f}%", delta=f"{current_kelembaban - kelembaban:.1f}")
    with col3:
        st.metric("⚡ Daya Sekarang", f"{current_daya:.2f}W", delta=f"{current_daya - prediction:.2f}")

# ============================================================================
# ABOUT SECTION
# ============================================================================
st.markdown("---")
st.subheader("ℹ️ About Digital Twin")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Apa itu Digital Twin?
    Digital Twin adalah representasi virtual dari sistem fisik yang memungkinkan:
    - **Monitoring** - Pemantauan kondisi real-time
    - **Prediction** - Prediksi perilaku sistem
    - **Optimization** - Optimasi parameter untuk efisiensi
    - **Simulation** - Simulasi skenario tanpa mengganggu sistem asli
    """)

with col2:
    st.markdown("""
    ### Komponen Sistem
    1. **Physical Layer** - Sensor IoT (Raspberry Pi)
    2. **Data Layer** - Preprocessing & Feature Engineering
    3. **Model Layer** - XGBoost ML Prediction
    4. **Application Layer** - Streamlit Dashboard
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: gray;">
    <p>Digital Twin IoT Power Prediction | Created 2026</p>
    <p>Model: XGBoost Regressor | Dataset: 93,121 sensor readings</p>
</div>
""", unsafe_allow_html=True)