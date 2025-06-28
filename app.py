# app.py

import streamlit as st
import pandas as pd
from config import LOCATIONS, MAX_SPEED, CONFIDENCE_STEP, HOURS_IN_DAY
from utils import load_model, fetch_live_traffic_data

# App configuration
st.set_page_config(page_title="Traffic Congestion Predictor", layout="centered")
st.title("Dantata Traffic Congestion Predictor")
st.markdown("Predict traffic congestion levels at key bridge locations using real-time or manual input.")

# Load model and encoder
MODEL_PATH = "models/traffic_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
model, label_encoder = load_model(MODEL_PATH, ENCODER_PATH)

# Location selection
st.header("Select Location & Input Mode")
location = st.selectbox("Bridge Location", list(LOCATIONS.keys()))
input_method = st.radio("Input Method", ["Live Traffic Data", "Manual Input"])
lat, lon = LOCATIONS[location]

# Input Section
if input_method == "Manual Input":
    st.subheader("Manual Traffic Data Input")
    col1, col2 = st.columns(2)
    with col1:
        current_speed = st.slider("Current Speed (km/h)", 0, MAX_SPEED, 60)
        free_flow_speed = st.slider("Free Flow Speed (km/h)", 0, MAX_SPEED, 60)
    with col2:
        confidence = st.slider("Confidence Level", 0.0, 1.0, 0.90, step=CONFIDENCE_STEP)
        hour_of_day = st.selectbox("Hour of Day", list(range(HOURS_IN_DAY)))

else:
    st.subheader("Fetching Live Traffic Data")
    try:
        traffic_data = fetch_live_traffic_data(lat, lon)
        current_speed = traffic_data["current_speed"]
        free_flow_speed = traffic_data["free_flow_speed"]
        confidence = traffic_data["confidence"]
        hour_of_day = traffic_data["hour_of_day"]

        st.success("Real-time traffic data loaded.")
        st.write(f"**Current Speed:** {current_speed} km/h")
        st.write(f"**Free Flow Speed:** {free_flow_speed} km/h")
        st.write(f"**Confidence:** {confidence}")
        st.write(f"**Hour:** {hour_of_day}")

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

# Prediction
if st.button("Predict Congestion Level"):
    input_data = pd.DataFrame([{
        "current_speed": current_speed,
        "free_flow_speed": free_flow_speed,
        "confidence": confidence,
        "hour_of_day": hour_of_day
    }])

    prediction = model.predict(input_data)[0]
    congestion_level = label_encoder.inverse_transform([prediction])[0]

    st.subheader("Prediction Result")
    if congestion_level.lower() == "high":
        st.error("🔴 HIGH Traffic Congestion")
    elif congestion_level.lower() == "medium":
        st.warning("🟠 MEDIUM Traffic Congestion")
    else:
        st.success("🟢 LOW Traffic Congestion")

    st.markdown("**Model Input Data:**")
    st.json(input_data.to_dict(orient="records")[0])
