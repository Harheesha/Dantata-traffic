# utils.py

import requests
import pandas as pd
import joblib
from config import TOMTOM_API_KEY, TOMTOM_API_URL


def load_model(model_path, encoder_path):
    """Load trained model and label encoder."""
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    return model, encoder


def fetch_live_traffic_data(lat, lon):
    """Fetch real-time traffic data from TomTom API."""
    params = {"point": f"{lat},{lon}", "key": TOMTOM_API_KEY}
    response = requests.get(TOMTOM_API_URL, params=params)

    if response.status_code != 200:
        raise ConnectionError("Failed to fetch data from TomTom API.")

    segment = response.json()["flowSegmentData"]
    return {
        "current_speed": segment["currentSpeed"],
        "free_flow_speed": segment["freeFlowSpeed"],
        "confidence": segment["confidence"],
        "hour_of_day": pd.Timestamp.now().hour
    }
