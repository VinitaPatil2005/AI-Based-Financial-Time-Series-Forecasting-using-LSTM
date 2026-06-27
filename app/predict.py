import os
import numpy as np
import joblib

from tensorflow.keras.models import load_model

# Get project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_lstm_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Load model and scaler
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def predict_stock_price(last_60_days):

    input_data = np.array(last_60_days)

    input_data = input_data.reshape(1, 60, 1)

    prediction = model.predict(input_data, verbose=0)

    prediction = scaler.inverse_transform(prediction)

    return float(prediction[0][0])