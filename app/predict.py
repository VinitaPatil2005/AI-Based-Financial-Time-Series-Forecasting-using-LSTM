import numpy as np
import joblib

from tensorflow.keras.models import load_model


# Load Model
model = load_model("../models/best_lstm_model.h5")

# Load Scaler
scaler = joblib.load("../models/scaler.pkl")


def predict_stock_price(last_60_days):

    """
    Predict the next day's closing stock price.

    Parameters
    ----------
    last_60_days : numpy array
        Shape should be (60,1)

    Returns
    -------
    float
        Predicted stock price
    """

    input_data = np.array(last_60_days)

    input_data = input_data.reshape(1,60,1)

    prediction = model.predict(input_data, verbose=0)

    prediction = scaler.inverse_transform(prediction)

    return float(prediction[0][0])