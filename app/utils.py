import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


def load_data():

    return pd.read_csv(
        os.path.join(
            BASE_DIR,
            "data",
            "clean_stock_data.csv"
        )
    )


def load_model_and_scaler():

    model = load_model(
        os.path.join(
            BASE_DIR,
            "models",
            "best_lstm_model.h5"
        )
    )

    scaler = joblib.load(
        os.path.join(
            BASE_DIR,
            "models",
            "scaler.pkl"
        )
    )

    return model, scaler


def calculate_metrics():

    model, scaler = load_model_and_scaler()

    X_test = np.load(
        os.path.join(
            BASE_DIR,
            "data",
            "X_test.npy"
        )
    )

    y_test = np.load(
        os.path.join(
            BASE_DIR,
            "data",
            "y_test.npy"
        )
    )

    predictions = model.predict(X_test, verbose=0)

    predictions = scaler.inverse_transform(predictions)

    actual = scaler.inverse_transform(y_test)

    mae = mean_absolute_error(
        actual,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predictions
        )
    )

    return actual, predictions, mae, rmse