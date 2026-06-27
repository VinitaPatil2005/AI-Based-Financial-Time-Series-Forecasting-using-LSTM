import streamlit as st

from config import (
    APP_TITLE,
    COMPANY_NAME,
    MODEL_NAME,
    SEQUENCE_LENGTH,
    EPOCHS,
    OPTIMIZER,
    LOSS_FUNCTION
)

from utils import (
    load_data,
    calculate_metrics
)

from predict import predict_stock_price

from charts import (
    closing_price_chart,
    last30_chart,
    candlestick_chart,
    prediction_chart
)

# ----------------------------------------------------
# Page Configuration
# ----------------------------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide"
)

# ----------------------------------------------------
# Load Data
# ----------------------------------------------------

data = load_data()

actual, predicted, mae, rmse = calculate_metrics()

# ----------------------------------------------------
# Predict Next Day Price
# ----------------------------------------------------

last_60_days = data["Close"].values[-60:].reshape(60, 1)

next_price = predict_stock_price(last_60_days)

# ----------------------------------------------------
# Title
# ----------------------------------------------------

st.title(APP_TITLE)

st.markdown(
"""
### Apple Stock Price Forecasting using LSTM

This dashboard predicts the **next day's Apple stock closing price**
using a trained **Long Short-Term Memory (LSTM)** model.
"""
)

st.divider()

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

st.sidebar.title("Project Information")

st.sidebar.markdown(f"**Company** : {COMPANY_NAME}")

st.sidebar.markdown(f"**Model** : {MODEL_NAME}")

st.sidebar.markdown(f"**Sequence Length** : {SEQUENCE_LENGTH} Days")

st.sidebar.markdown(f"**Epochs** : {EPOCHS}")

st.sidebar.markdown(f"**Optimizer** : {OPTIMIZER}")

st.sidebar.markdown(f"**Loss Function** : {LOSS_FUNCTION}")

# ----------------------------------------------------
# KPI Cards
# ----------------------------------------------------

highest = data["Close"].max()
lowest = data["Close"].min()
average = data["Close"].mean()
latest = data["Close"].iloc[-1]

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Highest Price",
    f"${highest:.2f}"
)

c2.metric(
    "Lowest Price",
    f"${lowest:.2f}"
)

c3.metric(
    "Average Price",
    f"${average:.2f}"
)

c4.metric(
    "Latest Closing Price",
    f"${latest:.2f}"
)

c5, c6, c7 = st.columns(3)

c5.metric(
    "Predicted Next Price",
    f"${next_price:.2f}"
)

c6.metric(
    "RMSE",
    f"{rmse:.2f}"
)

c7.metric(
    "MAE",
    f"{mae:.2f}"
)

st.divider()

# ----------------------------------------------------
# Historical Closing Price
# ----------------------------------------------------

st.subheader("Historical Closing Price")

st.plotly_chart(
    closing_price_chart(data),
    use_container_width=True
)

# ----------------------------------------------------
# Last 30 Days
# ----------------------------------------------------

st.subheader("Last 30 Trading Days")

st.plotly_chart(
    last30_chart(data),
    use_container_width=True
)

# ----------------------------------------------------
# Candlestick Chart
# ----------------------------------------------------

st.subheader("Candlestick Chart")

st.plotly_chart(
    candlestick_chart(data),
    use_container_width=True
)

# ----------------------------------------------------
# Actual vs Predicted
# ----------------------------------------------------

st.subheader("Actual vs Predicted")

st.plotly_chart(
    prediction_chart(
        actual,
        predicted
    ),
    use_container_width=True
)

# ----------------------------------------------------
# Latest Records
# ----------------------------------------------------

st.subheader("Latest Stock Records")

st.dataframe(
    data.tail(10),
    use_container_width=True
)

# ----------------------------------------------------
# Download Dataset
# ----------------------------------------------------

csv = data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇ Download Dataset",
    data=csv,
    file_name="apple_stock.csv",
    mime="text/csv"
)

st.divider()

# ----------------------------------------------------
# Dataset Information
# ----------------------------------------------------

st.subheader("Dataset Information")

d1, d2, d3 = st.columns(3)

d1.metric(
    "Total Records",
    len(data)
)

d2.metric(
    "Features",
    len(data.columns)
)

d3.metric(
    "Latest Date",
    str(data["Date"].iloc[-1])
)

st.divider()

# ----------------------------------------------------
# Model Information
# ----------------------------------------------------

st.subheader("Model Information")

info1, info2 = st.columns(2)

with info1:
    st.markdown(f"**Model:** {MODEL_NAME}")
    st.markdown(f"**Sequence Length:** {SEQUENCE_LENGTH}")
    st.markdown(f"**Epochs:** {EPOCHS}")

with info2:
    st.markdown(f"**Optimizer:** {OPTIMIZER}")
    st.markdown(f"**Loss Function:** {LOSS_FUNCTION}")
    st.markdown("**Framework:** TensorFlow + Streamlit")

st.divider()

# ----------------------------------------------------
# About
# ----------------------------------------------------

st.subheader("About")

st.info(
"""
This project forecasts Apple's next-day closing stock price using a Long Short-Term Memory (LSTM) neural network.

### Workflow
- Load historical stock prices
- Normalize data using MinMaxScaler
- Create 60-day sequences
- Train an LSTM model
- Predict future closing prices
- Evaluate using RMSE and MAE
- Visualize results with interactive Plotly charts

**Technologies Used**
- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Plotly
- Streamlit
- Scikit-learn
"""
)

st.success("Dashboard Loaded Successfully")