import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import pickle

# Load dataset
def load_data():
    file_path = "AAPL_Processed.csv"
    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
    return df

def plot_stock_data(df):
    fig = px.line(df, x=df.index, y="Close", title="Apple Stock Closing Prices")
    st.plotly_chart(fig)

def train_arima_model(df):
    model = ARIMA(df['Close'], order=(5,1,0))
    model_fit = model.fit()
    with open("arima_model.pkl", "wb") as f:
        pickle.dump(model_fit, f)
    return model_fit

def load_model():
    with open("arima_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def forecast_next_days(model, steps=30):
    forecast = model.forecast(steps=steps)
    return forecast

# Streamlit UI
st.title("Apple Stock Market Analysis and Prediction")

df = load_data()
st.subheader("Stock Data Preview")
st.write(df.tail())

plot_stock_data(df)

if st.button("Train Model"):
    model = train_arima_model(df)
    st.success("Model Trained and Saved!")

if st.button("Predict Next 30 Days"):
    model = load_model()
    forecast = forecast_next_days(model, steps=30)
    st.subheader("30-Day Stock Price Prediction")
    st.line_chart(forecast)
