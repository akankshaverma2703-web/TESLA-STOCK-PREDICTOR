import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -----------------------------
# Load trained model
# -----------------------------
model = pickle.load(open('model.pkl', 'rb'))  # model.pkl same folder me

# -----------------------------
# Load data for default values
# -----------------------------
df = pd.read_csv('tesla2.csv')  # CSV same folder me

# Calculate default input values from last row
open_close_default = float(df['Open'].iloc[-1] - df['Close'].iloc[-1])
low_high_default = float(df['High'].iloc[-1] - df['Low'].iloc[-1])
quarter_end_default = 0  # Default value

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📈 Tesla Stock Price Predictor")
st.write("Enter values to predict stock movement:")

open_close = st.number_input("Open - Close", value=open_close_default)
low_high = st.number_input("Low - High", value=low_high_default)
quarter_end = st.selectbox("Is Quarter End?", [0, 1], index=quarter_end_default)

if st.button("Predict"):
    features = np.array([[open_close, low_high, quarter_end]])
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.success("📊 Price will go UP 🚀")
    else:
        st.error("📉 Price will go DOWN ⬇️")