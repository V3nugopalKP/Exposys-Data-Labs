import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('best_model.pkl')

st.title("📈 Company Profit Prediction App")
st.write("Enter the company’s spending details below to predict its profit.")

# Input fields
rd_spend = st.number_input("R&D Spend (Rs)", min_value=0.0, step=1000.0)
admin = st.number_input("Administration Cost (Rs)", min_value=0.0, step=1000.0)
marketing = st.number_input("Marketing Spend (Rs)", min_value=0.0, step=1000.0)

# Predict button
if st.button("Predict Profit"):
    input_data = np.array([[rd_spend, admin, marketing]])
    predicted_profit = model.predict(input_data)[0]
    st.success(f"💰 Predicted Profit: ${predicted_profit:,.2f}")
