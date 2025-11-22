import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load(r"C:\Users\ACER\Desktop\pythontutorial\abrar\venv\car_price_model.pkl")


st.set_page_config(page_title="Car Price Predictor", layout="wide")

st.title("🚗 Car Price Prediction App")
st.write("Enter car details below to estimate its price.")

# ----------------------------
# UI Inputs
# ----------------------------

# These must match the features used in training
features = ["Make", "Model", "Variant", "Displacement", "Cylinders", "Valves_Per_Cylinder"]

col1, col2 = st.columns(2)

with col1:
    Make = st.text_input("Make (e.g., Maruti, Hyundai)")
    Model = st.text_input("Model (e.g., Swift, Creta)")
    Variant = st.text_input("Variant (e.g., LXI, VXI)")

with col2:
    Displacement = st.number_input("Displacement (cc)", min_value=0, step=1)
    Cylinders = st.number_input("Cylinders", min_value=0, step=1)
    Valves_Per_Cylinder = st.number_input("Valves Per Cylinder", min_value=0, step=1)

# ----------------------------
# Predict Button
# ----------------------------

if st.button("Predict Price"):
    # Prepare data in DataFrame
    input_data = pd.DataFrame([{
        "Make": Make,
        "Model": Model,
        "Variant": Variant,
        "Displacement": Displacement,
        "Cylinders": Cylinders,
        "Valves_Per_Cylinder": Valves_Per_Cylinder
    }])

    # Predict
    prediction = model.predict(input_data)[0]

    st.success(f"💰 Estimated Price: **₹ {prediction:,.2f}**")

