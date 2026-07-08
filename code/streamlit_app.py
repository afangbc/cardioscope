import streamlit as st
import numpy as np
import sklearn
import pickle

# Set page tab configuration
st.set_page_config(page_title="CVD Risk Calculator", page_icon="❤️", layout="centered")

st.title("❤️ Framingham Risk Score Calculator")
st.write("Predict your 10-year risk of developing cardiovascular disease (CVD).")
st.markdown("---")

# Organize inputs into two columns for a much cleaner layout
col1, col2 = st.columns(2)

with col1:
    systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=0.0, value=120.0, step=1.0)
    total_cholesterol = st.number_input("Total Cholesterol (mg/dL)", min_value=0.0, value=200.0, step=1.0)
    LDL = st.number_input("LDL Cholesterol (mg/dL)", min_value=0.0, value=100.0, step=1.0)

with col2:
    bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, value=22.0, step=0.1)
    weight = st.number_input("Weight (kg)", min_value=0.0, value=70.0, step=0.1)
    diabetes = st.selectbox("Diabetes Status", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

st.markdown("---")

# Load the trained Random Forest model
filename = 'rf_reg_model.pickle'
try:
    with open(filename, 'rb') as file:
        loaded_model = pickle.load(file)
except FileNotFoundError:
    st.error(f"Could not find '{filename}'. Make sure it's in the same folder as this script!")
    st.stop()

# Trigger calculation on button click
if st.button("Calculate My Risk Score", type="primary"):
    
    # Format inputs for prediction
    X = np.array([diabetes, LDL, systolic_bp, total_cholesterol, bmi, weight]).reshape(1, -1)
    
    # Get prediction and round it for clean presentation
    y_pred = round(float(loaded_model.predict(X)[0]), 2)
    
    st.subheader("Your Results")
    
    # Categorize risk based on medical standards (Low <10%, Intermediate 10-20%, High >20%)
    if y_pred < 10.0:
        st.success(f"### Low Risk: {y_pred}%")
        st.write("✨ Your calculated risk is within a healthy, low range. Keep maintaining a balanced diet and regular exercise routine!")
        
    elif 10.0 <= y_pred <= 20.0:
        st.warning(f"### Intermediate Risk: {y_pred}%")
        st.write("⚠️ Your risk score is moderate. It is a good idea to monitor your health markers and discuss lifestyle adjustments with a healthcare professional.")
        
    else:
        st.error(f"### High Risk: {y_pred}%")
        st.write("🚨 **Critical Notice:** Your calculated risk factor is high. **We strongly recommend scheduling an appointment with a doctor or cardiologist** to review these stats and evaluate preventive measures.")

    # Mandatory medical disclaimer
    st.caption("ℹ️ *Disclaimer: This tool is for educational tracking purposes only and does not substitute for professional medical diagnoses or consultations.*")