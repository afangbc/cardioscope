import streamlit as st
import numpy as np
import sklearn
import pickle

st.title("Framingham Risk Score Calculator for Cardiovascular Disease")

diabetes = st.number_input("Enter diabetes status (0 for no, 1 for yes)", value=0.0)
LDL = st.number_input("Enter LDL value (mg/dL)", value=0.0)
total_cholesterol = st.number_input("Enter total cholesterol value (mg/dL)", value=0.0)
systolic_bp = st.number_input("Enter Systolic Blood Pressure value", value=0.0)
bmi = st.number_input("Enter BMI value", value=0.0)
weight = st.number_input("Enter weight in KG", value=0.0)

filename = 'rf_reg_model.pickle'
with open(filename,'rb') as file:
    loaded_model = pickle.load(file)


X = np.array([diabetes,LDL,systolic_bp,total_cholesterol,bmi,weight]).reshape(1, -1)
y_pred = loaded_model.predict(X)[0]

if st.button("Calculate risk"):
    st.success("Framingham Risk Score (Risk of developing CVD within 10 years): "+str(y_pred)+"%.")
