import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load trained model
model = joblib.load("model.joblib")

st.set_page_config(page_title="Salary Prediction", layout="centered")
st.title("💼 Salary Prediction App")

st.write("Write the job info, to See the pridicted Salary.")

col1, col2 = st.columns(2)

with col1:
    rating = st.number_input("Company Rating", 0.0, 5.0, 3.5)
    age = st.number_input("Company Age", 0, 200, 10)
    job_state = st.selectbox("Job State", ["CA", "NY", "TX", "IL", "FL"])

with col2:
    python_yn = st.checkbox("Python?")
    r_yn = st.checkbox("R?")
    spark = st.checkbox("Spark?")
    aws = st.checkbox("AWS?")
    excel = st.checkbox("Excel?")

if st.button("Predict Salary"):
    data = pd.DataFrame([{
        "Rating": rating,
        "age": age,
        "job_state": job_state,
        "python_yn": int(python_yn),
        "R_yn": int(r_yn),
        "spark": int(spark),
        "aws": int(aws),
        "excel": int(excel)
    }])

    prediction = model.predict(data)[0]

    st.success(f"💰 Estimated Salary: **${prediction:,.0f}**")
