import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
 
# ---------- Load saved model, scaler, encoders, feature list ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")
    with open("feature_columns.json", "r") as f:
        feature_columns = json.load(f)
    return model, scaler, encoders, feature_columns
 
model, scaler, encoders, feature_columns = load_artifacts()
 
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Risk Predictor")
st.write("Enter your health and lifestyle details below, and the model will predict your diabetes risk.")
 
# ---------- Categorical options (from your LabelEncoders) ----------
gender_options = list(encoders["gender"].classes_)
ethnicity_options = list(encoders["ethnicity"].classes_)
education_options = list(encoders["education_level"].classes_)
income_options = list(encoders["income_level"].classes_)
employment_options = list(encoders["employment_status"].classes_)
smoking_options = list(encoders["smoking_status"].classes_)
 
# ---------- Input form ----------
with st.form("prediction_form"):
    st.subheader("Demographics")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=40)
        gender = st.selectbox("Gender", gender_options)
        ethnicity = st.selectbox("Ethnicity", ethnicity_options)
    with col2:
        education_level = st.selectbox("Education Level", education_options)
        income_level = st.selectbox("Income Level", income_options)
        employment_status = st.selectbox("Employment Status", employment_options)
 
    st.subheader("Lifestyle")
    col3, col4 = st.columns(2)
    with col3:
        smoking_status = st.selectbox("Smoking Status", smoking_options)
        alcohol_consumption_per_week = st.number_input("Alcohol drinks per week", min_value=0, max_value=50, value=1)
        physical_activity_minutes_per_week = st.number_input("Physical activity (minutes/week)", min_value=0, max_value=2000, value=100)
    with col4:
        diet_score = st.slider("Diet Score (0-10)", 0.0, 10.0, 6.0)
        sleep_hours_per_day = st.slider("Sleep hours/day", 0.0, 14.0, 7.0)
        screen_time_hours_per_day = st.slider("Screen time hours/day", 0.0, 20.0, 6.0)
 
    st.subheader("Medical History")
    col5, col6, col7 = st.columns(3)
    with col5:
        family_history_diabetes = st.selectbox("Family history of diabetes?", ["No", "Yes"])
    with col6:
        hypertension_history = st.selectbox("Hypertension history?", ["No", "Yes"])
    with col7:
        cardiovascular_history = st.selectbox("Cardiovascular history?", ["No", "Yes"])
 
    st.subheader("Body Measurements")
    col8, col9 = st.columns(2)
    with col8:
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0)
        waist_to_hip_ratio = st.number_input("Waist-to-hip ratio", min_value=0.5, max_value=1.5, value=0.85)
        systolic_bp = st.number_input("Systolic BP", min_value=70, max_value=220, value=120)
        diastolic_bp = st.number_input("Diastolic BP", min_value=40, max_value=140, value=80)
    with col9:
        heart_rate = st.number_input("Heart rate (bpm)", min_value=40, max_value=180, value=75)
        cholesterol_total = st.number_input("Total cholesterol", min_value=100, max_value=400, value=190)
        hdl_cholesterol = st.number_input("HDL cholesterol", min_value=10, max_value=120, value=50)
        ldl_cholesterol = st.number_input("LDL cholesterol", min_value=30, max_value=300, value=110)
 
    st.subheader("Blood Test Values")
    col10, col11 = st.columns(2)
    with col10:
        triglycerides = st.number_input("Triglycerides", min_value=30, max_value=500, value=130)
        glucose_fasting = st.number_input("Fasting glucose", min_value=50, max_value=400, value=100)
        glucose_postprandial = st.number_input("Postprandial glucose", min_value=50, max_value=500, value=140)
    with col11:
        insulin_level = st.number_input("Insulin level", min_value=0.0, max_value=100.0, value=8.0)
        hba1c = st.number_input("HbA1c", min_value=3.0, max_value=15.0, value=5.5)
        diabetes_risk_score = st.number_input("Diabetes risk score (if known, else leave default)", min_value=0.0, max_value=100.0, value=25.0)
 
    submitted = st.form_submit_button("Predict")
 
# ---------- Prediction ----------
if submitted:
    yn_map = {"No": 0, "Yes": 1}
 
    raw_input = {
        "age": age,
        "gender": encoders["gender"].transform([gender])[0],
        "ethnicity": encoders["ethnicity"].transform([ethnicity])[0],
        "education_level": encoders["education_level"].transform([education_level])[0],
        "income_level": encoders["income_level"].transform([income_level])[0],
        "employment_status": encoders["employment_status"].transform([employment_status])[0],
        "smoking_status": encoders["smoking_status"].transform([smoking_status])[0],
        "alcohol_consumption_per_week": alcohol_consumption_per_week,
        "physical_activity_minutes_per_week": physical_activity_minutes_per_week,
        "diet_score": diet_score,
        "sleep_hours_per_day": sleep_hours_per_day,
        "screen_time_hours_per_day": screen_time_hours_per_day,
        "family_history_diabetes": yn_map[family_history_diabetes],
        "hypertension_history": yn_map[hypertension_history],
        "cardiovascular_history": yn_map[cardiovascular_history],
        "bmi": bmi,
        "waist_to_hip_ratio": waist_to_hip_ratio,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "heart_rate": heart_rate,
        "cholesterol_total": cholesterol_total,
        "hdl_cholesterol": hdl_cholesterol,
        "ldl_cholesterol": ldl_cholesterol,
        "triglycerides": triglycerides,
        "glucose_fasting": glucose_fasting,
        "glucose_postprandial": glucose_postprandial,
        "insulin_level": insulin_level,
        "hba1c": hba1c,
        "diabetes_risk_score": diabetes_risk_score,
    }
 
    # Build dataframe in the exact column order the model was trained on
    input_df = pd.DataFrame([raw_input])
    missing_cols = [c for c in feature_columns if c not in input_df.columns]
    if missing_cols:
        st.error(f"These columns are missing from the app — please check feature_columns.json: {missing_cols}")
    else:
        input_df = input_df[feature_columns]
        input_scaled = scaler.transform(input_df)
 
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
 
        st.divider()
        if prediction == 1:
            st.error(f"⚠️ Model predicts: **Diabetic risk detected** (probability: {probability:.1%})")
        else:
            st.success(f"✅ Model predicts: **No diabetes indicated** (probability of diabetes: {probability:.1%})")
 
        st.caption("Disclaimer: This tool is for educational/demo purposes only and is not a medical diagnosis. Please consult a doctor for any health concerns.")
