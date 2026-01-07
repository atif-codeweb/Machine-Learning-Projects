import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('loan_model.pkl')
sc = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
feature_names = joblib.load('feature_names.pkl')

# Create mapping dictionaries for display
gender_map = {0: 'Male', 1: 'Female'}
married_map = {0: 'No', 1: 'Yes'}
education_map = {0: 'Graduate', 1: 'Not Graduate'}
self_employed_map = {0: 'No', 1: 'Yes'}
property_map = {0: 'Rural', 1: 'Semiurban', 2: 'Urban'}
status_map = {0: 'Rejected', 1: 'Approved'}

st.title('Loan Eligibility Prediction')

with st.form("loan_form"):
    st.header("Applicant Information")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.radio("Gender", options=list(gender_map.keys()), 
                         format_func=lambda x: gender_map[x])
        married = st.radio("Married", options=list(married_map.keys()),
                         format_func=lambda x: married_map[x])
        education = st.radio("Education", options=list(education_map.keys()),
                           format_func=lambda x: education_map[x])
    with col2:
        self_employed = st.radio("Self Employed", options=list(self_employed_map.keys()),
                               format_func=lambda x: self_employed_map[x])
        dependents = st.selectbox("Dependents", [0, 1, 2, 3])
        credit_history = st.radio("Credit History", [0, 1],
                                format_func=lambda x: "Good" if x == 1 else "Bad")

    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.selectbox("Loan Term (months)", [360, 120, 240, 60, 180, 300, 480, 84, 12])
    property_area = st.selectbox("Property Area", options=list(property_map.keys()),
                               format_func=lambda x: property_map[x])

    submitted = st.form_submit_button("Predict")
if submitted:

    input_data = np.array([[
        0,  
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        property_area
    ]])
    scaled_input = sc.transform(input_data)

    # Predict
    proba = model.predict_proba(scaled_input)[0][1]
    prediction = int(proba >= 0.7)

    # Display results
    st.subheader("Loan Decision")
    if prediction == 1:
        st.success(f"{status_map[prediction]} (Probability: {proba*100:.1f}%)")
        st.balloons()
    else:
        st.error(f"{status_map[prediction]} (Probability: {proba*100:.1f}%)")


    st.progress(proba)
    st.metric("Confidence Score", f"{proba*100:.1f}%")

print("Streamlit app created! Run with: streamlit run loan_app.py")
