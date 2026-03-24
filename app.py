import streamlit as st
import pandas as pd
import joblib

# Load the saved brains from your notebook
model = joblib.load('credit_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("🏦 SmartBank: Credit Risk AI")
st.write("Determine loan eligibility instantly using Machine Learning.")

# Create input fields based on your Credit_Data.csv
duration = st.number_input("Duration (Months)", 1, 72, 12)
amount = st.number_input("Credit Amount", 100, 20000, 2000)
age = st.slider("Age", 18, 90, 30)
checking = st.selectbox("Checking Account", ["< 0 DM", "0 <= ... < 200 DM", "no checking account"])

if st.button("Predict Risk"):
    # 1. Create a row of data
    input_df = pd.DataFrame([{
        'duration_in_month': duration,
        'credit_amount': amount,
        'age': age,
        'account_check_status': checking
    }])
    
    # 2. Match the encoding from the notebook
    input_encoded = pd.get_dummies(input_df)
    
    # Fill in missing columns with 0s to match the training set
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    input_encoded = input_encoded[model_columns] # Sort columns correctly
    
    # 3. Scale and Predict
    scaled_data = scaler.transform(input_encoded)
    prob = model.predict_proba(scaled_data)[0][1]
    
    # 4. Display Result
    if prob >= 0.34:
        st.error(f"High Risk Detected: {prob:.2%} Probability of Default")
    else:
        st.success(f"Low Risk: {prob:.2%} Probability of Default. Recommended for Approval.")