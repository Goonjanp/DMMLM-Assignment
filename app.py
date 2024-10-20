import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained model
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create Streamlit app
st.title("Loan Approval Prediction")

# Create input fields for user
employment_status = st.selectbox("Employment Status", ['Employed', 'Unemployed', 'Self-employed'])
education_level = st.selectbox("Education Level", ['Graduate', 'Undergraduate', 'High School'])
marital_status = st.selectbox("Marital Status", ['Married', 'Single', 'Divorced'])
home_ownership_status = st.selectbox("Home Ownership Status", ['Owned', 'Rented', 'Mortgaged'])
loan_purpose = st.selectbox("Loan Purpose", ['Home Improvement', 'Debt Consolidation', 'Business'])
loan_amount = st.number_input("Loan Amount", min_value=0)
income = st.number_input("Income", min_value=0)
credit_history = st.number_input("Credit History", min_value=0)

# Create a button to predict
if st.button("Predict"):
    # Create a DataFrame with user input
    input_data = pd.DataFrame({
        'EmploymentStatus': [employment_status],
        'EducationLevel': [education_level],
        'MaritalStatus': [marital_status],
        'HomeOwnershipStatus': [home_ownership_status],
        'LoanPurpose': [loan_purpose],
        'LoanAmount': [loan_amount],
        'Income': [income],
        'CreditHistory': [credit_history],
    })

    # Encode categorical features using LabelEncoder
    le = LabelEncoder()
    for column in ['EmploymentStatus', 'EducationLevel', 'MaritalStatus', 'HomeOwnershipStatus', 'LoanPurpose']:
        input_data[column] = le.fit_transform(input_data[column])


    # Make prediction using the loaded model
    prediction = model.predict(input_data)

    # Display the prediction
    if prediction[0] == 1:
        st.success("Loan Approved!")
    else:
        st.error("Loan Not Approved.")
