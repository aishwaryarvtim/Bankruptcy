"""
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Title of the app
st.title('Bankruptcy Prediction App Using Logistic Regression')

# Sidebar for user inputs
st.sidebar.header('Input Company Risk Factors')

# Function to take user inputs
def user_input_features():
    industrial_risk = st.sidebar.selectbox('Industrial Risk', ['Low Risk', 'Medium Risk', 'High Risk'], index=1)
    management_risk = st.sidebar.selectbox('Management Risk', ['Low Risk', 'Medium Risk', 'High Risk'], index=1)
    financial_flexibility = st.sidebar.selectbox('Financial Flexibility', ['Low Risk', 'Medium Risk', 'High Risk'], index=1)
    credibility = st.sidebar.selectbox('Credibility', ['Low Risk', 'Medium Risk', 'High Risk'], index=1)
    competitiveness = st.sidebar.selectbox('Competitiveness', ['Low Risk', 'Medium Risk', 'High Risk'], index=1)
    operating_risk = st.sidebar.selectbox('Operating Risk', ['Low Risk', 'Medium Risk', 'High Risk'], index=1)

    # Convert categorical inputs into numerical values (for model compatibility)
    risk_mapping = {'Low Risk': 0.0, 'Medium Risk': 0.5, 'High Risk': 1.0}
    
    data = {
        'industrial_risk': risk_mapping[industrial_risk],
        'management_risk': risk_mapping[management_risk],
        'financial_flexibility': risk_mapping[financial_flexibility],
        'credibility': risk_mapping[credibility],
        'competitiveness': risk_mapping[competitiveness],
        'operating_risk': risk_mapping[operating_risk]
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Display user input parameters
st.subheader('User Input Parameters')
st.write(input_df)

# For demonstration, training a new model
# Load dataset
df = pd.read_excel('bankruptcy-prevention.xlsx')

# Preprocess data
X = df.drop('class', axis=1)
y = df['class']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model (Logistic Regression)
model = LogisticRegression()
model.fit(X_scaled, y)

# Scale the input data
input_scaled = scaler.transform(input_df)

# Make prediction
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# Display prediction results
st.subheader('Prediction')
if prediction[0] == 'non-bankruptcy':
    st.write('Non-Bankrupt')
else:
    st.write('Bankrupt')

st.subheader('Prediction Probability')
st.write(prediction_proba)
