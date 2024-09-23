"""
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Title of the app
st.title('Bankruptcy Prediction App Using Random Forest')

# Sidebar for user inputs
st.sidebar.header('Input Company Risk Factors')

# Function to take user inputs
def user_input_features():
    industrial_risk = st.sidebar.slider('Industrial Risk', 0.0, 1.0, 0.5, step=0.1)
    management_risk = st.sidebar.slider('Management Risk', 0.0, 1.0, 0.5, step=0.1)
    financial_flexibility = st.sidebar.slider('Financial Flexibility', 0.0, 1.0, 0.5, step=0.1)
    credibility = st.sidebar.slider('Credibility', 0.0, 1.0, 0.5, step=0.1)
    competitiveness = st.sidebar.slider('Competitiveness', 0.0, 1.0, 0.5, step=0.1)
    operating_risk = st.sidebar.slider('Operating Risk', 0.0, 1.0, 0.5, step=0.1)
    
    data = {
        'industrial_risk': industrial_risk,
        'management_risk': management_risk,
        'financial_flexibility': financial_flexibility,
        'credibility': credibility,
        'competitiveness': competitiveness,
        'operating_risk': operating_risk
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

# Train the model (Random Forest)
model = RandomForestClassifier()
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
