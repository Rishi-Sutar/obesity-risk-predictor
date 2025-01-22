import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from sklearn.preprocessing import StandardScaler

scoring_uri = st.secrets["scoring_uri"]

st.title('Obesity Predictor')

st.header('Please enter the following information to predict obesity')

col1, col2 = st.columns(2)

with st.form("input_form"):
    with col1:
        Gender = st.radio('Select Gender', options=['Male', 'Female'])
        family_history_with_overweight = st.radio('Family History With Overweight', ['yes', 'no'])
        FAVC = st.radio('Frequent consumption of high-caloric food (FAVC)', ['yes', 'no'])
        FCVC = st.number_input('Frequency of consumption of vegetables (FCVC)', min_value=0.0, max_value=3.0, value=0.0, step=0.1)  # Set float value
        NCP = st.number_input('Number of main meals (NCP)', min_value=0.0, max_value=3.0, value=0.0, step=0.1)  # Set float value
        CALC = st.radio('Consumption of alcohol (CALC)', ['no', 'Sometimes', 'Frequently', 'Always'])
        MTRANS = st.radio('Mode of transportation (MTRANS)', ['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'])
        
    with col2:    
        Age = st.number_input('Age', min_value=0.0, max_value=100.0, value=0.0, step=0.1)  # Set float value
        Height = st.number_input('Height (cm)', min_value=0.0, max_value=300.0, value=0.0, step=0.1)  # Set float value
        Weight = st.number_input('Weight (kg)', min_value=0.0, max_value=300.0, value=0.0, step=0.1)  # Set float value
        CAEC = st.radio('Consumption of food between meals (CAEC)', ['no', 'Sometimes', 'Frequently', 'Always'])
        SMOKE = st.radio('Smoke', ['yes', 'no'])
        CH2O = st.number_input('Daily water consumption (CH2O)', min_value=0.0, max_value=3.0, value=0.0, step=0.1)  # Set float value
        SCC = st.radio('Caloric beverages consumption (SCC)', ['yes', 'no'])
        FAF = st.number_input('Physical activity frequency (FAF)', min_value=0.0, max_value=3.0, value=0.0, step=0.1)  # Set float value
        TUE = st.number_input('Time spent using technological devices (TUE)', min_value=0.0, max_value=3.0, value=0.0, step=0.1)  # Set float value

    submitted = st.form_submit_button("Submit")

# Create a dictionary to store input data
input_data = {
    'Gender': [Gender],
    'Age': [Age],
    'Height': [Height],
    'Weight': [Weight],
    'family_history_with_overweight': [family_history_with_overweight],
    'FAVC': [FAVC],
    'FCVC': [FCVC],
    'NCP': [NCP],
    'CAEC': [CAEC],
    'SMOKE': [SMOKE],
    'CH2O': [CH2O],
    'SCC': [SCC],
    'FAF': [FAF],
    'TUE': [TUE],
    'CALC': [CALC],
    'MTRANS': [MTRANS]
}

obesity_levels = {
    0: 'Normal_Weight', 
    1: 'Insufficient_Weight', 
    2: 'Obesity_Type_I', 
    3: 'Obesity_Type_II', 
    4: 'Obesity_Type_III', 
    5: 'Overweight_Level_I', 
    6: 'Overweight_Level_II'
}


# Create a DataFrame from input data
df = pd.DataFrame(input_data)

# Perform label encoding on Gender
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['family_history_with_overweight'] = df['family_history_with_overweight'].map({'yes': 1, 'no': 0}).astype(float)
df['FAVC'] = df['FAVC'].map({'yes': 1, 'no': 0}).astype(float)
df['SMOKE'] = df['SMOKE'].map({'yes': 1, 'no': 0}).astype(float)
df['SCC'] = df['SCC'].map({'yes': 1, 'no': 0}).astype(float)
df['CAEC'] = df['CAEC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}).astype(float)
df['CALC'] = df['CALC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}).astype(float)
df['MTRANS'] = df['MTRANS'].map({'Automobile': 0, 'Motorbike': 1, 'Bike': 2, 'Public_Transportation': 3, 'Walking': 4}).astype(float)

# Perform standard scaling on Height, FCVC, CH2O, FAF, TUE, BMI
scaler = StandardScaler()
df[['Age', 'Height', 'Weight', 'family_history_with_overweight','FAVC', 'FCVC', 'NCP', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE']] = scaler.fit_transform(df[['Age', 'Height', 'Weight', 'family_history_with_overweight','FAVC', 'FCVC', 'NCP', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE']])

# Convert DataFrame to JSON
data = {
    "data" : df.values.tolist()
}

input_data = json.dumps(data)

# Send JSON data to Azure Machine Learning endpoint
if submitted:
    headers = {'Content-Type': 'application/json'}

    # Send data to Azure Machine Learning endpoint
    response = requests.post(scoring_uri, data=input_data, headers=headers)
    
    # Check if the response was successful
    if response.status_code == 200:
        result = json.loads(response.json())
        res = result['result'][0]
        st.write("Prediction:", obesity_levels[res])
    else:
        st.write("Error:", response.text)
