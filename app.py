import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model_path = 'heart_disease_model.pkl'
model = joblib.load(model_path)

st.title("Heart Disease Prediction")
st.write("Enter the details of the patient to predict heart disease likelihood.")

# Input fields for user to enter data
age = st.number_input("Age", min_value=1, max_value=120, value=25)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox("Chest Pain Type", options=["typical angina","atypical angina", "non-angina pain","asyptomatic"])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol in mg/dl", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "True" if x == 1 else "False")
restecg = st.selectbox("Resting Electrocardiographic Results", options=["normal", "abnormal", "ventricular hypertrophy"])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
oldpeak = st.number_input("ST depression induced by exercise relative to rest", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("The slope of the peak exercise ST segment", options=["upsloping","flat","downsploping"])
ca = st.selectbox("Number of major vessels colored by fluoroscopy", options=[0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia", options=["normal","fixed defect","reversable defect", "unknown"])

# Create a dictionary of the inputs
input_data = {
    'age': age,
    'trestbps': trestbps,
    'chol': chol,
    'thalach': thalach,
    'oldpeak': oldpeak,
    'sex_1': sex,
    'cp_1': 1 if cp == 1 else 0,
    'cp_2': 1 if cp == 2 else 0,
    'cp_3': 1 if cp == 3 else 0,
    'fbs_1': fbs,
    'restecg_1': 1 if restecg == 1 else 0,
    'restecg_2': 1 if restecg == 2 else 0,
    'exang_1': exang,
    'slope_1': 1 if slope == 1 else 0,
    'slope_2': 1 if slope == 2 else 0,
    'ca_1': 1 if ca == 1 else 0,
    'ca_2': 1 if ca == 2 else 0,
    'ca_3': 1 if ca == 3 else 0,
    'ca_4': 1 if ca == 4 else 0,
    'thal_1': 1 if thal == 1 else 0,
    'thal_2': 1 if thal == 2 else 0,
    'thal_3': 1 if thal == 3 else 0
}

# Convert the dictionary to a DataFrame
input_df = pd.DataFrame([input_data])

# Display the input DataFrame for debugging purposes
st.write("Input Data:", input_df)

# Make a prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    prediction_prob = model.predict_proba(input_df)
    st.write(f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")


