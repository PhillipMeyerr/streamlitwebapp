
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model_path = 'heart_data/heart_disease_rf_model.pkl'  # Adjust path if needed
model = joblib.load(model_path)

st.title("Heart Disease Prediction")
st.write("Enter the details of the patient to predict heart disease likelihood.")

age = st.number_input("Age", min_value=1, max_value=120, value=25)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=200, value=120)
chol = st.number_input("Serum Cholestoral in mg/dl", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "True" if x == 1 else "False")
restecg = st.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
oldpeak = st.number_input("ST depression induced by exercise relative to rest", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("The slope of the peak exercise ST segment", options=[0, 1, 2])
ca = st.selectbox("Number of major vessels colored by fluoroscopy", options=[0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia", options=[1, 2, 3])

# Create a dictionary of the inputs
input_data = {
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}

# Convert the dictionary to a DataFrame
input_df = pd.DataFrame([input_data])

# Display the input DataFrame for debugging purposes
st.write("Input Data:", input_df)

if st.button("Predict"):
    prediction = model.predict(input_df)
    prediction_prob = model.predict_proba(input_df)
    st.write(f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
    st.write(f"Prediction Probability: {prediction_prob}")