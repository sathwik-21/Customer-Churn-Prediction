import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# Load model (Keras format recommended)
model = tf.keras.models.load_model("model.keras")

# Load encoders/scaler
with open('ohe_geography.pkl', 'rb') as f:
    ohe_geography = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_gen = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Customer Churn Prediction")

# UI Inputs
geography = st.selectbox("Geography", ohe_geography.categories_[0])
gender = st.selectbox("Gender", label_gen.classes_)
credit_score = st.number_input("Credit Score")
age = st.slider("Age", 18, 100)
tenure = st.slider("Tenure", 0, 10)
balance = st.number_input("Balance")
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card?", [0, 1])
is_active_member = st.selectbox("Is Active Member?", [0, 1])
estimated_salary = st.number_input("Estimated Salary")

# Create raw dataframe (without geography)
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_gen.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

# One-hot encode geography
geo_encoded = ohe_geography.transform([[geography]])
geo_df = pd.DataFrame(
    geo_encoded,
    columns=ohe_geography.get_feature_names_out()
)

# Combine all features
final_input = pd.concat([input_data, geo_df], axis=1)

# Scale with the same scaler used during training
final_scaled = scaler.transform(final_input)

# Predict
prediction = model.predict(final_scaled)[0][0]

# Output
if prediction > 0.5:
    st.error("🚨 The customer is likely to CHURN!")
else:
    st.success("✅ The customer is NOT likely to churn.")
