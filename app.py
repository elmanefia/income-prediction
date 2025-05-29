
import streamlit as st
import pickle
import pandas as pd

# Load model and column names
with open('xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('columns.pkl', 'rb') as f:
    columns = pickle.load(f)

# App title
st.markdown("<h1 style='text-align:center;'>Income Category Prediction</h1>", unsafe_allow_html=True)
st.markdown("This app predicts whether a person earns >50K or <=50K based on input features.")

# Input form
def user_input():
    age = st.slider('Age', 17, 90, 30)
    education_num = st.slider('Education Number', 1, 16, 9)
    hours_per_week = st.slider('Hours per Week', 1, 100, 40)
    capital_gain = st.number_input('Capital Gain', 0, 100000, 0)
    capital_loss = st.number_input('Capital Loss', 0, 5000, 0)
    final_weight = st.number_input('Final Weight', 10000, 1000000, 50000)

    data = {
        'age': age,
        'education_num': education_num,
        'hours_per_week': hours_per_week,
        'capital_gain': capital_gain,
        'capital_loss': capital_loss,
        'fnlwgt': final_weight
    }

    # Create DataFrame and align with training columns
    input_df = pd.DataFrame([data])
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[columns]
    return input_df

df = user_input()

# Prediction
if st.button('Predict'):
    prediction = model.predict(df)[0]
    result = '>50K' if prediction == 1 else '<=50K'
    st.success(f"Predicted Income: {result}")
