
import streamlit as st
import pickle
import pandas as pd

# Load model and column names
with open('xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('columns.pkl', 'rb') as f:
    columns = pickle.load(f)

st.markdown("<h1 style='text-align:center;'>Income Category Prediction</h1>", unsafe_allow_html=True)
st.markdown("This app predicts whether a person earns >50K or <=50K based on full demographic and employment features.")

# Input form
def user_input():
    age = st.slider('Age', 17, 90, 30)
    education_num = st.slider('Education Number', 1, 16, 9)
    hours_per_week = st.slider('Hours per Week', 1, 100, 40)
    capital_gain = st.number_input('Capital Gain', 0, 100000, 0)
    capital_loss = st.number_input('Capital Loss', 0, 5000, 0)
    fnlwgt = st.number_input('Final Weight', 10000, 1000000, 50000)

    workclass = st.selectbox('Workclass', ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Federal-gov', 'Self-emp-inc', 'Without-pay'])
    marital_status = st.selectbox('Marital Status', ['Never-married', 'Married-civ-spouse', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent'])
    occupation = st.selectbox('Occupation', ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty',
                                             'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
                                             'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
    sex = st.selectbox('Sex', ['Male', 'Female'])
    relationship = st.selectbox('Relationship', ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'])
    race = st.selectbox('Race', ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
    native_country = st.selectbox('Native Country', ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'India', 'Other'])

    data = {
        'age': age,
        'education_num': education_num,
        'hours_per_week': hours_per_week,
        'capital_gain': capital_gain,
        'capital_loss': capital_loss,
        'fnlwgt': fnlwgt,
        'workclass_' + workclass: 1,
        'marital_status_' + marital_status: 1,
        'occupation_' + occupation: 1,
        'sex_' + sex: 1,
        'relationship_' + relationship: 1,
        'race_' + race: 1,
        'native_country_' + native_country: 1
    }

    # Convert to DataFrame and align with model columns
    input_df = pd.DataFrame([data])
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[columns]
    return input_df

df = user_input()

if st.button('Predict'):
    prediction = model.predict(df)[0]
    result = '>50K' if prediction == 1 else '<=50K'
    st.success(f"Predicted Income: {result}")
