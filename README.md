
# Income Category Prediction App

This Streamlit web app predicts whether an individual earns more than 50K or not based on various personal and employment features. It uses a pre-trained XGBoost classifier.

## 📊 Features Used
- Age
- Education Number
- Hours per Week
- Capital Gain
- Capital Loss
- Final Weight

## 🧠 Model
The model used is an XGBoost classifier trained on the Adult Income dataset from UCI (via Kaggle). It was trained after preprocessing, encoding, and cleaning the data.

## 🚀 How to Run

1. Clone this repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## 📁 Files in the Repo

- `app.py`: Main Streamlit app
- `xgboost_model.pkl`: Trained model
- `columns.pkl`: Columns expected by the model
- `requirements.txt`: Python dependencies

## 📚 Dataset

Dataset Source: [UCI Adult Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/adult-census-income)

## ✨ Output Example

The app will display whether the input profile results in a predicted income of `<=50K` or `>50K`.

---
Made for Credit Team 8.
