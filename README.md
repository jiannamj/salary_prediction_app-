# 💼 Employee Salary Prediction App

This is a Streamlit web app that predicts whether an employee earns more than $50K or not based on input features.

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` – main Streamlit app
- `*.pkl` – model and preprocessing files
- `*.csv` – datasets
- `employee_salary_prediction.ipynb` – training and preprocessing notebook

## Features
- Clean user interface with Streamlit
- Input employee details via sidebar
- Income prediction: >50K or <=50K
- Visualizations:
  - Capital Gain/Loss bar chart
  - Age boxplot
  - Hours per week histogram
