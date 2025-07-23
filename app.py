import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and encoders
model = joblib.load("salary_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_names = joblib.load("feature_names.pkl")

# Streamlit UI setup
st.set_page_config(page_title="Employee Salary Prediction", layout="centered")
st.title("💼 Employee Salary Prediction")
st.markdown("---")

# Sidebar for user input
st.sidebar.header("Input Employee Details")
input_dict = {}

for feature in feature_names:
    display_name = feature.replace("_", " ").title()
    if feature in label_encoders:
        input_dict[feature] = st.sidebar.selectbox(display_name, label_encoders[feature].classes_)
    else:
        input_dict[feature] = st.sidebar.number_input(display_name, step=1.0)

# Encode inputs
for feature in label_encoders:
    if feature in input_dict:
        input_dict[feature] = label_encoders[feature].transform([input_dict[feature]])[0]

# Prepare dataframe
input_df = pd.DataFrame([input_dict])
scaled_input = scaler.transform(input_df)

# Predict class
prediction = model.predict(scaled_input)[0]
prediction_label = ">50K" if prediction == 1 else "<=50K"

# Show result
st.subheader("💰 Predicted Income Category:")
st.success(f"Predicted Income: {prediction_label}")
st.markdown("---")

# Visualization
st.subheader("📊 User-Specific Visualization")

# Capital gain/loss
fig1, ax1 = plt.subplots(figsize=(4, 3))
sns.barplot(x=["Capital Gain", "Capital Loss"], y=[
    input_dict.get("capital_gain", 0),
    input_dict.get("capital_loss", 0)
], ax=ax1)
ax1.set_ylabel("Amount")
st.pyplot(fig1)

# Hours per week
fig2, ax2 = plt.subplots(figsize=(4, 3))
sns.histplot(x=[input_dict.get("hours_per_week", 0)], bins=10, kde=True, ax=ax2)
ax2.set_title("Hours Per Week")
st.pyplot(fig2)

# Age boxplot
fig3, ax3 = plt.subplots(figsize=(4, 3))
sns.boxplot(x=["Age"], y=[input_dict.get("age", 0)], ax=ax3)
ax3.set_title("Age")
st.pyplot(fig3)

st.markdown("---")
st.caption("🔁 Reload the page to test with new input.")
