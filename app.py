import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sklearn
import numpy as np 
from sklearn.ensemble import RandomForestClassifier


def load_ml_model():
    try:
        with open(f'model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading machine learning model: {e}")
        return None

def predict_with_model(model, data):
    return model.predict(data)

st.title("DDoS Attack Detection Using Ensemble Learning")

model = load_ml_model()
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    csv_data = pd.read_csv(uploaded_file)
    st.write(csv_data.head())
    csv_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    csv_data = csv_data.apply(pd.to_numeric, errors='coerce')

    max_value = csv_data.values.max()
    if max_value > np.finfo(np.float32).max:
        st.warning(f"Input data contains large values. Max value found: {max_value}")
        csv_data = csv_data / max_value * np.finfo(np.float32).max

    mean_values = csv_data.mean()
    csv_data.fillna(mean_values, inplace=True)
    predictions = predict_with_model(model, csv_data)

    if predictions is not None:
        st.write("Predictions:")
        unique_labels, label_counts = np.unique(predictions, return_counts=True)
        total_predictions = len(predictions)
        percentages = (label_counts / total_predictions) * 100
        df_percentages = pd.DataFrame({ 'Attack Type': unique_labels,'Percentage': percentages})
        df_percentages = df_percentages[df_percentages['Percentage'] > 0]
        st.write("Attack Type Percentages:")
        st.write(df_percentages)

        fig, ax = plt.subplots()
        ax.pie(df_percentages['Percentage'], labels=df_percentages['Attack Type'], autopct='%1.1f%%', shadow=True, startangle=90)
        ax.axis('equal')  
        st.pyplot(fig)
