import streamlit as st
import pandas as pd
import joblib
import os

def load_model(model_path='suspicious_model.pkl'):
    if not os.path.exists(model_path):
        st.error("Model file not found! Please train the model first.")
        return None
    return joblib.load(model_path)

def preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['user_id'] = df['user_id'].astype(str).str.strip().str.lower()
    
    features = df.groupby('user_id').agg({
        'actual_hours': 'sum',
        '_pause': 'sum',
        '_seek': 'sum',
        'lesson_id': 'nunique',
        '_d_id': 'nunique'
    }).rename(columns={'lesson_id': 'unique_lessons', '_d_id': 'unique_devices'})
    
    return features

def predict_suspicious_users(features, model):
    if model is None:
        return None
    
    features['is_predicted_suspicious'] = model.predict(features)
    suspicious_users = features[features['is_predicted_suspicious'] == 1]
    suspicious_users = suspicious_users.reset_index()
    
    if len(suspicious_users) > 50:
        suspicious_users = suspicious_users.nlargest(50, 'actual_hours')
    
    return suspicious_users[['user_id', 'actual_hours']]

# Streamlit UI
st.set_page_config(page_title="Suspicious User Detector", layout="wide")
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #1E3C72, #2A5298);
        color: white;
    }
    .stApp {
        background: transparent;
    }
    .stTextInput, .stFileUploader, .stButton, .stDataFrame, .stTable {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔍 Suspicious User Detector")
st.write("Upload a CSV file to detect the top 50 suspicious users for that day.")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    model = load_model()
    features = preprocess_data(uploaded_file)
    suspicious_users = predict_suspicious_users(features, model)
    
    if suspicious_users is not None and not suspicious_users.empty:
        st.success(f"Top {len(suspicious_users)} suspicious users detected!")
        st.dataframe(suspicious_users)
    else:
        st.warning("No suspicious users found for this file.")
