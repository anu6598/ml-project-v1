# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import ollama

st.title("🚨 Human vs Bot Signup Detection")

# Load model
model = joblib.load("signup_bot_detector.pkl")

uploaded_file = st.file_uploader("Upload API Logs CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df['first_request_time'] = pd.to_datetime(df['first_request_time'])
    df['last_request_time'] = pd.to_datetime(df['last_request_time'])
    df['session_duration'] = (df['last_request_time'] - df['first_request_time']).dt.total_seconds()
    df['requests_per_minute'] = df['total_requests'] / (df['session_duration'] / 60 + 1)
    df['api_paths_called_count'] = df['api_paths_called'].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
    df['vpn_flag_count'] = df['vpn_detection_flags'].apply(lambda x: len(set(str(x).split(','))) if pd.notnull(x) else 0)
    df.fillna(0, inplace=True)

    X = df[['total_requests', 'session_duration', 'requests_per_minute', 
            'unique_user_agents', 'api_paths_called_count', 'vpn_flag_count']]
    
    df['prediction'] = model.predict(X)
    df['prediction_label'] = df['prediction'].map({0: 'Human', 1: 'Bot'})
    
    st.dataframe(df[['x_real_ip', 'total_requests', 'session_duration', 'prediction_label']])

    # Chatbot interface (with Ollama)
    st.subheader("🧠 Ask your data anything")
    user_input = st.text_input("Ask something like: 'Which IPs had the most OTP requests today?'")

    if user_input:
        # Chat with the data (via Ollama)
        response = ollama.chat(
            model='llama3',
            messages=[
                {"role": "system", "content": "You are a data analyst. You answer based on the CSV data provided."},
                {"role": "user", "content": f"{user_input}\nHere is the CSV data:\n{df.to_csv(index=False)}"}
            ]
        )
        st.write(response['message']['content'])
