# signup_spike_detector.py

import pandas as pd
import streamlit as st
import ollama
from sklearn.ensemble import IsolationForest
from datetime import datetime

# 1. Upload CSV API Logs
st.title("Signup Spike & Bot Detection Tool")
uploaded_file = st.file_uploader("Upload your API Logs CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.write(df.head())

    # 2. Basic Preprocessing - assuming a column named 'timestamp' and 'user_id'
    if 'timestamp' in df.columns:
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['date'] = df['start_time'].dt.date
        signup_counts = df.groupby('date').size().reset_index(name='signup_count')

        st.subheader("📈 Daily Signup Count")
        st.line_chart(signup_counts.set_index('date'))

        # 3. Detect Spikes using Isolation Forest
        model = IsolationForest(contamination=0.1, random_state=42)
        signup_counts['is_anomaly'] = model.fit_predict(signup_counts[['signup_count']])
        signup_counts['anomaly'] = signup_counts['is_anomaly'].apply(lambda x: 'Spike' if x == -1 else 'Normal')

        st.subheader("🔍 Anomaly Detection")
        st.write(signup_counts[['date', 'signup_count', 'anomaly']])

        st.markdown("---")
        st.subheader("💬 Ask Questions about the Data")
        question = st.text_input("Ask a question")
        if question:
            response = ollama.chat(
                model='llama3',
                messages=[
                    {'role': 'system', 'content': f"You are a data analyst. Here is the dataset:\n{signup_counts.to_csv(index=False)}"},
                    {'role': 'user', 'content': question}
                ]
            )
            st.write(response['message']['content'])

        st.markdown("---")
        st.subheader("🤖 Bot Signup Detection")

        # Feature Engineering (example: frequency per minute per IP/email etc.)
        if 'ip_address' in df.columns and 'email' in df.columns:
            df['minute'] = df['timestamp'].dt.floor('T')
            freq_features = df.groupby(['ip_address', 'minute']).size().reset_index(name='attempts')

            model_bot = IsolationForest(contamination=0.05)
            freq_features['is_bot'] = model_bot.fit_predict(freq_features[['attempts']])
            freq_features['bot_label'] = freq_features['is_bot'].apply(lambda x: 'Bot' if x == -1 else 'Genuine')

            st.write(freq_features[['ip_address', 'minute', 'attempts', 'bot_label']])

    else:
        st.error("The file must contain a 'timestamp' column to proceed.")
else:
    st.info("👆 Upload a CSV to begin analysis.")
