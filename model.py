import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['created_on'] = pd.to_datetime(df['created_on'], errors='coerce')
    df['last_updated'] = pd.to_datetime(df['last_updated'], errors='coerce')
    return df

def preprocess_data(df):
    user_activity = df.groupby('user_id').agg(
        rate_limit_violations=('limit_name', 'count'),
        unique_devices=('device_id', 'nunique'),
        unique_ips=('ip', 'nunique'),
        first_access=('created_on', 'min'),
        last_access=('last_updated', 'max')
    ).reset_index()
    user_activity['active_days'] = (user_activity['last_access'] - user_activity['first_access']).dt.days + 1
    return user_activity

def detect_anomalies(df):
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    df['anomaly_score'] = model.fit_predict(df[['rate_limit_violations', 'unique_devices', 'unique_ips', 'active_days']])
    suspicious_users = df[df['anomaly_score'] == -1]
    return suspicious_users

st.title("Piracy User Detection System")

uploaded_file = st.file_uploader("Upload Video Rate Limit Log (CSV)", type=['csv'])

if uploaded_file:
    df = load_data(uploaded_file)
    st.write("### Raw Data Sample")
    st.write(df.head())
    
    user_data = preprocess_data(df)
    suspicious_users = detect_anomalies(user_data)
    
    st.write("### Suspicious Users Detected")
    st.write(suspicious_users[['user_id', 'rate_limit_violations', 'unique_devices', 'unique_ips', 'active_days']])
    
    st.download_button("Download Suspicious Users", suspicious_users.to_csv(index=False), "suspicious_users.csv", "text/csv")
