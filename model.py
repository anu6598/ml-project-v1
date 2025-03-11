import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np

# Load trained model
model_path = 'suspicious_model.pkl'
model = joblib.load(model_path)

# Streamlit UI
st.set_page_config(page_title='User Analysis', layout='wide')
st.title("📊 User Analysis Dashboard")

# Sidebar navigation
page = st.sidebar.radio("Select Page", ["Upload Data", "Suspicious Users", "User Analysis", "Forecasting"])

if page == "Upload Data":
    st.header("📂 Upload CSV File")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Data:")
        st.dataframe(df.head())
        st.session_state['uploaded_data'] = df

elif page == "Suspicious Users":
    st.header("🚨 Suspicious Users Detection")
    if 'uploaded_data' in st.session_state:
        df = st.session_state['uploaded_data']
        features = df.groupby('user_id').agg({
            'actual_hours': 'sum',
            '_pause': 'sum',
            '_seek': 'sum',
            'lesson_id': 'nunique'
        }).rename(columns={'lesson_id': 'unique_lessons'})
        
        features['is_predicted_suspicious'] = model.predict(features)
        suspicious_users = features[features['is_predicted_suspicious'] == 1].reset_index()
        suspicious_users = suspicious_users.sort_values(by='actual_hours', ascending=False).head(50)
        st.write("Top 50 Suspicious Users:")
        st.dataframe(suspicious_users[['user_id', 'actual_hours']])
    else:
        st.warning("Please upload a CSV file first.")

elif page == "User Analysis":
    st.header("🔍 User Analysis")
    user_id = st.text_input("Enter User ID:")
    if user_id and 'uploaded_data' in st.session_state:
        df = st.session_state['uploaded_data']
        user_data = df[df['user_id'] == user_id]
        if not user_data.empty:
            user_features = user_data.groupby('user_id').agg({
                'actual_hours': 'sum',
                '_pause': 'sum',
                '_seek': 'sum',
                'lesson_id': 'nunique'
            }).rename(columns={'lesson_id': 'unique_lessons'})
            
            prediction = model.predict(user_features)
            result = "Suspicious" if prediction[0] == 1 else "Not Suspicious"
            reason = "High actual hours / No pause & seek activity" if prediction[0] == 1 else "Normal User"
            
            st.write(f"### User ID: {user_id}")
            st.write(f"#### Result: {result}")
            st.write(f"#### Reason: {reason}")
            st.write("User Data:")
            st.dataframe(user_data)
        else:
            st.warning("User ID not found.")

elif page == "Forecasting":
    st.header("📈 Forecasting Actual Hours")
    if 'uploaded_data' in st.session_state:
        df = st.session_state['uploaded_data']
        if 'timestamp' in df.columns and 'actual_hours' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.groupby('timestamp')['actual_hours'].sum().reset_index()
            
            fig = px.line(df, x='timestamp', y='actual_hours', title='Actual Hours Over Time')
            st.plotly_chart(fig)
            
            # ARIMA Forecasting
            df.set_index('timestamp', inplace=True)
            model = ARIMA(df['actual_hours'], order=(5,1,0))
            arima_model = model.fit()
            forecast = arima_model.forecast(steps=1)
            
            st.subheader("📊 Forecast for Next Day")
            st.write(f"Predicted Actual Hours: {forecast.values[0]:.2f}")
        else:
            st.warning("Timestamp or Actual Hours column missing in uploaded data.")
    else:
        st.warning("Please upload a CSV file first.")
