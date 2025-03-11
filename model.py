import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

def load_model():
    return joblib.load("suspicious_model.pkl")

def preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['user_id'] = df['user_id'].astype(str).str.strip().str.lower()
    df['event_date'] = pd.to_datetime(df['event_date'])
    features = df.groupby('user_id').agg({
        'actual_hours': 'sum',
        '_pause': 'sum',
        '_seek': 'sum',
        'lesson_id': 'nunique'
    }).rename(columns={'lesson_id': 'unique_lessons'})
    return features, df

def predict_suspicious_users(features, model):
    features['is_predicted_suspicious'] = model.predict(features)
    suspicious_users = features[features['is_predicted_suspicious'] == 1]
    return suspicious_users.sort_values(by='actual_hours', ascending=False).head(50)

def show_usage_trends(df):
    st.subheader("📊 Video Usage Trends")
    
    # User Journey Flowchart
    st.write("### User Navigation Flow")
    user_journey = df.groupby(['user_id', 'lesson_id']).size().reset_index(name='count')
    fig_flow = px.sankey(user_journey, source='user_id', target='lesson_id', value='count', title='User Video Journey')
    st.plotly_chart(fig_flow)
    
    # Power Users Chart
    st.write("### Power Users")
    power_users = df.groupby('user_id').agg({'actual_hours': 'sum', 'lesson_id': 'nunique'}).reset_index()
    fig_power = px.bar(power_users, x='user_id', y='actual_hours', color='lesson_id', title='Power Users')
    st.plotly_chart(fig_power)
    
    # Time Series Trend
    st.write("### Video Consumption Over Time")
    df['hour'] = df['event_date'].dt.hour
    hourly_trend = df.groupby('hour')['actual_hours'].sum().reset_index()
    fig_time = px.line(hourly_trend, x='hour', y='actual_hours', title='Video Usage Per Hour')
    st.plotly_chart(fig_time)

# Streamlit App UI
st.set_page_config(page_title="Suspicious User Detector", layout="wide")
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #005aa7, #fffde4);
        color: white;
    }
    .stApp {
        background-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔍 Suspicious User Detector & Video Trends")
menu = st.sidebar.radio("Navigation", ["Usage Trends", "Suspicious Users"])
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    features, df = preprocess_data(uploaded_file)
    model = load_model()
    
    if menu == "Usage Trends":
        show_usage_trends(df)
    else:
        st.subheader("🚨 Top 50 Suspicious Users")
        suspicious_users = predict_suspicious_users(features, model)
        st.dataframe(suspicious_users)
        st.download_button("Download CSV", suspicious_users.to_csv(index=False), file_name="suspicious_users.csv", mime="text/csv")
