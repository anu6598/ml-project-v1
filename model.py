import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load trained model
MODEL_PATH = "suspicious_model.pkl"
model = joblib.load(MODEL_PATH)

def load_and_preprocess(csv_file):
    df = pd.read_csv(csv_file)
    df['user_id'] = df['user_id'].astype(str).str.strip().str.lower()
    df['event_date'] = pd.to_datetime(df['event_date'])

    features = df.groupby('user_id').agg({
        'actual_hours': 'sum',
        '_pause': 'sum',
        '_seek': 'sum',
        'lesson_id': 'nunique',
        '_d_id': 'nunique',
        '_region': 'first'  # Assuming region is a user-level attribute
    }).rename(columns={'lesson_id': 'unique_lessons', '_d_id': 'unique_devices'})
    
    return df, features

def get_suspicious_users(features):
    features['is_predicted_suspicious'] = model.predict(features)
    suspicious_users = features[features['is_predicted_suspicious'] == 1]
    suspicious_users = suspicious_users.nlargest(50, 'actual_hours')  # Top 50 users
    return suspicious_users.reset_index()[['user_id', 'actual_hours', 'unique_lessons']]

def show_trends(df):
    st.subheader("User Video Usage Trends")
    
    # User journey visualization
    user_journey = df.groupby(['user_id', 'event_date']).size().reset_index(name='video_count')
    fig1 = px.line(user_journey, x='event_date', y='video_count', color='user_id', title="User Video Consumption Over Time")
    st.plotly_chart(fig1)
    
    # Power Users
    power_users = df.groupby('user_id').agg({'actual_hours': 'sum', 'lesson_id': 'nunique'}).reset_index()
    power_users = power_users.nlargest(10, 'actual_hours')
    st.subheader("Top Power Users")
    st.dataframe(power_users)
    
    # Regional Distribution
    region_dist = df['_region'].value_counts().reset_index()
    region_dist.columns = ['Region', 'User Count']
    fig2 = px.bar(region_dist, x='Region', y='User Count', title="User Distribution by Region")
    st.plotly_chart(fig2)

def main():
    st.set_page_config(page_title="Video Analytics", layout="wide")
    st.markdown("""
        <style>
            body {
                background: linear-gradient(to right, #1e3c72, #2a5298);
                color: white;
            }
            .stApp {
                background-color: transparent;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📊 Video Analytics Dashboard")
    
    menu = ["Usage Trends", "Suspicious Users"]
    choice = st.sidebar.selectbox("Select Page", menu)
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df, features = load_and_preprocess(uploaded_file)
        
        if choice == "Usage Trends":
            show_trends(df)
        elif choice == "Suspicious Users":
            st.subheader("Top 50 Suspicious Users")
            suspicious_users = get_suspicious_users(features)
            st.dataframe(suspicious_users)

if __name__ == "__main__":
    main()
