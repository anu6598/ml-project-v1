import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import numpy as np

# Load datasets (replace with actual file paths or database connections)
@st.cache_data
def load_data():
    playback_data = pd.read_csv("video_playback_data.csv")
    license_data = pd.read_csv("license_consumption_data.csv")
    return playback_data, license_data

playback_data, license_data = load_data()

# Page configuration
st.set_page_config(page_title="User Insights Dashboard", layout="wide")

# Custom light background theme
st.markdown("""
    <style>
        .stApp { background-color: #f9fafb; color: #333333; }
        .block-container { padding: 2rem; }
        .metric { color: #333333; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 User Insights Dashboard: Playback & License Consumption")

# ML-based User Segmentation
st.header("🔍 User Segmentation Based on Completion Speed & Interaction")
seg_data = playback_data.groupby('user_id').agg({
    'event_date': 'min',
    '_lesson_id': 'nunique',
    '_pause': 'sum',
    '_seek': 'sum'
}).reset_index()

seg_data['completion_score'] = seg_data['_lesson_id'] / (
    (pd.to_datetime(playback_data['event_date']).max() - pd.to_datetime(seg_data['event_date'])).dt.days + 1)
seg_data['combination_score'] = seg_data['completion_score'] / (seg_data['_pause'] + seg_data['_seek'] + 1)

seg_data = seg_data.sort_values('combination_score', ascending=False).head(100)

kmeans = KMeans(n_clusters=4, random_state=0)
seg_data['segment'] = kmeans.fit_predict(seg_data[['combination_score', '_pause', '_seek']])

segment_labels = {
    0: "Highly Suspicious",
    1: "Mild Suspicious",
    2: "Moderate User",
    3: "Normal User"
}
seg_data['Category'] = seg_data['segment'].map(segment_labels)

# Page Navigation
page = st.sidebar.selectbox("Select Page", ["User Segmentation", "Detailed User Insights", "Overall License Consumption"])

if page == "User Segmentation":
    st.header("👥 User Categories")
    for category in segment_labels.values():
        st.subheader(f"{category}")
        st.write(seg_data[seg_data['Category'] == category]['user_id'].tolist())

elif page == "Detailed User Insights":
    st.header("🧑‍💻 Detailed User Analysis")
    user_id_input = st.text_input("Enter User ID to view detailed insights:")

    if user_id_input:
        user_playback = playback_data[playback_data['user_id'] == user_id_input]
        user_license = license_data[license_data['user_id'] == user_id_input]

        if user_playback.empty and user_license.empty:
            st.warning("No data found for the given User ID.")
        else:
            st.subheader(f"🎬 Playback Summary for {user_id_input}")
            total_hours = user_playback['actual_hours'].sum()
            st.metric("Total Hours Watched", f"{total_hours:.2f} hours")

            subject_wise = user_playback.groupby('_subject_title')['actual_hours'].sum().reset_index()
            fig_subject = px.bar(subject_wise, x='_subject_title', y='actual_hours', title='Hours Watched per Subject')
            st.plotly_chart(fig_subject)

            st.subheader("📈 License Consumption Trend")
            license_trend = user_license.groupby('date').size().reset_index(name='licenses_consumed')
            fig_license_trend = px.line(license_trend, x='date', y='licenses_consumed', title='License Consumption Over Time')
            st.plotly_chart(fig_license_trend)

            

elif page == "Overall License Consumption":
    st.header("🔒 Overall License Consumption Patterns")
    total_licenses = license_data.shape[0]
    st.metric("Total Licenses Consumed", total_licenses)

    platform_usage = license_data['platform'].value_counts().reset_index()
    platform_usage.columns = ['Platform', 'Count']
    fig_platform = px.pie(platform_usage, names='Platform', values='Count', title='Platform Usage Distribution')
    st.plotly_chart(fig_platform)

    license_trend_overall = license_data.groupby('event_date').size().reset_index(name='licenses_consumed')
    fig_overall_trend = px.line(license_trend_overall, x='event_date', y='licenses_consumed', title='Overall License Consumption Over Time')
    st.plotly_chart(fig_overall_trend)
