import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import numpy as np

# Load datasets
@st.cache_data
def load_data():
    playback_data = pd.read_csv("video_playback_data.csv")
    license_data = pd.read_csv("license_consumption_data.csv")
    return playback_data, license_data

playback_data, license_data = load_data()

# Page Configuration
st.set_page_config(page_title="User Insights Platform", layout="wide")

# Custom CSS for website-like UI
st.markdown("""
    <style>
        .main { background-color: #f9fbfd; }
        .stApp { background-image: linear-gradient(to right, #dde7f0, #e9f1fa); color: black; }
        .css-1d391kg .st-bh {background: transparent;}
        .css-1v0mbdj {color: black;}
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["User Segmentation Overview", "User Insights", "License Consumption Overview"])

# ML-based User Segmentation
seg_data = playback_data.groupby('user_id').agg({
    'event_date': 'min',
    '_lesson_id': 'nunique',
    '_pause': 'sum',
    '_seek': 'sum'
}).reset_index()

seg_data['completion_days'] = (pd.to_datetime(playback_data['event_date']).max() - pd.to_datetime(seg_data['event_date'])).dt.days
seg_data['completion_score'] = seg_data['_lesson_id'] / seg_data['completion_days'].replace(0, np.nan)

seg_data['Category'] = np.where(
    (seg_data['completion_score'] > 0.5) & (seg_data['_pause'] < 5) & (seg_data['_seek'] < 5), "Highly Suspicious",
    np.where((seg_data['completion_score'] > 0.3) & (seg_data['_pause'] < 10) & (seg_data['_seek'] < 10), "Suspicious", "Normal User")
)

# Page 1: Segmentation Overview
if page == "User Segmentation Overview":
    st.title("User Segmentation Overview")
    st.markdown("### Explore user categories based on their interaction and completion patterns.")

    for category in ["Highly Suspicious", "Suspicious", "Normal User"]:
        st.subheader(f"{category} Users")
        user_list = seg_data[seg_data['Category'] == category]['user_id'].tolist()
        st.write(f"**Count:** {len(user_list)}")
        st.json(user_list)

# Page 2: User-specific Insights
elif page == "User Insights":
    st.title("Detailed User Insights")
    user_id_input = st.text_input("Enter User ID:")

    if user_id_input:
        user_playback = playback_data[playback_data['user_id'] == user_id_input]
        user_license = license_data[license_data['user_id'] == user_id_input]

        if user_playback.empty and user_license.empty:
            st.warning("No data found for the given User ID.")
        else:
            st.subheader(f"Playback Summary for {user_id_input}")
            total_hours = user_playback['actual_hours'].sum()
            st.metric("Total Hours Watched", f"{total_hours:.2f} hours")

            subject_wise = user_playback.groupby('_subject_title')['actual_hours'].sum().reset_index()
            fig_subject = px.bar(subject_wise, x='_subject_title', y='actual_hours', title='Hours Watched per Subject')
            st.plotly_chart(fig_subject)

            st.subheader("License Consumption Pattern")
            license_count = user_license.shape[0]
            st.metric("Total Licenses Consumed", f"{license_count}")

            platform_usage = user_license['platform'].value_counts().reset_index()
            platform_usage.columns = ['Platform', 'Count']
            fig_platform = px.pie(platform_usage, names='Platform', values='Count', title='Platform Usage')
            st.plotly_chart(fig_platform)

# Page 3: Overall License Consumption
elif page == "License Consumption Overview":
    st.title("Overall License Consumption Overview")
    st.markdown("### Analyze general trends in license consumption across all users.")

    total_licenses = license_data.shape[0]
    st.metric("Total Licenses Issued", total_licenses)

    platform_dist = license_data['platform'].value_counts().reset_index()
    platform_dist.columns = ['Platform', 'Count']
    fig_all_platforms = px.pie(platform_dist, names='Platform', values='Count', title='Platform Usage Distribution')
    st.plotly_chart(fig_all_platforms)

    region_dist = license_data['_region'].value_counts().reset_index()
    region_dist.columns = ['Region', 'Count']
    fig_region = px.bar(region_dist, x='Region', y='Count', title='Region-wise License Distribution')
    st.plotly_chart(fig_region)
