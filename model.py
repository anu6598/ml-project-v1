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

# Page selection
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["User Segmentation Overview", "Detailed User Insights", "Overall License Consumption"])

# Combination score calculation
seg_data = playback_data.groupby('user_id').agg({
    'event_date': 'min',
    '_lesson_id': 'nunique',
    '_pause': 'sum',
    '_seek': 'sum',
    'actual_hours': 'sum'
}).reset_index()

seg_data['completion_speed'] = seg_data['_lesson_id'] / (
    (pd.to_datetime(playback_data['event_date']).max() - pd.to_datetime(seg_data['event_date'])).dt.days + 1
)

license_counts = license_data.groupby('user_id').size().reset_index(name='license_count')
seg_data = seg_data.merge(license_counts, on='user_id', how='left').fillna(0)

seg_data['combination_score'] = (
    0.4 * seg_data['actual_hours'] +
    0.3 * seg_data['_lesson_id'] +
    0.2 * seg_data['completion_speed'] +
    0.1 * seg_data['license_count']
)

seg_data.sort_values(by='combination_score', ascending=False, inplace=True)
seg_data = seg_data.head(100)

kmeans = KMeans(n_clusters=3, random_state=0)
seg_data['segment'] = kmeans.fit_predict(seg_data[['combination_score', '_pause', '_seek']])

segment_labels = {
    0: "Highly Suspicious",
    1: "Suspicious",
    2: "Normal User"
}
seg_data['Category'] = seg_data['segment'].map(segment_labels)

# Page 1: User Segmentation Overview
if page == "User Segmentation Overview":
    st.title("User Segmentation Overview (Top 100 Users)")
    for category in segment_labels.values():
        st.subheader(f"{category} Users")
        st.write(seg_data[seg_data['Category'] == category]['user_id'].tolist())

# Page 2: Detailed User Insights
elif page == "Detailed User Insights":
    st.title("Detailed User Insights")
    user_id_input = st.text_input("Enter User ID to view details:")

    if user_id_input:
        user_playback = playback_data[playback_data['user_id'] == user_id_input]
        user_license = license_data[license_data['user_id'] == user_id_input]

        if user_playback.empty and user_license.empty:
            st.warning("No data found for the given User ID.")
        else:
            st.header(f"Insights for User ID: {user_id_input}")
            st.metric("Total Hours Watched", f"{user_playback['actual_hours'].sum():.2f} hours")

            subject_wise = user_playback.groupby('_subject_title')['actual_hours'].sum().reset_index()
            fig_subject = px.bar(subject_wise, x='_subject_title', y='actual_hours', title='Hours Watched per Subject')
            st.plotly_chart(fig_subject)

            st.subheader("License Consumption Patterns")
            st.metric("Total Licenses Consumed", f"{user_license.shape[0]}")

            platform_usage = user_license['platform'].value_counts().reset_index()
            platform_usage.columns = ['Platform', 'Count']
            fig_platform = px.pie(platform_usage, names='Platform', values='Count', title='Platform Usage Distribution')
            st.plotly_chart(fig_platform)

            st.subheader("Subject Completion Status")
            completion = user_playback.groupby('_subject_title')['percentage'].mean().reset_index()
            completion.columns = ['Subject', 'Avg Completion %']
            fig_completion = px.bar(completion, x='Subject', y='Avg Completion %', title='Subject Completion Status')
            st.plotly_chart(fig_completion)

# Page 3: Overall License Consumption
else:
    st.title("Overall License Consumption Patterns")
    overall_license = license_data['platform'].value_counts().reset_index()
    overall_license.columns = ['Platform', 'Count']
    fig_overall = px.pie(overall_license, names='Platform', values='Count', title='Overall Platform Usage Distribution')
    st.plotly_chart(fig_overall)
