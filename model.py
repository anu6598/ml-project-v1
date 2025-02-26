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

# Light Theme Styling
st.markdown("""
    <style>
        .main { background-color: #f9fbfd; }
        .stApp { background-image: linear-gradient(to right, #f5f7fa, #c3cfe2); color: #333333; }
        .css-1d391kg .st-bh {background: transparent;}
    </style>
""", unsafe_allow_html=True)

st.title("User Insights Dashboard: Playback & License Consumption")

# Page Selection
page = st.sidebar.selectbox("Select Page:", [
    "Overall User Segmentation",
    "Detailed User Analysis",
    "Overall License Consumption"
])

if page == "Overall User Segmentation":
    st.header("User Segmentation Overview")
    
    seg_data = playback_data.groupby('user_id').agg({
        'event_date': 'min',
        '_lesson_id': 'nunique',
        '_pause': 'sum',
        '_seek': 'sum'
    }).reset_index()

    seg_data['completion_score'] = seg_data['_lesson_id'] / (
        (pd.to_datetime(playback_data['event_date']).max() - pd.to_datetime(seg_data['event_date'])).dt.days + 1
    )

    # Combination Score for better segmentation
    seg_data['combination_score'] = seg_data['completion_score'] / (seg_data['_pause'] + seg_data['_seek'] + 1)

    kmeans = KMeans(n_clusters=3, random_state=0)
    seg_data['segment'] = kmeans.fit_predict(seg_data[['combination_score', '_pause', '_seek']])

    segment_labels = {0: "Highly Suspicious", 1: "Suspicious", 2: "Normal User"}
    seg_data['Category'] = seg_data['segment'].map(segment_labels)

    st.subheader("Top 100 Users by Combination Score")
    for category in segment_labels.values():
        top_users = seg_data[seg_data['Category'] == category].sort_values(
            "combination_score", ascending=False).head(100)
        st.write(f"**{category} Users:**", top_users['user_id'].tolist())

elif page == "Detailed User Analysis":
    st.header("Detailed User Insights")
    user_id_input = st.text_input("Enter User ID to view details:")

    if user_id_input:
        user_playback = playback_data[playback_data['user_id'] == user_id_input]
        user_license = license_data[license_data['user_id'] == user_id_input]

        if user_playback.empty and user_license.empty:
            st.warning("No data found for the given User ID.")
        else:
            st.subheader(f"Playback Summary for User ID: {user_id_input}")
            total_hours = user_playback['actual_hours'].sum()
            st.metric("Total Hours Watched", f"{total_hours:.2f} hours")

            subject_wise = user_playback.groupby('_subject_title')['actual_hours'].sum().reset_index()
            fig_subject = px.bar(subject_wise, x='_subject_title', y='actual_hours',
                                 title='Hours Watched per Subject')
            st.plotly_chart(fig_subject)

            st.subheader("License Consumption Over Time")
            if not user_license.empty:
                user_license['date'] = pd.to_datetime(user_license['event_date'])
                license_trend = user_license.groupby('date').size().reset_index(name='licenses')
                fig_license = px.line(license_trend, x='date', y='licenses',
                                      title='License Consumption Trend')
                st.plotly_chart(fig_license)

            st.subheader("License Consumption Patterns")
            license_count = user_license.shape[0]
            st.metric("Total Licenses Consumed", f"{license_count}")

            platform_usage = user_license['platform'].value_counts().reset_index()
            platform_usage.columns = ['Platform', 'Count']
            fig_platform = px.pie(platform_usage, names='Platform', values='Count',
                                  title='Platform Usage Distribution')
            st.plotly_chart(fig_platform)

            st.subheader("User Journey Insights")
            subject_order = (
                user_playback.groupby(['_subject_title', '_lesson_id'])
                .agg({'event_date': 'min'}).reset_index().sort_values('event_date')
            )
            st.dataframe(subject_order, use_container_width=True)

            st.subheader("Completion Status per Subject")
            completion = user_playback.groupby('_subject_title')['percentage'].mean().reset_index()
            completion.columns = ['Subject', 'Avg Completion %']
            fig_completion = px.bar(completion, x='Subject', y='Avg Completion %',
                                    title='Subject Completion Status')
            st.plotly_chart(fig_completion)

            st.subheader("Finisher Category")
            total_lessons = user_playback['_lesson_id'].nunique()
            completed_lessons = user_playback[user_playback['percentage'] >= 85]['_lesson_id'].nunique()
            finisher_type = "Fast Finisher" if completed_lessons / total_lessons >= 0.8 else "Slow Finisher"
            st.success(f"This user is categorized as a: {finisher_type}")

            st.subheader("Playback Trends Over Time")
            playback_trend = (
                user_playback.groupby('event_date')['actual_hours'].sum().reset_index()
            )
            fig_playback_trend = px.line(playback_trend, x='event_date', y='actual_hours',
                                         title='Playback Hours Over Time')
            st.plotly_chart(fig_playback_trend)

            st.subheader("Ask a Custom Question")
            query = st.text_input("Type your question here:")

            def answer_query(q):
                q = q.lower()
                if "top most viewed subject" in q:
                    top_subject = subject_wise.sort_values("actual_hours", ascending=False).iloc[0]['_subject_title']
                    return f"Top most viewed subject: {top_subject}"
                elif "total hours" in q:
                    return f"Total hours watched: {total_hours:.2f} hours"
                else:
                    return "Query not recognized. Please try a different question."

            if query:
                st.info(answer_query(query))

elif page == "Overall License Consumption":
    st.header("Overall License Consumption Patterns")
    license_data['event_date'] = pd.to_datetime(license_data['event_date'])
    overall_license_trend = license_data.groupby('event_date').size().reset_index(name='licenses')
    fig_overall_license = px.line(overall_license_trend, x='event_date', y='licenses',
                                  title='Overall License Consumption Over Time')
    st.plotly_chart(fig_overall_license)

    st.subheader("Platform-wise Distribution")
    overall_platform = license_data['platform'].value_counts().reset_index()
    overall_platform.columns = ['Platform', 'Count']
    fig_overall_platform = px.pie(overall_platform, names='Platform', values='Count',
                                  title='Overall Platform Usage Distribution')
    st.plotly_chart(fig_overall_platform)
