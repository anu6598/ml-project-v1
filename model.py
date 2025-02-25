import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Load datasets (replace with actual file paths or database connections)
@st.cache_data
def load_data():
    playback_data = pd.read_csv("video_playback_data.csv")
    license_data = pd.read_csv("license_consumption_data.csv")
    return playback_data, license_data

playback_data, license_data = load_data()

# Updated background color for better text visibility
st.markdown("""
    <style>
        .main { background-color: #f8f9fc; }
        .stApp {
            background-image: linear-gradient(to right, #d3cce3, #e9e4f0);
            color: #222222;
        }
        .css-1d391kg .st-bh {background: transparent;}
    </style>
""", unsafe_allow_html=True)

st.title("User Insights Dashboard: Playback & License Consumption")

# Updated User Segregation Based on Explicit Conditions
st.subheader("User Segmentation Based on Completion Speed & Interaction")
seg_data = playback_data.groupby('user_id').agg({
    'event_date': 'min',
    '_lesson_id': 'nunique',
    '_pause': 'sum',
    '_seek': 'sum'
}).reset_index()

seg_data['completion_score'] = seg_data['_lesson_id'] / (
    (pd.to_datetime(playback_data['event_date']).max() - pd.to_datetime(seg_data['event_date'])).dt.days + 1)

# Updated category logic based on defined thresholds
def categorize_user(row):
    if row['completion_score'] >= 0.8 and row['_pause'] < 5 and row['_seek'] < 5:
        return "Highly Suspicious"
    elif row['completion_score'] >= 0.6 and row['_pause'] < 10 and row['_seek'] < 10:
        return "Mild Suspicious"
    elif row['completion_score'] >= 0.4:
        return "Moderate User"
    else:
        return "Normal User"

seg_data['Category'] = seg_data.apply(categorize_user, axis=1)

for category in seg_data['Category'].unique():
    st.write(f"**{category}:**", seg_data[seg_data['Category'] == category]['user_id'].tolist())

# User specific insights
user_id_input = st.text_input("Enter User ID to view details:")

if user_id_input:
    user_playback = playback_data[playback_data['user_id'] == user_id_input]
    user_license = license_data[license_data['user_id'] == user_id_input]

    if user_playback.empty and user_license.empty:
        st.warning("No data found for the given User ID.")
    else:
        st.header(f"Insights for User ID: {user_id_input}")

        st.subheader("Playback Summary")
        total_hours = user_playback['actual_hours'].sum()
        st.metric("Total Hours Watched", f"{total_hours:.2f} hours")

        subject_wise = user_playback.groupby('_subject_title')['actual_hours'].sum().reset_index()
        fig_subject = px.bar(subject_wise, x='_subject_title', y='actual_hours', title='Hours Watched per Subject')
        fig_subject.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_subject)

        st.subheader("License Consumption Patterns")
        license_count = user_license.shape[0]
        st.metric("Total Licenses Consumed", f"{license_count}")

        platform_usage = user_license['platform'].value_counts().reset_index()
        platform_usage.columns = ['Platform', 'Count']
        fig_platform = px.pie(platform_usage, names='Platform', values='Count', title='Platform Usage Distribution')
        fig_platform.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_platform)

        st.subheader("User Journey Insights")
        subject_order = (
            user_playback.groupby(['_subject_title', '_lesson_id'])
            .agg({'event_date': 'min'})
            .reset_index()
            .sort_values('event_date')
        )
        st.dataframe(subject_order, use_container_width=True)

        st.subheader("Completion Status")
        completion = user_playback.groupby('_subject_title')['percentage'].mean().reset_index()
        completion.columns = ['Subject', 'Avg Completion %']
        fig_completion = px.bar(completion, x='Subject', y='Avg Completion %', title='Subject Completion Status')
        fig_completion.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_completion)

        st.subheader("Finisher Category")
        total_lessons = user_playback['_lesson_id'].nunique()
        completed_lessons = user_playback[user_playback['percentage'] >= 85]['_lesson_id'].nunique()
        finisher_type = "Fast Finisher" if completed_lessons / total_lessons >= 0.8 else "Slow Finisher"
        st.success(f"This user is categorized as a: {finisher_type}")

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
else:
    st.info("Please enter a User ID to load insights.")
