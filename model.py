import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(MLSheet1.csv)
        return df
    return None

# User Segmentation based on completion and event dates
def segment_users(df):
    user_completion = df.groupby(['user_id', 'lesson_id']).agg({
        'percentage': 'mean',
        'event_date': pd.Series.nunique
    }).reset_index()

    user_summary = user_completion.groupby('user_id').agg({
        'percentage': 'mean',
        'event_date': 'sum'
    }).reset_index()

    user_summary['segment'] = user_summary.apply(
        lambda x: 'Fast Finisher' if x['percentage'] >= 85 and x['event_date'] <= user_summary['event_date'].quantile(0.25)
        else ('Slow Finisher' if x['percentage'] >= 85 else 'Regular'), axis=1)

    return user_summary

# Visualize user journey for a selected user
def user_journey(df, user_id):
    user_data = df[df['user_id'] == user_id].sort_values('event_date')
    fig = px.line(user_data, x='event_date', y='percentage', color='topic_title',
                  title=f'Learning Journey for User {user_id}')
    st.plotly_chart(fig)

# Main Streamlit app
def main():
    st.title("User Segmentation & Study Journey Analysis - NEET PG & FMGE")

    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    df = load_data(uploaded_file)

    if df is not None:
        st.subheader("User Segmentation")
        user_summary = segment_users(df)
        st.dataframe(user_summary)

        segment_counts = user_summary['segment'].value_counts().reset_index()
        fig_seg = px.pie(segment_counts, values='segment', names='index',
                         title='User Segmentation Overview')
        st.plotly_chart(fig_seg)

        st.subheader("User Journey Visualization")
        user_list = df['user_id'].unique()
        selected_user = st.selectbox("Select a User ID", user_list)
        user_journey(df, selected_user)

        st.subheader("Fast vs Slow Finishers Distribution")
        fig_dist = px.histogram(user_summary, x='event_date', color='segment',
                                title='Completion Speed Distribution')
        st.plotly_chart(fig_dist)

        st.success("Analysis complete. Adjust filters for more insights.")
    else:
        st.info("Please upload a dataset to proceed.")

if __name__ == "__main__":
    main()
