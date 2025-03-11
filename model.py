import streamlit as st
import pandas as pd
import plotly.express as px

def load_data(file):
    df = pd.read_csv(file)
    return df

def display_top_subjects(df):
    # Filter for top subjects
    top_subjects = ['OBG', 'Anatomy', 'Surgery', 'ENT', 'Medicine']
    df_filtered = df[df['subject_title'].isin(top_subjects)]
    
    # Aggregate hours per subject
    subject_hours = df_filtered.groupby('subject_title')['actual_hours'].sum().reset_index()
    
    # Plot
    fig = px.bar(subject_hours, x='subject_title', y='actual_hours', title='Actual Hours Consumed per Subject',
                 labels={'actual_hours': 'Total Hours'}, color='subject_title')
    st.plotly_chart(fig)

def display_top_users(df):
    # Get top 50 users by actual hours
    top_users = df.groupby('user_id')['actual_hours'].sum().reset_index()
    top_users = top_users.sort_values(by='actual_hours', ascending=False).head(50)
    
    st.write("### Top 50 Users by Actual Hours")
    st.dataframe(top_users)

def main():
    st.set_page_config(page_title='Video Usage Insights', layout='wide')
    st.title('Video Usage Insights - Page 1')
    
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
    if uploaded_file:
        df = load_data(uploaded_file)
        
        display_top_subjects(df)
        display_top_users(df)

if __name__ == "__main__":
    main()
