import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load("suspicious_users_model.pkl")

def load_and_preprocess(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.fillna(0, inplace=True)
    df['user_id'] = df['user_id'].astype(str).str.strip().str.lower()
    
    # Select necessary features
    features = df[['actual_hours', '_pause', '_seek', 'unique_lessons']]
    features = StandardScaler().fit_transform(features)
    features = pd.DataFrame(features, columns=['actual_hours', '_pause', '_seek', 'unique_lessons'], index=df['user_id'])
    
    return df, features

# Streamlit App Layout
st.set_page_config(page_title="User Analysis Dashboard", page_icon="📊", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Video Usage Trends", "🔍 Suspicious Users Detection", "🔎 Check a User"])

# **PAGE 1: Video Usage Trends**
if page == "📊 Video Usage Trends":
    st.title("📊 Video Usage Trends")
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df, _ = load_and_preprocess(uploaded_file)
        
        # Subject-wise total actual hours
        subject_hours = df.groupby('subject_title')['actual_hours'].sum().nlargest(5)
        st.subheader("📌 Top 5 Subjects by Watch Hours")
        st.bar_chart(subject_hours)
        
        # Top 50 users by actual hours
        top_users = df.groupby('user_id')['actual_hours'].sum().nlargest(50)
        st.subheader("🏆 Top 50 Users by Watch Hours")
        st.dataframe(top_users)

# **PAGE 2: Suspicious Users Detection**
elif page == "🔍 Suspicious Users Detection":
    st.title("🔍 Detect Suspicious Users")
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df, features = load_and_preprocess(uploaded_file)
        
        # Predict suspicious users
        features['is_predicted_suspicious'] = model.predict(features)

        # Extract flagged users & filter top 50
        suspicious_users = features[features['is_predicted_suspicious'] == 1].nlargest(50, 'actual_hours')
        suspicious_users = suspicious_users.reset_index()
        
        # Display Results
        st.subheader("🚨 Top 50 Suspicious Users")
        st.dataframe(suspicious_users[['user_id', 'actual_hours', 'unique_lessons']])
        
        # Save results
        suspicious_users.to_csv("predicted_suspicious_users.csv", index=False)
        st.success("✅ Predictions saved to predicted_suspicious_users.csv")

# **PAGE 3: Check a User**
elif page == "🔎 Check a User":
    st.title("🔎 Check if a User is Suspicious")
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df, features = load_and_preprocess(uploaded_file)
        
        # User ID input
        user_id = st.text_input("Enter a User ID to Check", "").strip().lower()
        
        if user_id:
            if user_id in features.index:
                user_data = features.loc[user_id]
                prediction = model.predict([user_data])[0]
                
                # Display user data
                st.subheader("📊 User Data")
                st.dataframe(pd.DataFrame(user_data).T)

                # Display Result
                if prediction == 1:
                    st.error(f"🚨 User {user_id} is **SUSPICIOUS** 🚨")
                else:
                    st.success(f"✅ User {user_id} is **NOT Suspicious** ✅")

                # Reasoning
                st.subheader("🧐 Why is this user suspicious?")
                if prediction == 1:
                    reasons = []
                    if user_data['actual_hours'] > 10:
                        reasons.append("Unusually high watch hours.")
                    if user_data['_pause'] == 0 and user_data['_seek'] == 0:
                        reasons.append("No pauses or seeks detected.")
                    if user_data['unique_lessons'] < 2:
                        reasons.append("Very few unique lessons watched.")
                    
                    for reason in reasons:
                        st.write(f"🔸 {reason}")
            else:
                st.warning("⚠️ User ID not found in the uploaded data.")
