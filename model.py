import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load("suspicious_model.pkl")

def load_and_preprocess(uploaded_file):
    df = pd.read_csv(uploaded_file)
    
    # Ensure all necessary columns exist
    required_columns = ['user_id', 'actual_hours', '_pause', '_seek', 'lesson_id']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in uploaded file: {missing_cols}")
        return None, None
    
    # Convert user_id to string and clean
    df['user_id'] = df['user_id'].astype(str).str.strip().str.lower()
    
    # Calculate unique lessons per user
    unique_lessons = df.groupby('user_id')['lesson_id'].nunique().rename('unique_lessons')
    
    # Aggregate required features
    features = df.groupby('user_id').agg({
        'actual_hours': 'sum',
        '_pause': 'sum',
        '_seek': 'sum'
    })
    
    # Merge with unique lesson count
    features = features.join(unique_lessons, how='left').fillna(0)
    
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
        
        # _subject-wise total actual hours
        _subject_hours = df.groupby('_subject_title')['actual_hours'].sum().nlargest(5)
        st.subheader("📌 Top 5 _subjects by Watch Hours")
        st.bar_chart(_subject_hours)
        
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

        # Define the features the model was trained on
        trained_features = ['actual_hours', '_pause', '_seek', 'unique_lessons']  # Adjust if needed

        # Ensure all required features exist
        missing_features = [col for col in trained_features if col not in features.columns]
        if missing_features:
            st.error(f"Missing required features: {missing_features}")
        else:
            # Predict suspicious users
            features['is_suspicious'] = model.predict(features[trained_features])

            # Extract flagged users & filter top 50
            suspicious_users = features[features['is_predicted_suspicious'] == 1].nlargest(50, 'actual_hours').reset_index()

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
