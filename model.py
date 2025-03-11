import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
MODEL_PATH = "suspicious_model.pkl"
model = joblib.load(MODEL_PATH)

# Streamlit page configuration
st.set_page_config(page_title="User Analysis Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Video Usage Trends", "🔍 Suspicious Users Detection", "🔎 Check a User"])

def load_and_preprocess(csv_file):
    """ Load and preprocess data for analysis """
    df = pd.read_csv(csv_file)
    df['user_id'] = df['user_id'].astype(str).str.strip().str.lower()
    
    # Aggregate user interactions
    features = df.groupby('user_id').agg({
        'actual_hours': 'sum',
        '_pause': 'sum',
        '_seek': 'sum',
        'lesson_id': 'nunique',
        '_d_id': 'nunique'  # Ensure 'unique_devices' is derived
    }).rename(columns={'lesson_id': 'unique_lessons', '_d_id': 'unique_devices'})

    # Ensure all expected columns exist
    expected_features = ['actual_hours', '_pause', '_seek', 'unique_lessons', 'unique_devices']
    for col in expected_features:
        if col not in features.columns:
            features[col] = 0  # Fill missing columns with 0

    return df, features


# **PAGE 1: Video Usage Trends**
if page == "📊 Video Usage Trends":
    st.title("📊 Video Usage Analysis")
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df, user_features = load_and_preprocess(uploaded_file)
        
        st.write("Data Preview:")
        st.dataframe(df.head())
        
        # Plot total watch hours distribution
        st.subheader("📉 Distribution of Watch Hours")
        fig, ax = plt.subplots()
        sns.histplot(user_features['actual_hours'], bins=20, kde=True, ax=ax)
        ax.set_xlabel("Total Watch Hours")
        ax.set_ylabel("Number of Users")
        st.pyplot(fig)
        
        # Plot relationship between watch hours and unique lessons
        st.subheader("📊 Watch Hours vs. Unique Lessons")
        fig, ax = plt.subplots()
        sns.scatterplot(data=user_features, x='unique_lessons', y='actual_hours', alpha=0.7, ax=ax)
        ax.set_xlabel("Unique Lessons")
        ax.set_ylabel("Total Watch Hours")
        st.pyplot(fig)

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
