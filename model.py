import streamlit as st
import ollama

st.title("Ollama Chat Demo")

user_input = st.text_input("Ask something")

if user_input:
    response = ollama.chat(
        model='llama3',
        messages=[{'role': 'user', 'content': user_input}]
    )
    st.write(response['message']['content'])
