# import stremlit and chatbot file

import streamlit as st
import bedrock_backend as demo

st.title("Hi, This is your Virtual Assistant Mae :sunglasses:")

if 'memory' not in st.session_state:
    st.session_state.memory = demo.memory()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])

input_text = st.chat_input("Powered by Bedrock and Claude")
if input_text:

    with st.chat_message("user"):
        st.markdown(input_text)

    st.session_state.chat_history.append({"role": "user", "text": input_text})

    chat_response = demo.conversation(input_text=input_text, memory=st.session_state.memory)

    with st.chat_message("assistant"):
        st.markdown(chat_response)

    st.session_state.chat_history.append({"role": "assistant", "text": chat_response})