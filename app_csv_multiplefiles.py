import json
import os
import pandas as pd

import boto3

from langchain.llms.bedrock import Bedrock

import streamlit as st
from streamlit_chat import message
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import DirectoryLoader


boto_session = boto3.Session()
aws_region = boto_session.region_name
DB_FAISS_PATH = 'vectorstore/db_faiss'
csvfile_path = '../bedrock_test/csvstore/'


br_client = boto_session.client(service_name='bedrock', region_name=aws_region)
br_runtime = boto_session.client(service_name='bedrock-runtime', region_name=aws_region)

# Sidebar contents
st.sidebar.title('💬 LLM Chat App')
st.sidebar.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [AWS Bedrock](https://aws.amazon.com/bedrock/) LLM model

    ''')
add_vertical_space(3)
st.sidebar.write('Data Products Team')

# upload CSV files

loader = DirectoryLoader(csvfile_path, glob='**/*.csv', loader_cls=CSVLoader)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )

chunks = text_splitter.transform_documents(data)
embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1',
                               client=br_runtime)
vectordb = FAISS.from_documents(chunks, embedding=embeddings)
vectordb.save_local(DB_FAISS_PATH)


llm = Bedrock(model_id='anthropic.claude-v2:1', client=br_runtime)
chain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=vectordb.as_retriever(search_kwargs={"k":10}))


def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]


if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello! Ask Questions"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey!"]

#Container for chat history
response_container = st.container()
# container for user text
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):

        user_input = st.text_input("Query:", placeholder="Ask questions here", key='input')
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input:
        output = conversational_chat(user_input)
        # store output
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))