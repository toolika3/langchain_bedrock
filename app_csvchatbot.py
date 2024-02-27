import json
import os
import pandas as pd

import boto3

from langchain.llms.bedrock import Bedrock

import streamlit as st
import pickle
import tempfile
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader

boto_session = boto3.Session()
aws_region = boto_session.region_name

br_client = boto_session.client(service_name='bedrock', region_name=aws_region)
br_runtime = boto_session.client(service_name='bedrock-runtime', region_name=aws_region)

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [AWS Bedrock](https://aws.amazon.com/bedrock/) LLM model

    ''')
    add_vertical_space(5)
    st.write('Data Products Team')


def main():
    st.header("Chat with CSV Files 💬")

    # upload a PDF file
    file = st.file_uploader("Upload your file", type="csv")

    if file is not None:
        store_name = file.name[-3:]
        # CSVLoader looks for a filepath, so use tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name
        loader = CSVLoader(file_path=tmp_file_path,
                           encoding="utf-8",
                           csv_args={"delimiter": ","})
        data = loader.load()

   # embeddings
        store_name = file.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)
        if os.path.exists(f"{store_name}.pkl"):
            embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1', client=br_runtime)
            VectorStore = FAISS.from_documents(data, embedding=embeddings)
            #with open(f"{store_name}.pkl", "rb") as f:
             #   VectorStore = pickle.load(f)
              #  st.write('Embeddings Loaded from the Disk')
        else:
            embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1', client=br_runtime)
            VectorStore = FAISS.from_documents(data, embedding=embeddings)
            
            #with open(f"{store_name}.pkl", "wb") as f:
             #   pickle.dump(VectorStore, f)
              #  st.write('Embeddings Completed')

        llm = Bedrock(model_id='anthropic.claude-v2:1', client=br_runtime)
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=VectorStore.as_retriever())
        # Accept user questions/query
        query = st.text_area("Ask questions about your file:")
        #st.write(query)
        if query:
            result = chain({"question": query, "chat_history": st.session_state['history']})
            st.session_state['history'].append((query, result["answer"]))
            return result["answer"]
     
if __name__ == '__main__':
    main()