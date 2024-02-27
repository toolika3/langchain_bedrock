import json
import os
import pandas as pd

import boto3

from langchain.llms.bedrock import Bedrock

import streamlit as st
import pickle
import tempfile
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
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
    st.header("Chat with Files 💬")

    # upload a PDF file
    file = st.file_uploader("Upload your file")
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )

    if file is not None:
        store_name = file.name[-3:]
        if store_name == 'pdf':
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            #st.write(text)
            chunks = text_splitter.split_text(text=text)
            #st.write(chunks)
        elif store_name == 'csv':
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name
                #st.write(tmp_file_path)
            loader = CSVLoader(file_path=tmp_file_path,
                               encoding="utf-8",
                               csv_args={"delimiter": ","})
            text = loader.load()
            st.write(text)
            chunks = text_splitter.transform_documents(text)
            #st.write(chunks)
        else:
            st.write("No file recieved")

       # embeddings
        store_name = file.name[:-4]
        #st.write(f'{store_name}')
        # st.write(chunks)

        if os.path.exists(f"{store_name}.pkl"):
            embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1', client=br_runtime)
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            #with open(f"{store_name}.pkl", "rb") as f:
             #   VectorStore = pickle.load(f)
              #  st.write('Embeddings Loaded from the Disk')
        else:
            embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1', client=br_runtime)
            store_name = file.name[-3:]
            if store_name == 'pdf':
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            elif store_name == 'csv':
                VectorStore = FAISS.from_documents(chunks, embedding=embeddings)
            else:
                st.write("No Data to process")
            #with open(f"{store_name}.pkl", "wb") as f:
             #   pickle.dump(VectorStore, f)
              #  st.write('Embeddings Completed')

        # Accept user questions/query
        query = st.text_area("Ask questions about your file:")
        #st.write(query)
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = Bedrock(model_id='anthropic.claude-v2:1', client=br_runtime)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write(response)
     
if __name__ == '__main__':
    main()