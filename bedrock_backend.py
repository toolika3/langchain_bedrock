import json
import os
import pandas as pd

import boto3
import botocore

from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

boto_session = boto3.Session()
aws_region = boto_session.region_name

br_client = boto_session.client(service_name='bedrock', region_name=aws_region)
br_runtime = boto_session.client(service_name='bedrock-runtime', region_name=aws_region)

# Function for invoking model


def chatbot():
    llm = Bedrock(
        model_id='anthropic.claude-v2:1',
        client=br_runtime
    )
    return llm

# Test Code 
    #return demo_llm.predict(input_text)
#response = demo_chat("hat is your name?")
#print(response)

# Create a function for COnversationBufferMemory


def memory():
    llm_data = chatbot()
    memory = ConversationBufferMemory(llm=llm_data, max_token_limit=512)
    return memory

# Create a function for Conversation Chain - Input text + Memory


def conversation(input_text, memory):
    llm_chain_data = chatbot()
    llm_conversation = ConversationChain(llm=llm_chain_data, memory=memory, verbose=True)
    chat_reply = llm_conversation.predict(input=input_text)
    return chat_reply