from itertools import chain
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Set HuggingFace token from Streamlit secrets
os.environ['HUGGINGFACE_API_KEY'] = st.secrets["HF_TOKEN"]

# Create embeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

##Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "Question:{question}"),
    ]
)

st.title("Simple Chatbot with Ollama")

input_text = st.text_input("Ask a question:")

##Model Initialization
llm = ChatGroq(groq_api_key=st.secrets["GROQ_API_KEY"], model="Gemma2-9B-it")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    with st.spinner("Generating response..."):
        try:
            response = chain.invoke({"question": input_text})
            st.write("Response:", response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            

            

