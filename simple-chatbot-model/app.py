import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceHubEmbeddings  # ✅ new import

# Load environment variables
load_dotenv()

# ✅ Set secrets (Streamlit Cloud style)
HF_TOKEN = st.secrets["HF_TOKEN"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# ✅ Set HuggingFaceHubEmbeddings (cloud-compatible; no torch backend)
embeddings = HuggingFaceHubEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=HF_TOKEN
)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "Question: {question}"),
])

# Streamlit UI
st.title("Simple Chatbot using Groq + HuggingFace API")

input_text = st.text_input("Ask a question:")

# Use Groq hosted model
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model="Gemma2-9B-it")

# Chain setup
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# On submit
if input_text:
    with st.spinner("Generating response..."):
        try:
            response = chain.invoke({"question": input_text})
            st.write("Response:", response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
