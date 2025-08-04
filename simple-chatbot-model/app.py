import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# ✅ Get HF token from Streamlit secrets (secure)
os.environ["HUGGINGFACE_API_KEY"] = st.secrets["HF_TOKEN"]

# ✅ Set embeddings (forces use of CPU by not overriding any device)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "Question:{question}"),
])

# UI
st.title("Simple Chatbot using Groq + HuggingFace")

input_text = st.text_input("Ask a question:")

# Use Groq's hosted LLM instead of local Ollama
llm = ChatGroq(groq_api_key=st.secrets["GROQ_API_KEY"], model="Gemma2-9B-it")

# Build chain
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Generate response
if input_text:
    with st.spinner("Generating response..."):
        try:
            response = chain.invoke({"question": input_text})
            st.write("Response:", response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
