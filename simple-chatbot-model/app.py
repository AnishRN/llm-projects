import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceHubEmbeddings

# Streamlit UI
st.title("🔍 Simple Chatbot using Groq + HuggingFace")

# User-provided API keys
hf_token = st.text_input("Enter your Hugging Face API token:", type="password")
groq_api_key = st.text_input("Enter your Groq API key:", type="password")

# Only continue if both keys are provided
if hf_token and groq_api_key:

    # Initialize embedding (uses Hugging Face API — no local torch)
    embeddings = HuggingFaceHubEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=hf_token
    )

    # Prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("user", "Question: {question}"),
    ])

    # LLM setup
    llm = ChatGroq(groq_api_key=groq_api_key, model="Gemma2-9B-it")

    # Chain
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    # User input for question
    input_text = st.text_input("Ask a question:")

    if input_text:
        with st.spinner("Generating response..."):
            try:
                response = chain.invoke({"question": input_text})
                st.write("🧠 Response:", response)
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
else:
    st.warning("Please enter both your Hugging Face and Groq API keys to continue.")
