import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Streamlit UI
st.set_page_config(page_title="CodeGuru", layout="centered")
st.title("👨‍💻 CodeGuru - Your AI Coding Assistant")
st.markdown("Ask any programming or code-related question.")

# Prompt for Groq API Key
groq_api_key = st.text_input("🔑 Enter your Groq API Key", type="password")

# Prompt for Question
user_input = st.text_area("💬 Your code question:")

# Model and Prompt Setup
if groq_api_key and user_input:
    with st.spinner("Generating code solution..."):
        try:
            # Load model
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model="mixtral-8x7b-32768"  # Best for code & reasoning
            )

            # Prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", 
                 "You are CodeGuru, an expert programming assistant created by ARN. "
                 "Help users with code, explain bugs, and provide code examples when needed."),
                ("user", "{question}")
            ])

            chain = prompt | llm | StrOutputParser()

            response = chain.invoke({"question": user_input})
            st.success(response)

        except Exception as e:
            st.error(f"❌ Error: {e}")
else:
    st.info("Enter your Groq API Key and a coding question to begin.")
