import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceHubEmbeddings

# UI Setup
st.set_page_config(page_title="Conversational PDF Chatbot", layout="centered")
st.title("📚 Conversational RAG with PDF Uploads")
st.markdown("Chat with uploaded PDF documents and retain conversation history across sessions.")

# API Key Inputs
groq_api_key = st.text_input("🔐 Enter your **Groq API key**", type="password")
hf_token = st.text_input("🧠 Enter your **Hugging Face API token**", type="password")

if groq_api_key and hf_token:
    # Set embeddings using Hugging Face Hub (no torch device issues)
    embeddings = HuggingFaceHubEmbeddings(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=hf_token
    )

    # LLM (Groq)
    llm = ChatGroq(groq_api_key=groq_api_key, model="Gemma2-9b-It")

    # Session ID
    session_id = st.text_input("🗂️ Session ID", value="default_session")

    # History store
    if 'store' not in st.session_state:
        st.session_state.store = {}

    # PDF Upload
    uploaded_files = st.file_uploader("📄 Upload one or more PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temp_path = f"./temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_path)
            documents.extend(loader.load())

        # Split and Embed
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Contextualization Prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(
            llm=llm,
            retriever=retriever,
            prompt=contextualize_q_prompt
        )

        # Style of Answer
        response_style = st.selectbox("📝 Choose response length", ["Concise", "Elaborate", "Detailed"], index=1)

        if response_style == "Concise":
            instruction = "Answer in 2 to 3 concise sentences. Avoid repetition and keep it to the point."
        elif response_style == "Elaborate":
            instruction = "Answer in one full paragraph. Be clear and informative. Avoid bullet points."
        else:
            instruction = (
                "Answer in detail using 1 to 2 paragraphs. Include bullet points if needed. "
                "Maintain clarity and structure."
            )

        # QA Prompt
        system_prompt = (
            f"You are an assistant for question-answering tasks. "
            f"Use the following pieces of retrieved context to answer the question. "
            f"If you don't know the answer, say you don't know. {instruction}"
            "\n\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Ask a Question
        st.markdown("---")
        user_input = st.text_input("💬 Ask a question from the PDF:")

        if user_input:
            session_history = get_session_history(session_id)
            with st.spinner("Thinking..."):
                try:
                    response = conversational_rag_chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": session_id}}
                    )

                    # Display
                    st.markdown("### 🤖 Assistant Response:")
                    st.success(response["answer"])

                    # Chat History
                    with st.expander("🕘 View Chat History"):
                        for msg in session_history.messages:
                            role = "🧑‍💻 You" if msg.type == "human" else "🤖 Assistant"
                            st.markdown(f"**{role}:** {msg.content}")
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")
else:
    st.info("Please enter your Hugging Face and Groq API keys to begin.")
