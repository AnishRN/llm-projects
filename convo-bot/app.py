import os
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS  # ✅ Replacing Chroma with FAISS

# UI setup
st.set_page_config(page_title="Conversational PDF Chatbot", layout="centered")
st.title("📚 Conversational RAG with PDF Uploads")
st.markdown("Chat with uploaded PDF documents and retain conversation history across sessions.")

# Ask for user API keys
hf_token = st.text_input("🔐 Enter your **HuggingFace API key**", type="password")
groq_api_key = st.text_input("🔐 Enter your **Groq API key**", type="password")

if hf_token and groq_api_key:
    # Set HuggingFace API key in environment
    os.environ["HUGGINGFACE_API_KEY"] = hf_token

    # Embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # LLM model from Groq
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-2-9b-it")

    # Session ID
    session_id = st.text_input("🗂️ Session ID", value="default_session")

    # Initialize session store
    if 'store' not in st.session_state:
        st.session_state.store = {}

    # PDF Upload
    uploaded_files = st.file_uploader("📄 Upload one or more PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temp_pdf = "./temp.pdf"
            with open(temp_pdf, "wb") as file:
                file.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)

        # Split and embed using FAISS
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # History-aware retriever
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and a new user question, rewrite the question as standalone."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # Response style selector
        response_style = st.selectbox("📝 Choose response length", ["Concise", "Elaborate", "Detailed"], index=1)
        if response_style == "Concise":
            instruction = "Answer in 2–3 concise sentences. Avoid repetition."
        elif response_style == "Elaborate":
            instruction = "Answer in one informative paragraph. Avoid bullet points."
        else:
            instruction = "Answer in 1–2 paragraphs, and include key points as bullet points if useful."

        # QA Prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are an assistant using the following context to answer user questions. "
                       f"If unsure, say 'I don't know'. {instruction}\n\n{{context}}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # History setup
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        from langchain_core.runnables.history import RunnableWithMessageHistory
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Input and response
        st.markdown("---")
        user_input = st.text_input("💬 Ask a question from the PDF:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )

            # Show assistant response
            st.markdown("### 🤖 Assistant Response:")
            st.success(response["answer"])

            # Chat history
            with st.expander("🕘 View Chat History"):
                for msg in session_history.messages:
                    role = "🧑‍💻 You" if msg.type == "human" else "🤖 Assistant"
                    st.markdown(f"**{role}:** {msg.content}")
else:
    st.warning("⚠️ Please enter both HuggingFace and Groq API keys to use the chatbot.")
