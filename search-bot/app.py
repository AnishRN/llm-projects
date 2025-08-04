import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# LangChain Tool Setup
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# Page Config
st.set_page_config(page_title="LangChain Web Search Chat", page_icon="🔎", layout="centered")

# Title and Intro
st.markdown("<h1 style='text-align: center;'>🔎 LangChain - Web Search Chatbot</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 16px;'>
Talk to an AI agent that can search <b>Wikipedia</b>, <b>Arxiv</b>, and <b>the Web</b> in real time.<br>
Built with LangChain + Streamlit.
</div><br>""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## 🔧 Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password", help="You can get your API key from https://console.groq.com/keys")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi 👋 I'm a chatbot that can search the web. How can I help you today?"}
    ]

# Display chat messages
for msg in st.session_state.messages:
    bg_color = "#3F485B" if msg['role'] == 'assistant' else "#2D5580"
    with st.chat_message(msg["role"]):
        st.markdown(
            f"<div style='padding: 8px 12px; border-radius: 8px; color: white; background-color: {bg_color}'>{msg['content']}</div>",
            unsafe_allow_html=True
        )
        st.markdown("---")

# User input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f"<div style='padding: 8px 12px; border-radius: 8px; background-color: #2D5580; color: white'>{prompt}</div>", unsafe_allow_html=True)

    # Run LangChain Agent
    if api_key:
        llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
        tools = [search, arxiv, wiki]

        search_agent = initialize_agent(
            tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
            except Exception as e:
                response = f"❌ An error occurred: {e}"
            st.markdown(f"<div style='padding: 8px 12px; border-radius: 8px; background-color: #3F485B; color: white'>{response}</div>", unsafe_allow_html=True)
            st.session_state.messages.append({'role': 'assistant', "content": response})
    else:
        st.warning("Please enter your Groq API key in the sidebar before chatting.")
