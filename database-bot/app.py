import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq
import urllib

# Page setup
st.set_page_config(page_title="LangChain SQL Chat", page_icon="🦜", layout="centered")

st.markdown("<h1 style='text-align: center;'>🦜 LangChain: Chat with Your SQL Database</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; font-size: 16px;'>Ask questions in plain English. Works with <b>SQLite</b>, <b>MySQL</b>, or <b>MSSQL</b>.</div><br>", unsafe_allow_html=True)

# Constants
LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"
MSSQL = "USE_MSSQL"

# Sidebar: DB Option
st.sidebar.markdown("## 🗃️ Database Connection")
radio_opt = [
    "Use SQLite 3 Database - `student.db`",
    "Connect to MySQL Database",
    "Connect to MSSQL Server"
]
selected_opt = st.sidebar.radio("Choose the DB you want to chat with:", options=radio_opt)

# Credentials input
if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("🔌 MySQL Host")
    mysql_user = st.sidebar.text_input("👤 MySQL User")
    mysql_password = st.sidebar.text_input("🔑 MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("📂 MySQL Database")
elif radio_opt.index(selected_opt) == 2:
    db_uri = MSSQL
    mssql_host = st.sidebar.text_input("🔌 MSSQL Server Name (e.g., DESKTOP\\SQLEXPRESS)")
    mssql_user = st.sidebar.text_input("👤 MSSQL User")
    mssql_password = st.sidebar.text_input("🔑 MSSQL Password", type="password")
    mssql_db = st.sidebar.text_input("📂 MSSQL Database")
else:
    db_uri = LOCALDB

# Sidebar: API Key
st.sidebar.markdown("---")
st.sidebar.markdown("## 🔐 Authentication")
api_key = st.sidebar.text_input("🧠 Groq API Key", type="password")

# Checks
if not api_key:
    st.warning("Please enter your Groq API key to continue.")

# Initialize LLM
llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None,
                 mssql_host=None, mssql_user=None, mssql_password=None, mssql_db=None):
    
    if db_uri == LOCALDB:
        dbfilepath = (Path(__file__).parent / "student.db").absolute()
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite://", creator=creator))
    
    elif db_uri == MYSQL:
        if not all([mysql_host, mysql_user, mysql_password, mysql_db]):
            st.error("❌ Please fill in all MySQL fields.")
            st.stop()
        return SQLDatabase(
            create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}")
        )

    elif db_uri == MSSQL:
        if not all([mssql_host, mssql_user, mssql_password, mssql_db]):
            st.error("❌ Please fill in all MSSQL fields.")
            st.stop()
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={mssql_host};"
            f"DATABASE={mssql_db};"
            f"UID={mssql_user};"
            f"PWD={mssql_password}"
        )
        encoded_conn = urllib.parse.quote_plus(conn_str)
        return SQLDatabase(
            create_engine(f"mssql+pyodbc:///?odbc_connect={encoded_conn}")
        )

# Connect DB
if api_key:
    if db_uri == MYSQL:
        db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
    elif db_uri == MSSQL:
        db = configure_db(db_uri, mssql_host=mssql_host, mssql_user=mssql_user,
                          mssql_password=mssql_password, mssql_db=mssql_db)
    else:
        db = configure_db(db_uri)

    # Toolkit & Agent
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
    )

    # Initialize chat history
    if "messages" not in st.session_state or st.sidebar.button("🧹 Clear Chat History"):
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    # Show chat history
    for msg in st.session_state.messages:
        bg_color = "#3F485B" if msg['role'] == 'assistant' else "#2D5580"
        with st.chat_message(msg["role"]):
            st.markdown(
                f"<div style='padding: 8px 12px; border-radius: 8px; color: white; background-color: {bg_color}'>{msg['content']}</div>",
                unsafe_allow_html=True
            )
        st.markdown("<hr>", unsafe_allow_html=True)

    # Chat Input
    user_query = st.chat_input("Type your question about the database...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(
                f"<div style='padding: 8px 12px; border-radius: 8px; background-color: #2D5580; color: white'>{user_query}</div>",
                unsafe_allow_html=True
            )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container())
            try:
                response = agent.run(user_query, callbacks=[st_cb])
            except Exception as e:
                response = f"❌ Error: {e}"
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(
                f"<div style='padding: 8px 12px; border-radius: 8px; background-color: #3F485B; color: white'>{response}</div>",
                unsafe_allow_html=True
            )
