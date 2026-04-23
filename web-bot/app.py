import streamlit as st
import validators
import re
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# ─────────────────────────────────────────────────────────────
# 🎨 UI Styling
st.set_page_config(page_title="🦜 LangChain Summarizer", layout="centered")
st.markdown("""
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 10px 24px;
        border: none;
        border-radius: 8px;
    }
    .stTextInput input {
        padding: 0.75rem;
        font-size: 1rem;
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 🧠 Header
st.title("🦜 LangChain: Summarize YouTube or Website Content")
st.subheader("Paste a YouTube or Website URL below to get a summary.")

# ─────────────────────────────────────────────────────────────
# 🔐 API Key
with st.sidebar:
    st.header("🔐 API Settings")
    groq_api_key = st.text_input("Enter your Groq API Key", type="password")
    st.markdown("[Get your API key](https://console.groq.com/keys)", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 🔗 URL Input
generic_url = st.text_input("🔗 Enter a YouTube or Website URL")

# Prompt Template
prompt_template = """
Please summarize the following content in approximately 300 words:

{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# ─────────────────────────────────────────────────────────────
# 🔍 YouTube URL Validator
def is_valid_youtube_video(url):
    return re.match(r'(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w\-]{11}', url)

# ─────────────────────────────────────────────────────────────
# 🚀 Button Logic
if st.button("Summarize Now 🚀"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("❌ Please provide both a valid Groq API Key and a URL.")
    elif not validators.url(generic_url):
        st.error("⚠️ Please enter a valid URL that starts with http or https.")
    else:
        docs = []
        try:
            with st.spinner("⏳ Fetching and processing content..."):

                # Load content from YouTube
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    if not is_valid_youtube_video(generic_url):
                        st.error("⚠️ This is not a valid YouTube video URL.")
                    else:
                        try:
                            loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                            docs = loader.load()
                        except Exception as yt_error:
                            st.error("❌ Failed to load YouTube video content.")
                            st.exception(yt_error)

                # Load content from a website
                else:
                    try:
                        loader = UnstructuredURLLoader(
                            urls=[generic_url],
                            ssl_verify=False,
                            headers={"User-Agent": "Mozilla/5.0"}
                        )
                        docs = loader.load()
                    except Exception as web_error:
                        st.error("❌ Failed to load content from the website.")
                        st.exception(web_error)

                # Run summarization if documents loaded
                if docs:
                    llm = ChatGroq(model="llama3-70b-8192", groq_api_key=groq_api_key)
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    summary = chain.run(docs)
                    st.success("✅ Summary generated successfully!")
                    st.markdown(f"### 📄 Summary:\n{summary}")
                else:
                    st.warning("⚠️ No content found to summarize.")

        except Exception as e:
            st.error("An unexpected error occurred.")
            st.exception(e)
