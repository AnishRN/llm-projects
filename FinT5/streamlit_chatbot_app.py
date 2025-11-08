import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Page configuration
st.set_page_config(
    page_title="Finance QA Chatbot",
    page_icon="💰",
    layout="centered"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextInput > div > div > input {
        background-color: white;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
    }
    .bot-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4CAF50;
    }
    .message-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_model():
    """Load the fine-tuned T5 model from HuggingFace Hub"""
    try:
        st.info("⏳ Loading fine-tuned model from HuggingFace Hub...")

        # Load the custom fine-tuned model
        model_name = "yakul259/fint5-financeqa-customised"
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)

        st.success("✅ Loaded custom fine-tuned model from HuggingFace!")
        st.balloons()
        return model, tokenizer, "Custom Fine-Tuned (yakul259/fint5-financeqa-customised)"

    except Exception as e:
        st.warning(f"⚠️ Error loading custom model: {e}")
        st.info("Falling back to base T5-small model...")

        try:
            model = T5ForConditionalGeneration.from_pretrained("t5-small")
            tokenizer = T5Tokenizer.from_pretrained("t5-small")
            st.success("✅ Loaded base T5-small model from HuggingFace")
            return model, tokenizer, "Base Model (t5-small)"
   
