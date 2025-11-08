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

# Cache the model loading
@st.cache_resource
def load_model():
    """Load the fine-tuned T5 model from HuggingFace Hub"""
    try:
        st.info("⏳ Loading fine-tuned model from HuggingFace Hub...")
        
        # Load custom fine-tuned model
        model_name = "yakul259/fint5-financeqa-customised"
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        st.success("✅ Loaded custom fine-tuned model from HuggingFace!")
        st.balloons()
        return model, tokenizer, "Custom Fine-Tuned (yakul259/fint5-financeqa-customised)"
        
    except Exception as e:
        st.warning(f"⚠️ Could not load custom model. Using base model instead.")
        
        try:
            model = T5ForConditionalGeneration.from_pretrained("t5-small")
            tokenizer = T5Tokenizer.from_pretrained("t5-small")
            st.success("✅ Loaded base T5-small model from HuggingFace")
            return model, tokenizer, "Base Model (t5-small)"
        except Exception as e2:
            st.error(f"❌ Failed to load model: {e2}")
            raise

def generate_response(question, context, model, tokenizer):
    """Generate answer using the T5 model"""
    input_text = f"{question} context: {context}"
    
    inputs = tokenizer(
        input_text, 
        max_length=512, 
        truncation=True, 
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=128,
            num_beams=4,
            early_stopping=True,
            temperature=0.7,
            top_p=0.9
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# App header
st.title("💰 Finance QA Chatbot")
st.markdown("Ask financial questions with context, and get AI-powered answers!")

# Sidebar
with st.sidebar:
    st.header("ℹ️ Model Information")
    
    try:
        model, tokenizer, model_info = load_model()
        
        st.markdown(f"""
        **Model:** {model_info}
        
        **Framework:** PyTorch + Transformers
        **Task:** Financial Question Answering
        **Dataset:** FinanceQA (sweatSmile/FinanceQA)
        """)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
    
    st.divider()
    
    st.header("⚙️ Settings")
    show_context = st.checkbox("Show context in chat", value=True)
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Chat interface
st.subheader("💬 Chat")

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="message-header">👤 You</div>
            <div><strong>Question:</strong> {message["question"]}</div>
            {"<div><strong>Context:</strong> " + message["context"] + "</div>" if show_context and message.get("context") else ""}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <div class="message-header">🤖 Assistant</div>
            <div>{message["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

# Input section
st.divider()
col1, col2 = st.columns([3, 1])

with col1:
    question = st.text_input(
        "Ask a question:",
        placeholder="e.g., What is compound interest?",
        key="question_input"
    )

with col2:
    use_context = st.toggle("Add Context", value=True)

if use_context:
    context = st.text_area(
        "Provide context (optional but recommended):",
        placeholder="e.g., Compound interest is calculated on the initial principal...",
        height=100,
        key="context_input"
    )
else:
    context = ""

# Submit button
if st.button("📤 Send", type="primary"):
    if question.strip():
        st.session_state.messages.append({
            "role": "user",
            "question": question,
            "context": context
        })
        
        with st.spinner("Thinking..."):
            try:
                answer = generate_response(question, context, model, tokenizer)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
            except Exception as e:
                st.error(f"Error generating response: {e}")
        
        st.rerun()
    else:
        st.warning("Please enter a question!")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>Powered by Hugging Face Transformers & Streamlit</small>
    <br>
    <small>Model: yakul259/fint5-financeqa-customised | Dataset: sweatSmile/FinanceQA</small>
</div>
""", unsafe_allow_html=True)
