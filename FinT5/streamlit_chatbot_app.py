import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Finance QA Chatbot", layout="centered")
st.title("Finance Question Answering Chatbot")
st.write("Ask me anything about finance!")

# --- Model Loading (Cached) ---
# Use st.cache_resource to load the model and tokenizer only once
# This is crucial for Streamlit apps as it prevents reloading on every rerun
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("yakul259/fint5-financeqa-customised")
    model = AutoModelForSeq2SeqLM.from_pretrained("yakul259/fint5-financeqa-customised")
    # Move model to GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

# Load model components
tokenizer, model, device = load_model_and_tokenizer()

# --- Answer Generation Function ---
def generate_answer(question):
    # Ensure the 'question: ' prefix as required by the T5-based model
    input_text = f"question: {question}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    with st.spinner("Generating answer..."):
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2 # Helps prevent repetitive text
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# --- Streamlit UI Elements ---
user_question = st.text_input("Enter your finance question here:", key="question_input")

if st.button("Get Answer"):
    if user_question:
        answer = generate_answer(user_question)
        st.subheader("Answer:")
        st.info(answer)
    else:
        st.warning("Please enter a question.")

st.markdown("---")
st.write("Powered by [yakul259/fint5-financeqa-customised](https://huggingface.co/yakul259/fint5-financeqa-customised)")
