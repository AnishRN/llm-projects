import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# ------------------------------
# Load Model Once
# ------------------------------
@st.cache_resource
def load_model():
    model_name = "yakul259/fint5-financeqa-customised"

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return tokenizer, model, device


tokenizer, model, device = load_model()

# ------------------------------
# Fallback Financial Context
# ------------------------------
FALLBACK_CONTEXT = (
    "Financial terms: Equity share capital is the amount raised by issuing equity shares. "
    "Net profit margin measures profitability as a percentage of revenue. "
    "Earnings per share (EPS) shows profit per share. "
    "Return on equity (ROE), return on assets (ROA), debt-equity ratio, "
    "and asset turnover ratio indicate a company's performance."
)

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Finance Chatbot", page_icon="💹", layout="centered")
st.title("💬 Finance Chatbot (T5-small Fine-Tuned)")

st.markdown("""
Your model was fine-tuned to answer **questions based on context**,  
so this demo automatically injects **fallback finance context**  
when the user leaves it empty.  
""")

question = st.text_input("📌 Enter your Question")
user_context = st.text_area("📄 Optional: Add Context (e.g., financial report text)")

if st.button("🔍 Generate Answer"):

    if not question.strip():
        st.warning("Please enter a question.")
    else:

        # Decide which context to use
        context_to_use = user_context.strip() if user_context.strip() else FALLBACK_CONTEXT

        # Prepare model input
        input_text = f"{question} context: {context_to_use}"

        with st.spinner("Generating answer..."):

            enc = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)

            out_ids = model.generate(
                enc.input_ids,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )

            answer = tokenizer.decode(out_ids[0], skip_special_tokens=True)

        st.success("✅ Answer Generated")
        st.markdown(f"### 📌 **{answer}**")

st.markdown("---")
st.markdown("""
✅ Works **with or without context**  
✅ Uses fine-tuned **T5-small FinanceQA**  
✅ Best for demos and presentations  
""")
