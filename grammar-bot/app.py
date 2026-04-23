import streamlit as st
import language_tool_python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import plotly.express as px
import nltk
import re

# -------------------------
# NLTK Setup
# -------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Grammar AI Tool", layout="wide")

# -------------------------
# Custom UI
# -------------------------
st.markdown("""
<style>
.main {background-color: #0e1117;}
.title {text-align:center; font-size:42px; color:#00FFA6;}
.subtitle {text-align:center; color:#aaa;}
.box {padding:15px; border-radius:10px; background:#1c1f26;}
.error {background-color:rgba(255,0,0,0.3); padding:3px; border-radius:5px;}
.correct {color:#00FFAA;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 Grammar Error Detection Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">NLP-based Academic Writing Assistant</div>', unsafe_allow_html=True)

# -------------------------
# Load Models
# -------------------------
@st.cache_resource
def load_tool():
    return language_tool_python.LanguageTool('en-US')

@st.cache_resource
def load_model():
    model_name = "vennify/t5-base-grammar-correction"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tool = load_tool()
tokenizer, model = load_model()

# -------------------------
# NLP Functions
# -------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    steps = {}

    # Step 1: Lowercase
    steps["Lowercasing"] = text.lower()

    # Step 2: Sentence Tokenization
    sentences = sent_tokenize(text)
    steps["Sentence Segmentation"] = sentences

    # Step 3: Word Tokenization
    words = word_tokenize(text)
    steps["Tokenization"] = words

    # Step 4: Stopword Removal
    filtered = [w for w in words if w.lower() not in stop_words]
    steps["Stopword Removal"] = filtered

    # Step 5: Lemmatization
    lemmas = [lemmatizer.lemmatize(w) for w in filtered]
    steps["Lemmatization"] = lemmas

    return steps

def correct_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def highlight_errors(text, matches):
    offset = 0
    for m in matches:
        start = m.offset + offset
        end = start + m.errorLength
        wrong = text[start:end]
        suggestion = m.replacements[0] if m.replacements else "?"
        tag = f'<span class="error" title="Suggestion: {suggestion}">{wrong}</span>'
        text = text[:start] + tag + text[end:]
        offset += len(tag) - len(wrong)
    return text

# -------------------------
# Input Section
# -------------------------
st.markdown("### ✍️ Enter Text")
text = st.text_area("", height=200)

uploaded_file = st.file_uploader("📂 Upload .txt file", type=["txt"])
if uploaded_file:
    text = uploaded_file.read().decode("utf-8")

# -------------------------
# MAIN EXECUTION
# -------------------------
if st.button("🚀 Analyze"):
    if text.strip() == "":
        st.warning("Enter some text first.")
    else:
        # NLP Preprocessing
        steps = preprocess(text)

        # Grammar Check
        matches = tool.check(text)
        highlighted = highlight_errors(text, matches)

        # Correction
        corrected = correct_text(text)

        # Error Table
        errors = []
        for m in matches:
            errors.append({
                "Error": text[m.offset:m.offset + m.errorLength],
                "Message": m.message,
                "Suggestion": ", ".join(m.replacements[:3]),
                "Type": m.ruleIssueType
            })
        df = pd.DataFrame(errors)

        # -------------------------
        # OUTPUT FIRST (Important)
        # -------------------------
        st.markdown("## ✅ Final Output")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📄 Original (Highlighted)")
            st.markdown(f'<div class="box">{highlighted}</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("### ✨ Corrected Text")
            st.markdown(f'<div class="box correct">{corrected}</div>', unsafe_allow_html=True)

        # Metrics
        st.markdown("### 📊 Summary")
        m1, m2 = st.columns(2)
        m1.metric("Errors", len(errors))
        m2.metric("Words", len(text.split()))

        # Chart
        if not df.empty:
            fig = px.bar(df, x="Type", title="Error Types")
            st.plotly_chart(fig, use_container_width=True)

        # -------------------------
        # STEPWISE NLP (AFTER OUTPUT)
        # -------------------------
        st.markdown("---")
        st.markdown("## 🔍 NLP Processing Steps")

        for step, value in steps.items():
            with st.expander(f"👉 {step}"):
                st.write(value)

        # Error Table at End
        if not df.empty:
            st.markdown("## ❌ Detailed Errors")
            st.dataframe(df, use_container_width=True)
