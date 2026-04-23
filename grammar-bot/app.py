import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import pandas as pd
import plotly.express as px
import difflib

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
# UI Config
# -------------------------
st.set_page_config(page_title="Grammar AI Tool", layout="wide")

st.markdown("""
<style>
.title {text-align:center; font-size:40px; color:#00FFA6;}
.box {padding:15px; border-radius:10px; background:#1c1f26;}
.error {background-color:rgba(255,0,0,0.3); padding:3px; border-radius:5px;}
.correct {color:#00FFAA;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 Grammar AI Tool (No Java)</div>', unsafe_allow_html=True)

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    model_name = "vennify/t5-base-grammar-correction"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# -------------------------
# NLP Tools
# -------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    steps = {}
    steps["Lowercase"] = text.lower()
    steps["Sentences"] = sent_tokenize(text)
    words = word_tokenize(text)
    steps["Tokens"] = words
    filtered = [w for w in words if w.lower() not in stop_words]
    steps["Stopwords Removed"] = filtered
    lemmas = [lemmatizer.lemmatize(w) for w in filtered]
    steps["Lemmatized"] = lemmas
    return steps

def correct_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------------
# Highlight Differences
# -------------------------
def highlight_diff(original, corrected):
    diff = difflib.ndiff(original.split(), corrected.split())
    result = []

    for word in diff:
        if word.startswith("- "):
            result.append(f"<span class='error'>{word[2:]}</span>")
        elif word.startswith("+ "):
            result.append(f"<span class='correct'>{word[2:]}</span>")
        else:
            result.append(word[2:])

    return " ".join(result)

# -------------------------
# INPUT
# -------------------------
text = st.text_area("Enter Text", height=200)

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Enter text")
    else:
        steps = preprocess(text)
        corrected = correct_text(text)

        # -------------------------
        # OUTPUT FIRST
        # -------------------------
        st.markdown("## ✅ Output")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Original (Errors Highlighted)")
            highlighted = highlight_diff(text, corrected)
            st.markdown(f"<div class='box'>{highlighted}</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("### Corrected")
            st.markdown(f"<div class='box correct'>{corrected}</div>", unsafe_allow_html=True)

        # -------------------------
        # Metrics
        # -------------------------
        st.markdown("### 📊 Summary")
        errors = sum(1 for w in difflib.ndiff(text.split(), corrected.split()) if w.startswith("- "))
        words = len(text.split())

        m1, m2 = st.columns(2)
        m1.metric("Errors", errors)
        m2.metric("Words", words)

        # -------------------------
        # NLP Steps
        # -------------------------
        st.markdown("---")
        st.markdown("## 🔍 NLP Processing")

        for k, v in steps.items():
            with st.expander(k):
                st.write(v)
