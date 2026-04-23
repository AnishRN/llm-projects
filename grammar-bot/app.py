import streamlit as st
import language_tool_python
from transformers import pipeline
import pandas as pd
import plotly.express as px
import re

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Grammar AI Tool", layout="wide")

# -------------------------
# Custom CSS (Better UI)
# -------------------------
st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    .title {text-align: center; font-size: 40px; color: #4CAF50;}
    .subtitle {text-align: center; color: #aaa;}
    .box {padding: 15px; border-radius: 10px; background-color: #1c1f26;}
    .error {background-color: rgba(255,0,0,0.3); padding: 2px 5px; border-radius: 5px;}
    .correct {color: #00FFAA;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 Grammar Error Detection Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enhancing Academic & Professional Communication</div>', unsafe_allow_html=True)

# -------------------------
# Load Models
# -------------------------
@st.cache_resource
def load_tool():
    return language_tool_python.LanguageTool('en-US')

@st.cache_resource
def load_corrector():
    return pipeline("text2text-generation", model="vennify/t5-base-grammar-correction")

tool = load_tool()
corrector = load_corrector()

# -------------------------
# Input Section
# -------------------------
st.markdown("### ✍️ Enter Text")
text = st.text_area("", height=200)

uploaded_file = st.file_uploader("📂 Upload .txt file", type=["txt"])
if uploaded_file:
    text = uploaded_file.read().decode("utf-8")

# -------------------------
# Highlight Function
# -------------------------
def highlight_errors(text, matches):
    offset_shift = 0
    highlighted = text

    for match in matches:
        start = match.offset + offset_shift
        end = start + match.errorLength

        wrong_text = highlighted[start:end]
        replacement = match.replacements[0] if match.replacements else "?"

        new = f'<span class="error" title="Suggestion: {replacement}">{wrong_text}</span>'
        highlighted = highlighted[:start] + new + highlighted[end:]

        offset_shift += len(new) - len(wrong_text)

    return highlighted

# -------------------------
# Analyze Button
# -------------------------
if st.button("🚀 Analyze Text"):
    if text.strip() == "":
        st.warning("Please enter text.")
    else:
        matches = tool.check(text)

        # -------------------------
        # Highlighted Text
        # -------------------------
        highlighted_text = highlight_errors(text, matches)

        # -------------------------
        # Error Table
        # -------------------------
        errors = []
        for match in matches:
            severity = "Low"
            if match.ruleIssueType == "grammar":
                severity = "High"
            elif match.ruleIssueType == "misspelling":
                severity = "Medium"

            errors.append({
                "Error": text[match.offset:match.offset + match.errorLength],
                "Message": match.message,
                "Suggestion": ", ".join(match.replacements[:3]),
                "Type": match.ruleIssueType,
                "Severity": severity
            })

        df = pd.DataFrame(errors)

        # -------------------------
        # Corrected Text
        # -------------------------
        corrected = corrector(text, max_length=512)[0]['generated_text']

        # -------------------------
        # Layout
        # -------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📄 Original (Highlighted)")
            st.markdown(f'<div class="box">{highlighted_text}</div>', unsafe_allow_html=True)

            st.markdown("### ❌ Errors Found")
            if not df.empty:
                st.dataframe(df, use_container_width=True)
            else:
                st.success("No errors found!")

        with col2:
            st.markdown("### ✅ Corrected Text")
            st.markdown(f'<div class="box correct">{corrected}</div>', unsafe_allow_html=True)

        # -------------------------
        # Metrics
        # -------------------------
        st.markdown("### 📊 Analysis Summary")

        total_errors = len(errors)
        words = len(text.split())
        accuracy = round((1 - total_errors / words) * 100, 2) if words > 0 else 100

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Errors", total_errors)
        m2.metric("Word Count", words)
        m3.metric("Accuracy", f"{accuracy}%")

        # -------------------------
        # Visualization
        # -------------------------
        if not df.empty:
            st.markdown("### 📈 Error Distribution")

            fig = px.pie(df, names="Type", title="Error Types Distribution")
            st.plotly_chart(fig, use_container_width=True)
