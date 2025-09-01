import os
import re
import joblib
import streamlit as st
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import matplotlib.pyplot as plt

# --------------------------
# Helper function to extract text
# --------------------------
def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

# --------------------------
# Helper function to clean text
# --------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# --------------------------
# Extract topics from text
# --------------------------
def extract_topics(text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform([text])
    words = vectorizer.get_feature_names_out()
    scores = X.toarray().flatten()
    word_scores = dict(zip(words, scores))
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    return [w for w, s in sorted_words[:top_n]]

# --------------------------
# Streamlit App
# --------------------------
st.title("📘 Past Papers Topic Analyzer & Predictor")
st.write("Upload one or more `.docx` past papers to get topic predictions + analysis.")

uploaded_files = st.file_uploader(
    "Upload past paper(s)",
    type=["docx"],
    accept_multiple_files=True
)

if uploaded_files:
    all_text = ""

    for uploaded_file in uploaded_files:
        st.subheader(f"📄 {uploaded_file.name}")
        text = extract_text_from_docx(uploaded_file)
        cleaned = clean_text(text)

        topics = extract_topics(cleaned, top_n=10)
        st.write("**Top Predicted Topics:**", ", ".join(topics))

        # Show word frequency plot
        words = cleaned.split()
        word_counts = Counter(words)
        common_words = word_counts.most_common(10)

        fig, ax = plt.subplots()
        ax.bar([w for w, _ in common_words], [c for _, c in common_words])
        plt.xticks(rotation=45)
        plt.title("Most Frequent Words in Paper")
        st.pyplot(fig)

        all_text += " " + cleaned

    # --------------------------
    # Combined Analysis
    # --------------------------
    if len(uploaded_files) > 1:
        st.subheader("📊 Combined Analysis of All Papers")
        combined_topics = extract_topics(all_text, top_n=15)
        st.write("**Overall Predicted Topics Across Papers:**", ", ".join(combined_topics))

        words = all_text.split()
        word_counts = Counter(words)
        common_words = word_counts.most_common(15)

        fig, ax = plt.subplots()
        ax.bar([w for w, _ in common_words], [c for _, c in common_words])
        plt.xticks(rotation=45)
        plt.title("Most Frequent Words Across All Papers")
        st.pyplot(fig)

