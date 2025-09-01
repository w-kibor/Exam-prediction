import os
import re
import string
import streamlit as st
from docx import Document
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# --------------------------
# Text Preprocessing
# --------------------------
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already available
nltk.download('stopwords')

stop_words = set(stopwords.words("english"))

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = [w for w in text.split() if w not in stop_words and not w.isdigit()]
    return " ".join(words)

# --------------------------
# Topic Extraction (Bigrams + Trigrams)
# --------------------------
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_subtopics(texts, top_n=10):
    custom_stopwords = [
        "marks", "section", "question", "answer",
        "explain", "state", "give", "paper"
    ]

    vectorizer = TfidfVectorizer(
        ngram_range=(2, 3),  # look for 2-3 word phrases
        stop_words="english",  # built-in stopwords
        max_features=5000
    )

    X = vectorizer.fit_transform(texts)
    freqs = zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0])

    # filter out custom stopwords manually
    filtered_freqs = [(word, freq) for word, freq in freqs if word not in custom_stopwords]

    sorted_freqs = sorted(filtered_freqs, key=lambda x: -x[1])[:top_n]
    return sorted_freqs


# --------------------------
# Streamlit App
# --------------------------
st.title("📘 Past Papers Topic Analyzer & Predictor")
st.write("Upload one or more `.docx` past papers to get **sub-topic predictions + analysis**.")

uploaded_files = st.file_uploader(
    "Upload past paper(s)",
    type=["docx"],
    accept_multiple_files=True
)

if uploaded_files:
    all_texts = []

    for uploaded_file in uploaded_files:
        st.subheader(f"📄 {uploaded_file.name}")
        raw_text = extract_text_from_docx(uploaded_file)
        cleaned = preprocess(raw_text)
        all_texts.append(cleaned)

        # Extract subtopics for this paper
        subtopics = extract_subtopics([cleaned], top_n=10)
        st.write("**Top Predicted Sub-Topics:**")
        for topic, score in subtopics:
            st.write(f"- {topic} ({score:.2f})")

        # Show most frequent words (optional, for insight)
        words = cleaned.split()
        word_counts = Counter(words)
        common_words = word_counts.most_common(10)

        fig, ax = plt.subplots()
        ax.bar([w for w, _ in common_words], [c for _, c in common_words])
        plt.xticks(rotation=45)
        plt.title("Most Frequent Words in Paper")
        st.pyplot(fig)

    # --------------------------
    # Combined Analysis
    # --------------------------
    if len(all_texts) > 1:
        st.subheader("📊 Combined Analysis of All Papers")
        combined_subtopics = extract_subtopics(all_texts, top_n=15)
        st.write("**Overall Predicted Sub-Topics Across Papers:**")
        for topic, score in combined_subtopics:
            st.write(f"- {topic} ({score:.2f})")

        combined_words = " ".join(all_texts).split()
        word_counts = Counter(combined_words)
        common_words = word_counts.most_common(15)

        fig, ax = plt.subplots()
        ax.bar([w for w, _ in common_words], [c for _, c in common_words])
        plt.xticks(rotation=45)
        plt.title("Most Frequent Words Across All Papers")
        st.pyplot(fig)
