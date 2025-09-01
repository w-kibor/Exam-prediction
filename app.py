import os
import re
import joblib
import streamlit as st
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import matplotlib.pyplot as plt

# ----------------------------
# Topic tagging function
# ----------------------------
def tag_question_with_topics(question_text):
    topic_map = {
        "Audio & Sound Processing": ['audio', 'sound', 'sampled', 'rate', 'second', 'bits', 'bit', 'calculate'],
        "Image & Video Processing": ['image', 'pixel', 'video', 'frames', 'size'],
        "Compression & Coding": ['compression', 'huffman', 'code', 'coding', 'codeword', 'messages', 'data'],
        "Multimedia Authoring & Editing": ['authoring', 'editing', 'interface', 'text'],
        "Multimedia Systems & Theory": ['multimedia', 'systems', 'layer', 'set', 'types', 'used', 'different', 'terms', 'case', 'basic', 'derive', 'distinguish']
    }
    tags = []
    text = question_text.lower()
    for topic, keywords in topic_map.items():
        if any(kw in text for kw in keywords):
            tags.append(topic)
    return tags

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📘 Past Papers Topic Analyzer & Predictor")
st.write("Upload a `.docx` past paper and get topic predictions + analysis.")

uploaded = st.file_uploader("Upload a past paper (.docx)", type="docx")

if uploaded:
    # Extract text
    doc = Document(uploaded)
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip() != ""])

    # Split into questions (basic split by numbering)
    raw_questions = re.split(r'\n?\s*\d{1,2}\.\s+', text)
    questions = []
    for i, q in enumerate(raw_questions):
        q = q.strip()
        if len(q) > 30 and "UNIVERSITY EXAMINATIONS" not in q:
            questions.append({"No": i, "Text": q, "Topics": tag_question_with_topics(q)})

    # Show sample extracted questions
    st.subheader("📄 Extracted Questions (sample)")
    for q in questions[:5]:
        st.markdown(f"**Q{q['No']}**: {q['Text'][:200]}...")
        st.markdown(f"_Predicted Topics: {', '.join(q['Topics']) if q['Topics'] else 'None'}_")
        st.write("---")

    # Topic frequency
    all_topics = []
    for q in questions:
        all_topics.extend(q['Topics'])
    topic_counts = Counter(all_topics)

    st.subheader("📊 Topic Frequency")
    if topic_counts:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(topic_counts.keys(), topic_counts.values())
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    else:
        st.info("No topics detected in this file.")

    # TF-IDF keywords (optional quick insight)
    st.subheader("🔑 Top Keywords")
    question_texts = [q["Text"] for q in questions]
    if question_texts:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=20)
        X = vectorizer.fit_transform(question_texts)
        keywords = vectorizer.get_feature_names_out()
        st.write(", ".join(keywords))
