import streamlit as st
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Page configuration
st.set_page_config(
    page_title="HateGuard-AI",
    page_icon="🛡️",
    layout="centered"
)

# Basic sample dataset for demonstration
texts = [
    "You are wonderful",
    "Thank you for your help",
    "Have a great day",
    "I respect everyone",
    "You are stupid",
    "I hate you",
    "You are an idiot",
    "Nobody likes you",
    "You are amazing",
    "Great work",
    "Go away loser",
    "You are useless"
]

labels = [
    0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1
]

# Create machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# Train demonstration model
model.fit(texts, labels)

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# App title
st.title("🛡️ HateGuard-AI")
st.subheader("AI-Powered Hate Speech Detection System")

st.write(
    "Enter a comment below to analyze whether "
    "it may contain hateful or abusive language."
)

# Input box
comment = st.text_area(
    "Enter your comment:",
    placeholder="Type a comment here...",
    height=150
)

# Analyze button
if st.button("Analyze Comment", type="primary"):

    if comment.strip() == "":
        st.warning("Please enter a comment first.")

    else:
        processed_comment = clean_text(comment)

        prediction = model.predict([processed_comment])[0]
        probabilities = model.predict_proba(
            [processed_comment]
        )[0]

        confidence = max(probabilities) * 100

        st.markdown("---")
        st.subheader("Analysis Result")

        if prediction == 1:
            st.error("⚠️ Potentially Abusive or Harmful Language")
            st.write(
                "The comment may contain offensive "
                "or abusive language."
            )
        else:
            st.success("✅ Non-Abusive Language Detected")
            st.write(
                "The comment appears non-abusive "
                "according to the demonstration model."
            )

        st.metric(
            label="Model Confidence",
            value=f"{confidence:.2f}%"
        )

        st.caption(
            "This is a school-project demonstration. "
            "The model may make mistakes and should "
            "not be used to automatically punish users."
        )

# Sidebar
st.sidebar.title("About HateGuard-AI")
st.sidebar.write(
    "HateGuard-AI uses Natural Language Processing "
    "(NLP), TF-IDF and Logistic Regression "
    "to classify text comments."
)

st.sidebar.markdown("### Features")
st.sidebar.write("- Text input")
st.sidebar.write("- AI-based classification")
st.sidebar.write("- Confidence score")
st.sidebar.write("- Simple interactive interface")

st.sidebar.markdown("### Technology")
st.sidebar.write("Python, Streamlit, Scikit-learn")
