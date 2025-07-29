import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Smart Career Counsellor", page_icon="🎓", layout="centered")

st.markdown("<h1 style='text-align: center;'>🎓 Smart Career Counsellor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Describe your interests and let AI recommend a career that fits you best! 💡💼</p>", unsafe_allow_html=True)
st.markdown("---")

# Load model and vectorizer
model = joblib.load("../model/career_model.pkl")
vectorizer = joblib.load("../model/tfidf_vectorizer.pkl")


user_input = st.text_area("📝 What do you enjoy doing? What subjects or activities excite you?", height=150)

if user_input:
    with st.spinner("Analyzing your interests... 🔍"):
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        probabilities = model.predict_proba(input_vec)[0]

    st.success(f"🎯 Based on your input, you could explore a career in: **{prediction}**")

    st.markdown("#### 📊 Confidence Scores")
    results = pd.DataFrame({
        "Career": model.classes_,
        "Confidence": [f"{p*100:.2f}%" for p in probabilities]
    }).sort_values(by="Confidence", ascending=False).reset_index(drop=True)

    st.table(results)

    st.markdown("---")
    st.markdown("💡 _Tip: Try describing different passions to see what careers AI recommends!_")

st.markdown("<footer style='text-align: center; color: grey;'>Made with ❤️ by Sanjana</footer>", unsafe_allow_html=True)
