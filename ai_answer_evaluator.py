import streamlit as st
import time
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Automated Answer Sheet Evaluator", layout="centered")

st.title("📝 Automated Answer Sheet Evaluator")
st.markdown("Upload or enter answers to evaluate using AI-based similarity.")
st.markdown("---")

# Upload Section
st.subheader("📂 Upload Text Files (Optional)")

uploaded_model = st.file_uploader("Upload Model Answer (.txt)", type=["txt"])
uploaded_student = st.file_uploader("Upload Student Answer (.txt)", type=["txt"])

# Manual Input Section
st.markdown("### ✍️ Or Enter Answers Manually")

model_answer = st.text_area("📘 Enter Model Answer", height=150)
student_answer = st.text_area("🧑‍🎓 Enter Student Answer", height=150)

# If files uploaded, override manual input
if uploaded_model is not None:
    model_answer = uploaded_model.read().decode("utf-8")

if uploaded_student is not None:
    student_answer = uploaded_student.read().decode("utf-8")

st.markdown("---")

if st.button("Evaluate Answer"):

    # 🔄 Spinner (AI thinking effect)
    with st.spinner("Evaluating answer... ⏳"):
        time.sleep(2)

    if model_answer.strip() == "" or student_answer.strip() == "":
        st.warning("Please provide both Model and Student answers.")
    else:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([model_answer, student_answer])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100

        # Human-like grading logic
        if similarity >= 85:
            marks = 10
            feedback = "🌟 Excellent answer! Very well explained."

        elif similarity >= 70:
            marks = 8
            feedback = "👏 Very good answer. Minor details could be added."

        elif similarity >= 50:
            marks = 6
            feedback = "👍 Good understanding but missing some important points."

        elif similarity >= 30:
            marks = 4
            feedback = "⚠️ Basic idea is present but explanation is weak."

        elif similarity >= 15:
            marks = 2
            feedback = "❌ Very limited understanding. Needs improvement."

        else:
            marks = 0
            feedback = "🚫 Completely irrelevant answer."

    # 📊 Animated Progress Bar
    progress = st.progress(0)
    for i in range(int(similarity)):
        time.sleep(0.01)
        progress.progress(i + 1)

    st.write(f"Similarity: {similarity}%")
    st.write(f"Marks: {marks}/10")
    st.write(feedback)

    # 🎉 Celebration
    if marks >= 8:
        st.balloons()