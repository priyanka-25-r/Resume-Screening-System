import matplotlib.pyplot as plt
import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample required skills (you can customize this)
required_skills = ["Python", "Pandas", "Scikit-learn", "Machine Learning", "Deep Learning", "Communication", "Problem Solving", "SQL"]

st.title("📄 Resume Screening System")
st.write("👋 Upload your resume and compare it with a job description.")

uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])
job_description = st.text_area("Paste the job description here:")

resume_text = ""

if uploaded_file is not None:
    st.success("✅ Resume uploaded successfully!")
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            resume_text += page.extract_text()

    st.subheader("📃 Extracted Resume Text:")
    st.write(resume_text)

if uploaded_file is not None and job_description:
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([resume_text, job_description])
    similarity_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    percentage = round(similarity_score * 100, 2)

    st.subheader("📊 Resume Match Score:")
    st.write(f"✅ Your resume matches **{percentage}%** of the job description.")

    # Skill Matching
    st.subheader("🧠 Skills Match Summary:")

    matched_skills = [skill for skill in required_skills if skill.lower() in resume_text.lower()]
    missing_skills = [skill for skill in required_skills if skill.lower() not in resume_text.lower()]

    col1, col2 = st.columns(2)
    with col1:
        st.success("✅ Skills Found in Resume:")
        for skill in matched_skills:
            st.markdown(f"- {skill}")
    with col2:
        st.error("❌ Skills Missing:")
        for skill in missing_skills:
            st.markdown(f"- {skill}")



