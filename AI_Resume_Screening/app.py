import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Extract text from PDF
def extract_text(pdf_file):
    text = ""

    pdf = PdfReader(pdf_file)

    for page in pdf.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text

# Skills database
skills_db = [
    "python",
    "java",
    "sql",
    "machine learning",
    "deep learning",
    "data analysis",
    "tableau",
    "power bi",
    "aws",
    "excel",
    "html",
    "css",
    "javascript"
]

# Skill extraction
def extract_skills(text):
    found_skills = []

    text = text.lower()

    for skill in skills_db:
        if skill in text:
            found_skills.append(skill)

    return found_skills

# Resume matching
def calculate_match(resume_text, job_description):

    docs = [resume_text, job_description]

    vectorizer = TfidfVectorizer()

    matrix = vectorizer.fit_transform(docs)

    similarity = cosine_similarity(matrix[0], matrix[1])

    score = round(similarity[0][0] * 100, 2)

    return score

# UI
st.title("AI Resume Screening System")

st.write("Upload Resume PDF and compare it with a Job Description")

resume_file = st.file_uploader(
    "Upload Resume",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste Job Description"
)

if st.button("Analyze Resume"):

    if resume_file and job_description:

        resume_text = extract_text(resume_file)

        skills = extract_skills(resume_text)

        score = calculate_match(
            resume_text,
            job_description
        )

        st.success(f"Match Score: {score}%")

        st.subheader("Skills Found")

        if skills:
            for skill in skills:
                st.write("✅", skill)
        else:
            st.write("No skills found")

    else:
        st.warning(
            "Please upload a resume and enter a job description"
        )