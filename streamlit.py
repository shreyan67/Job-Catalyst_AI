import streamlit as st
import fitz  # PyMuPDF for PDF extraction
import pandas as pd
from model import JobRecommendationSystem
import torch
from sentence_transformers import SentenceTransformer

# ----------------- CACHE HEAVY STUFF -----------------
import os
@st.cache_resource
def load_model():
    """Load and quantize SentenceTransformer model once"""
    model = SentenceTransformer("./paraphrase-MiniLM-L6-v2", device="cpu")
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

@st.cache_resource
def load_recommender():
    """Load recommender system with enriched dataset"""
    if not os.path.exists("job_embeddings.npy"):
        with st.spinner("⚠️ Generating embeddings for the first time (~5–10 minutes). Future runs will be instant."):
            recommender = JobRecommendationSystem("JobsFE.csv")
    else:
        recommender = JobRecommendationSystem("JobsFE.csv")
    return recommender



MODEL = load_model()
recommender = load_recommender()

# ----------------- STREAMLIT APP -----------------

st.set_page_config(page_title="AI Job Recommender", page_icon="💼", layout="wide")

st.title("💼 AI-Powered Job Recommendation System (Enhanced)")

st.write("📄 Upload your resume as a **PDF file** and get tailored job recommendations with detailed company info & apply links.")

# File uploader for PDF resume
uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"], help="Only PDF resumes are supported.")

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF resume"""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return text.strip()

resume_text = ""
if uploaded_file:
    with st.spinner("⏳ Extracting text from your resume..."):
        resume_text = extract_text_from_pdf(uploaded_file)

if st.button("🔍 Recommend Jobs"):
    if resume_text:
        with st.spinner("🤖 Analyzing your resume and finding best matches..."):
            recommendations = recommender.recommend_jobs(resume_text, top_n=20)
            job_results = recommendations["recommended_jobs"]

        st.success(f"✅ Found {len(job_results)} job recommendations for you!")

        # Display recommended jobs
        for i, job in enumerate(job_results, start=1):
            with st.container():
                st.markdown(f"### {i}. {job['position']} at {job['workplace']}")

                st.write(f"**📍 Location:** {job['city']}, {job['state']}, {job['country']}")
                st.write(f"**🏢 Company Size:** {job.get('company_size', 'N/A')} employees")
                st.write(f"**👥 Employee Count:** {job.get('employee_count', 'N/A')}")
                st.write(f"**💼 Work Type:** {job['formatted_work_type']} ({job['work_type']})")
                st.write(f"**🧑‍💻 Experience Level:** {job['formatted_experience_level']}")

                if pd.notna(job['min_salary']) or pd.notna(job['max_salary']):
                    st.write(f"**💰 Salary:** {job['min_salary']} - {job['max_salary']} {job['currency']} ({job['pay_period']})")

                st.write(f"**📝 Duties:** {job['job_role_and_duties']}")
                st.write(f"**🛠 Skills:** {job['requisite_skill']}")
                st.write(f"**🎁 Benefits:** {job['benefits']}")
                st.write(f"**🏭 Industry IDs:** {job['industry_id']}")

                if pd.notna(job['apply_link']):
                    st.markdown(f"[👉 Apply Here]({job['apply_link']})")

                if pd.notna(job['company_website']):
                    st.markdown(f"[🌐 Company Website]({job['company_website']})")
    else:
        st.warning("⚠️ Please upload a valid PDF resume before proceeding.")
