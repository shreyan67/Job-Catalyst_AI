import streamlit as st
import fitz  # PyMuPDF for PDF extraction
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from model import JobRecommendationSystem

# ----------------- CACHE HEAVY STUFF -----------------

@st.cache_resource
def load_model():
    """Load and quantize the SentenceTransformer model once"""
    try:
        model = SentenceTransformer("./paraphrase-MiniLM-L6-v2", device="cpu")  # local
    except Exception:
        model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2", device="cpu")  # fallback
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

@st.cache_resource
def load_recommender():
    """Load recommender system once with cached embeddings + FAISS index"""
    return JobRecommendationSystem("JobsFE.csv")

MODEL = load_model()
recommender = load_recommender()

# ----------------- STREAMLIT APP -----------------

st.set_page_config(page_title="AI Job Recommender", page_icon="💼", layout="wide")

st.markdown(
    """
    <style>
    .recommend-card {
        padding: 20px;
        border-radius: 15px;
        background-color: #f9f9f9;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    .job-title {
        font-size: 20px;
        font-weight: 700;
        color: #2c3e50;
    }
    .company-name {
        font-size: 16px;
        font-weight: 500;
        color: #16a085;
    }
    .salary {
        font-size: 15px;
        font-weight: 500;
        color: #e67e22;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💼 AI-Powered Job Recommendation System")

st.write(
    "📄 Upload your resume as a **PDF file** and get tailored job recommendations with direct apply links."
)

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
            job_results = recommender.recommend_jobs(resume_text, top_n=20)

        st.success(f"✅ Found {len(job_results)} job recommendations for you!")

        # Display recommended jobs
        for i, job in enumerate(job_results, start=1):
            with st.container():
                st.markdown('<div class="recommend-card">', unsafe_allow_html=True)

                # Title + Company
                st.markdown(f"<div class='job-title'> {i}. {job.get('position', 'N/A')} </div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='company-name'>🏢 {job.get('workplace', 'N/A')} ({job.get('formatted_work_type', 'N/A')})</div>",
                    unsafe_allow_html=True,
                )

                # Salary Range
                if job.get("salary_range") and "N/A" not in job.get("salary_range"):
                    st.markdown(f"<div class='salary'>💰 {job['salary_range']}</div>", unsafe_allow_html=True)

                # Experience
                if job.get("experience_level") and job.get("experience_level") != "N/A":
                    st.write(f"**🎯 Experience Level:** {job['experience_level']}")

                # Duties
                if job.get("job_role_and_duties"):
                    st.write(f"**📝 Duties:** {job['job_role_and_duties']}")

                # Skills
                if job.get("skills"):
                    st.write(f"**🛠 Required Skills:** {job['skills']}")

                # Benefits
                if job.get("benefits"):
                    st.write(f"**🎁 Benefits:** {job['benefits']}")

                # Location
                if job.get("location") and job.get("location").strip(", "):
                    st.write(f"**📍 Location:** {job['location']}")

                # Company size & employees
                if job.get("company_size") and job.get("company_size") != "N/A":
                    st.write(f"**🏢 Company Size:** {job['company_size']}")
                if job.get("employee_count") and job.get("employee_count") != "N/A":
                    st.write(f"**👥 Employees:** {job['employee_count']}")

                # Company website
                if job.get("company_website") and job.get("company_website") != "N/A":
                    st.markdown(f"[🌐 Company Website]({job['company_website']})", unsafe_allow_html=True)

                # Apply Links
                if job.get("apply_link") and job.get("apply_link") != "N/A":
                    st.markdown(f"[👉 Apply Here]({job['apply_link']})", unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please upload a valid PDF resume before proceeding.")
