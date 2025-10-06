import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

# -------------------------------
# Load transformer model
# -------------------------------
def load_sentence_model():
    try:
        model = SentenceTransformer("./paraphrase-MiniLM-L6-v2", device="cpu")
        print("✅ Loaded local SentenceTransformer model")
    except Exception:
        print("⚠️ Local model not found, downloading from Hugging Face Hub...")
        model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2", device="cpu")

    # Dynamic quantization for speed/memory efficiency
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


MODEL = load_sentence_model()


# -------------------------------
# Job Recommendation System
# -------------------------------
class JobRecommendationSystem:
    def __init__(self, jobs_csv="JobsFE.csv"):
        # ✅ Load dataset: local first, else Hugging Face
        if os.path.exists(jobs_csv):
            print(f"✅ Loading dataset locally from {jobs_csv}")
            self.jobs_df = pd.read_csv(jobs_csv)
        else:
            print("⚠️ Local dataset not found, downloading from Hugging Face Hub...")
            dataset_path = hf_hub_download(
                repo_id="shreyan67/Job-Catalyst_AI",  # <-- make sure this repo exists on your HF account
                filename="JobsFE.csv"
            )
            self.jobs_df = pd.read_csv(dataset_path)

        # -------------------------------
        # Handle available columns safely
        # -------------------------------
        expected_cols = [
            "job_id", "workplace", "formatted_work_type", "remote_allowed",
            "min_salary", "max_salary", "med_salary", "currency", "pay_period",
            "position", "job_role_and_duties", "formatted_experience_level",
            "work_type", "apply_link", "job_posting_url", "application_type",
            "requisite_skill", "benefits", "industry_id", "employee_count",
            "company_website", "company_size", "country", "state", "city", "address"
        ]
        available_cols = [c for c in expected_cols if c in self.jobs_df.columns]
        self.job_info = self.jobs_df[available_cols]

        # -------------------------------
        # Build job description text field
        # -------------------------------
        text_cols = [
            "position", "job_role_and_duties", "requisite_skill", "benefits"
        ]
        text_cols = [c for c in text_cols if c in self.jobs_df.columns]

        self.jobs_texts = self.jobs_df[text_cols].fillna("").agg(" ".join, axis=1)

        # Deduplicate jobs
        self.jobs_df["combined_text"] = self.jobs_texts
        self.jobs_df = self.jobs_df.drop_duplicates(subset=["combined_text"]).reset_index(drop=True)
        self.jobs_texts = self.jobs_df["combined_text"]

        # -------------------------------
        # TF-IDF Vectorizer
        # -------------------------------
        print("⚡ Precomputing TF-IDF vectors...")
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.job_tfidf_matrix = self.vectorizer.fit_transform(self.jobs_texts)

        # -------------------------------
        # Sentence-BERT Embeddings
        # -------------------------------
        if os.path.exists("job_embeddings.npy"):
            print("✅ Loaded precomputed embeddings from job_embeddings.npy")
            self.job_embeddings = np.load("job_embeddings.npy")
        else:
            print("⚠️ job_embeddings.npy not found. Generating embeddings now...")
            self.job_embeddings = MODEL.encode(
                self.jobs_texts.tolist(),
                batch_size=64,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            np.save("job_embeddings.npy", self.job_embeddings)
            print("✅ Saved embeddings to job_embeddings.npy")

    # -------------------------------
    # Narrow down candidates with TF-IDF
    # -------------------------------
    def filter_top_jobs(self, resume_text, top_k=500):
        resume_vector = self.vectorizer.transform([resume_text])
        cosine_similarities = linear_kernel(resume_vector, self.job_tfidf_matrix).flatten()
        top_indices = cosine_similarities.argsort()[-top_k:][::-1]

        return (
            self.jobs_texts.iloc[top_indices],
            self.jobs_df.iloc[top_indices],
            self.job_embeddings[top_indices]
        )

    # -------------------------------
    # Final Job Recommendation
    # -------------------------------
    def recommend_jobs(self, resume_text, top_n=10):
        # Step 1: Narrow down using TF-IDF
        filtered_jobs_texts, filtered_jobs_df, filtered_embeddings = self.filter_top_jobs(resume_text)

        # Step 2: Resume embedding
        resume_embedding = MODEL.encode(resume_text, convert_to_numpy=True).reshape(1, -1)

        # Step 3: Similarities with Sentence-BERT
        similarities = cosine_similarity(resume_embedding, filtered_embeddings)[0]

        # Step 4: Top-N Ranking
        top_indices = similarities.argsort()[-top_n:][::-1]

        recommendations = []
        for idx in top_indices:
            job = filtered_jobs_df.iloc[idx]
            recommendations.append({
                "job_id": job.get("job_id", "N/A"),
                "position": job.get("position", "N/A"),
                "workplace": job.get("workplace", "N/A"),
                "working_mode": job.get("formatted_work_type", job.get("work_type", "N/A")),
                "salary": self._format_salary(job),
                "skills": job.get("requisite_skill", "N/A"),
                "benefits": job.get("benefits", "N/A"),
                "company_size": job.get("company_size", "N/A"),
                "location": f"{job.get('city', '')}, {job.get('state', '')}, {job.get('country', '')}",
                "apply_link": job.get("apply_link", job.get("job_posting_url", "N/A")),
                "similarity": round(float(similarities[idx]), 3)
            })

        return recommendations

    # -------------------------------
    # Helper: Salary formatting
    # -------------------------------
    def _format_salary(self, job):
        if pd.notna(job.get("min_salary")) and pd.notna(job.get("max_salary")):
            return f"{job['min_salary']} - {job['max_salary']} {job.get('currency', '')} / {job.get('pay_period', '')}"
        elif pd.notna(job.get("med_salary")):
            return f"{job['med_salary']} {job.get('currency', '')} / {job.get('pay_period', '')}"
        else:
            return "Not disclosed"
