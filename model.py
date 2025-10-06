import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# ---------------- Load Sentence-BERT Model ----------------
def load_sentence_model():
    try:
        model = SentenceTransformer("./paraphrase-MiniLM-L6-v2", device="cpu")  # local model
    except Exception:
        model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2", device="cpu")  # fallback
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

MODEL = load_sentence_model()

# ---------------- Job Recommendation System ----------------
class JobRecommendationSystem:
    def __init__(self, jobs_csv: str):
        print("✅ Loading dataset locally from", jobs_csv)
        self.jobs_df = pd.read_csv(jobs_csv)

        # Fill NaN with empty strings to avoid crashes
        self.jobs_df = self.jobs_df.fillna("")

        # Create combined job_text for semantic similarity
        text_cols = [
            "position",
            "job_role_and_duties",
            "requisite_skill",
            "benefits",
            "formatted_experience_level",
            "formatted_work_type",
            "work_type",
            "city",
            "state",
            "country",
        ]
        self.jobs_df["job_text"] = self.jobs_df[text_cols].astype(str).agg(" ".join, axis=1)

        # Deduplicate jobs to avoid repetition
        self.jobs_df = self.jobs_df.drop_duplicates(subset=["job_text"]).reset_index(drop=True)
        self.jobs_texts = self.jobs_df["job_text"].tolist()

        # ---------------- TF-IDF ----------------
        print("⚡ Precomputing TF-IDF vectors...")
        self.vectorizer = TfidfVectorizer(max_features=10000)
        self.job_tfidf_matrix = self.vectorizer.fit_transform(self.jobs_texts)

        # ---------------- Sentence-BERT embeddings ----------------
        if os.path.exists("job_embeddings.npy"):
            print("✅ Loaded precomputed embeddings from job_embeddings.npy")
            self.job_embeddings = np.load("job_embeddings.npy")
        else:
            print("⚠️ job_embeddings.npy not found. Generating embeddings now...")
            self.job_embeddings = MODEL.encode(
                self.jobs_texts,
                batch_size=64,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            np.save("job_embeddings.npy", self.job_embeddings)
            print("✅ Saved embeddings to job_embeddings.npy")

    # ---------------- TF-IDF Narrowing ----------------
    def filter_top_jobs(self, resume_text: str, top_k: int = 500):
        resume_vector = self.vectorizer.transform([resume_text])
        cosine_similarities = linear_kernel(resume_vector, self.job_tfidf_matrix).flatten()
        top_indices = cosine_similarities.argsort()[-top_k:][::-1]
        return (
            self.jobs_df.iloc[top_indices].reset_index(drop=True),
            self.job_embeddings[top_indices],
        )

    # ---------------- Final Recommendations ----------------
    def recommend_jobs(self, resume_text: str, top_n: int = 20):
        # Step 1: TF-IDF narrowing
        filtered_jobs_df, filtered_embeddings = self.filter_top_jobs(resume_text)

        # Step 2: SBERT embeddings for resume
        resume_embedding = MODEL.encode(resume_text, convert_to_numpy=True).reshape(1, -1)

        # Step 3: Cosine similarity
        similarities = cosine_similarity(resume_embedding, filtered_embeddings)[0]

        # Step 4: Rank top jobs
        top_indices = similarities.argsort()[-top_n:][::-1]

        recommendations = []
        for idx in top_indices:
            job = filtered_jobs_df.iloc[idx]
            recommendations.append({
                "job_id": job.get("job_id", ""),
                "position": job.get("position", "N/A"),
                "workplace": job.get("workplace", "N/A"),
                "formatted_work_type": job.get("formatted_work_type", "N/A"),
                "remote_allowed": job.get("remote_allowed", "N/A"),
                "salary_range": f"{job.get('min_salary','')} - {job.get('max_salary','')} {job.get('currency','')} ({job.get('pay_period','')})",
                "experience_level": job.get("formatted_experience_level", "N/A"),
                "job_role_and_duties": job.get("job_role_and_duties", "N/A"),
                "skills": job.get("requisite_skill", "N/A"),
                "benefits": job.get("benefits", "N/A"),
                "location": f"{job.get('city','')}, {job.get('state','')}, {job.get('country','')}",
                "company_size": job.get("company_size", "N/A"),
                "employee_count": job.get("employee_count", "N/A"),
                "company_website": job.get("company_website", "N/A"),
                "apply_link": job.get("apply_link", job.get("job_posting_url", "")),
                "similarity": float(similarities[idx]),
            })
        return recommendations
