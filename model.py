import string
import numpy as np
import faiss
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------- Load Model -----------------
MODEL =SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2", device="cpu")
MODEL = torch.quantization.quantize_dynamic(MODEL, {torch.nn.Linear}, dtype=torch.qint8)


class JobRecommendationSystem:
    def __init__(self, jobs_csv):
        """Initialize the system and load enriched job data from CSV file."""
        self.jobs_df = pd.read_csv(jobs_csv)

        # Ensure apply_link exists
        if "apply_link" not in self.jobs_df.columns:
            self.jobs_df["apply_link"] = None

        # --- Safe column concat ---
        def safe_col(col):
            return self.jobs_df[col].astype(str) + " " if col in self.jobs_df.columns else ""

        # Build job_text
        self.jobs_df["job_text"] = (
            safe_col("workplace") +
            safe_col("position") +
            safe_col("job_role_and_duties") +
            safe_col("requisite_skill") +
            safe_col("benefits") +
            safe_col("industry_id") +
            safe_col("formatted_work_type") +
            safe_col("work_type") +
            safe_col("formatted_experience_level") +
            safe_col("country") +
            safe_col("state") +
            safe_col("city")
        )

        self.jobs_texts = self.jobs_df["job_text"].tolist()
        self.job_info = self.jobs_df.copy()

        # --- Load or compute embeddings ---
        try:
            self.job_embeddings = np.load("job_embeddings.npy").astype(np.float16)
            print("✅ Loaded precomputed embeddings from job_embeddings.npy")
        except FileNotFoundError:
            print("⚠️ job_embeddings.npy not found. Generating embeddings now...")
            self.job_embeddings = MODEL.encode(
                self.jobs_texts,
                convert_to_numpy=True,
                batch_size=32,
                show_progress_bar=True
            ).astype(np.float16)
            np.save("job_embeddings.npy", self.job_embeddings)
            print("✅ Saved embeddings to job_embeddings.npy")

        # --- Build FAISS index (global, on all jobs) ---
        self.dim = self.job_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.job_embeddings.astype(np.float16))

        # --- Precompute TF-IDF once ---
        self.vectorizer = TfidfVectorizer()
        self.job_tfidf = self.vectorizer.fit_transform(self.jobs_texts)

    # ----------------- Helpers -----------------
    def clean_text(self, text):
        """Lowercase, strip punctuation, clean text."""
        return text.lower().translate(str.maketrans("", "", string.punctuation)).strip()

    def filter_top_jobs(self, resume_text, top_n=500):
        """Use TF-IDF to preselect most relevant jobs (fast)."""
        resume_vector = self.vectorizer.transform([resume_text])
        similarity_scores = (self.job_tfidf @ resume_vector.T).toarray().flatten()
        top_indices = np.argsort(similarity_scores)[-top_n:]
        return (
            [self.jobs_texts[i] for i in top_indices],
            self.job_info.iloc[top_indices].reset_index(drop=True),
            self.job_embeddings[top_indices],
        )

    def recommend_jobs(self, resume_text, top_n=20):
        """Recommend jobs using FAISS similarity search + deduplication."""
        resume_text = self.clean_text(resume_text)
        filtered_jobs_texts, filtered_jobs_df, filtered_embeddings = (
            self.filter_top_jobs(resume_text)
        )

        # Encode resume
        resume_embedding = MODEL.encode([resume_text], convert_to_numpy=True).astype(np.float16)

        # Build temporary FAISS index on filtered jobs
        index = faiss.IndexFlatIP(self.dim)
        index.add(filtered_embeddings.astype(np.float16))

        # Search more than top_n to handle duplicates
        distances, indices = index.search(resume_embedding.astype(np.float16), top_n * 2)
        results = filtered_jobs_df.iloc[indices[0]]

        # Deduplicate by job_id and return top_n
        results = results.drop_duplicates(subset=["job_id"]).head(top_n)
        recommended_jobs = results.to_dict(orient="records")

        return {"recommended_jobs": recommended_jobs}
