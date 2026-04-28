"""
rag_engine.py — FAISS-based RAG engine.
Converts dataset rows → text chunks → OpenAI embeddings → FAISS index.
Supports semantic search to retrieve context for the LLM.
"""
from __future__ import annotations

import hashlib
import json
import logging
import textwrap
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("university_agent.rag")

# Optional imports with graceful fallback
try:
    import faiss
    FAISS_OK = True
except ImportError:
    FAISS_OK = False
    logger.warning("faiss-cpu not installed — RAG will use simple TF-IDF fallback")

# Groq does not support embeddings — using TF-IDF fallback only
OPENAI_OK = False


# ── Chunk helpers ─────────────────────────────────────────────────────────────
def _row_to_text(row: pd.Series, name_col, dept_col, attend_col, subject_cols) -> str:
    parts = []
    if name_col and name_col in row.index:
        parts.append(f"Student: {row[name_col]}")
    if dept_col and dept_col in row.index:
        parts.append(f"Department: {row[dept_col]}")
    if attend_col and attend_col in row.index:
        parts.append(f"Attendance: {row[attend_col]}%")
    for sc in subject_cols:
        if sc in row.index:
            parts.append(f"{sc}: {row[sc]}")
    if "Average" in row.index:
        parts.append(f"Average: {row['Average']}")
    if "Grade" in row.index:
        parts.append(f"Grade: {row['Grade']}")
    return " | ".join(parts)


def build_chunks(meta: dict, max_chunks: int = 2000) -> list[str]:
    df = meta["df"]
    sample = df.sample(min(len(df), max_chunks), random_state=42) if len(df) > max_chunks else df

    chunks = []
    for _, row in sample.iterrows():
        text = _row_to_text(
            row,
            meta.get("name_col"),
            meta.get("dept_col"),
            meta.get("attend_col"),
            meta.get("subject_cols", []),
        )
        if text.strip():
            chunks.append(text)

    # Also add department-level summaries
    if meta.get("dept_col"):
        for dept, grp in df.groupby(meta["dept_col"]):
            summary_parts = [f"Department summary: {dept}", f"Count: {len(grp)}"]
            for sc in meta.get("subject_cols", []):
                summary_parts.append(f"Avg {sc}: {grp[sc].mean():.1f}")
            if meta.get("attend_col"):
                summary_parts.append(f"Avg Attendance: {grp[meta['attend_col']].mean():.1f}%")
            chunks.append(" | ".join(summary_parts))

    logger.info("Built %d RAG chunks", len(chunks))
    return chunks


# ── FAISS index ───────────────────────────────────────────────────────────────
class RAGEngine:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self.chunks: list[str] = []
        self.index = None           # faiss index
        self._tfidf = None          # fallback
        self._tfidf_matrix = None
        self._embed_dim: int = 1536
        self.client = None  # Groq does not support embeddings — TF-IDF used

    # ── Build ─────────────────────────────────────────────────────────────────
    def build(self, chunks: list[str]) -> None:
        self.chunks = chunks
        if self.client and FAISS_OK:
            self._build_faiss(chunks)
        else:
            self._build_tfidf(chunks)

    def _build_faiss(self, chunks: list[str]) -> None:
        logger.info("Building FAISS index with %d chunks …", len(chunks))
        vectors = self._embed_batch(chunks)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors.astype("float32"))
        logger.info("FAISS index built  dim=%d  n=%d", dim, self.index.ntotal)

    def _build_tfidf(self, chunks: list[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        logger.info("Building TF-IDF fallback index …")
        self._tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
        self._tfidf_matrix = self._tfidf.fit_transform(chunks)
        logger.info("TF-IDF index built")

    # ── Query ─────────────────────────────────────────────────────────────────
    def retrieve(self, query: str, top_k: int = 6) -> list[str]:
        if not self.chunks:
            return []
        if self.index is not None and self.client:
            return self._retrieve_faiss(query, top_k)
        if self._tfidf is not None:
            return self._retrieve_tfidf(query, top_k)
        # last resort: keyword match
        q = query.lower()
        scored = [(c, sum(w in c.lower() for w in q.split())) for c in self.chunks]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_k]]

    def _retrieve_faiss(self, query: str, top_k: int) -> list[str]:
        vec = self._embed_batch([query]).astype("float32")
        _, indices = self.index.search(vec, min(top_k, len(self.chunks)))
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]

    def _retrieve_tfidf(self, query: str, top_k: int) -> list[str]:
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = self._tfidf.transform([query])
        scores = cosine_similarity(q_vec, self._tfidf_matrix).flatten()
        top_idx = scores.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in top_idx]

    # ── Embedding ─────────────────────────────────────────────────────────────
    def _embed_batch(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            resp = self.client.embeddings.create(
                input=batch,
                model="text-embedding-ada-002",
            )
            vecs = [item.embedding for item in resp.data]
            all_vecs.extend(vecs)
        return np.array(all_vecs, dtype="float32")

    # ── Convenience ───────────────────────────────────────────────────────────
    def format_context(self, query: str, top_k: int = 6) -> str:
        docs = self.retrieve(query, top_k)
        if not docs:
            return "No relevant data found."
        return "\n".join(f"- {d}" for d in docs)

    @property
    def is_ready(self) -> bool:
        return bool(self.chunks)
