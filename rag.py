from __future__ import annotations

import os
from typing import List, Dict, Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from embeddings import get_openrouter_client


def embed_text(question: str, embedding_model: str | None = None) -> np.ndarray:
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if embedding_model is None:
        embedding_model = os.getenv("OPENROUTER_EMBEDDING_MODEL", "").strip()
    if not embedding_model:
        raise ValueError("OPENROUTER_EMBEDDING_MODEL is missing from the environment.")

    client = get_openrouter_client()
    response = client.embeddings.create(model=embedding_model, input=question)
    return np.asarray(response.data[0].embedding, dtype=float)


def retrieve_relevant_chunks(question: str, chunks: List[Dict[str, Any]], embeddings: np.ndarray, threshold: float = 0.12, embedding_model: str | None = None):
    if not chunks or embeddings.size == 0:
        return []

    query_vector = embed_text(question, embedding_model)
    similarities = cosine_similarity(query_vector.reshape(1, -1), embeddings)[0]

    ranked = []
    for idx, chunk in enumerate(chunks):
        score = float(similarities[idx])
        ranked.append({
            "score": score,
            "text": chunk.get("text", ""),
            "source": chunk.get("source", "unknown"),
            "index": idx,
        })

    ranked.sort(key=lambda item: item["score"], reverse=True)
    relevant = [item for item in ranked if item["score"] >= threshold]
    return relevant[:3] if relevant else []
