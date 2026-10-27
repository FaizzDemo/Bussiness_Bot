from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(dotenv_path=Path(__file__).with_name(".env"))


def get_env_config():
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    embedding_model = os.getenv("OPENROUTER_EMBEDDING_MODEL", "").strip()

    if not api_key or api_key == "your_openrouter_api_key":
        raise ValueError("OPENROUTER_API_KEY is missing or still set to the placeholder value.")
    if not embedding_model:
        raise ValueError("OPENROUTER_EMBEDDING_MODEL is missing in the .env file.")

    return api_key, embedding_model


def get_openrouter_client(api_key: str | None = None):
    if api_key is None:
        api_key, _ = get_env_config()
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")


def validate_embedding_model(client: OpenAI, embedding_model: str):
    try:
        response = client.embeddings.create(model=embedding_model, input="validation")
        if not getattr(response, "data", None):
            raise ValueError("No embedding data returned.")
        return True
    except Exception as exc:
        raise ValueError(
            "The configured OPENROUTER_EMBEDDING_MODEL is not compatible with the embeddings endpoint. "
            "Use a compatible OpenRouter embedding model such as openai/text-embedding-3-small or openai/text-embedding-3-large. "
            "Do not use a chat-only model for embeddings."
        ) from exc


def load_knowledge_base(path: str | Path) -> str:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as file:
        return file.read()


def chunk_text(text: str, source: str = "knowledge", chunk_size: int = 800, overlap: int = 120) -> List[Dict[str, Any]]:
    if not text or not text.strip():
        return []

    normalized = re.sub(r"\r\n?", "\n", text)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    paragraphs = [part.strip() for part in normalized.split("\n\n") if part.strip()]

    chunks: List[Dict[str, Any]] = []
    current = ""
    for paragraph in paragraphs:
        if len(current) + len(paragraph) <= chunk_size:
            current = (current + "\n\n" + paragraph).strip()
        else:
            if current:
                chunks.append({"text": current, "source": source})
            if len(paragraph) > chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                sentence_buffer = ""
                for sentence in sentences:
                    if len(sentence_buffer) + len(sentence) <= chunk_size:
                        sentence_buffer = (sentence_buffer + " " + sentence).strip()
                    else:
                        if sentence_buffer:
                            chunks.append({"text": sentence_buffer, "source": source})
                        sentence_buffer = sentence.strip()
                if sentence_buffer:
                    current = sentence_buffer
            else:
                current = paragraph

    if current:
        chunks.append({"text": current, "source": source})

    if not chunks:
        return [{"text": normalized[:chunk_size], "source": source}]

    result = []
    for index, chunk in enumerate(chunks):
        text_value = chunk["text"]
        if index > 0 and len(result) > 0:
            previous_text = result[-1]["text"]
            overlap_text = " ".join(previous_text.split()[-overlap // 2:]) if overlap else ""
            text_value = (overlap_text + " " + text_value).strip()
        result.append({"text": text_value, "source": chunk["source"], "chunk_index": index})

    return result


def generate_embeddings_for_chunks(chunks: List[Dict[str, Any]], client: OpenAI, embedding_model: str):
    if not chunks:
        return np.empty((0, 0), dtype=float)

    embedding_vectors = []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        response = client.embeddings.create(model=embedding_model, input=text)
        embedding_vectors.append(np.asarray(response.data[0].embedding, dtype=float))

    if not embedding_vectors:
        return np.empty((0, 0), dtype=float)

    return np.vstack(embedding_vectors)
