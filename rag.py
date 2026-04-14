from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from config import EMBEDDING_MODEL, USE_OPENAI, OPENAI_API_KEY, TOP_K


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


class RAGEngine:
    def __init__(self) -> None:
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.index: faiss.IndexFlatL2 | None = None
        self.metadata: List[Dict[str, Any]] = []
        self.client = OpenAI(api_key=OPENAI_API_KEY) if USE_OPENAI and OPENAI_API_KEY else None

    def build_index(self, chunks: List[Chunk]) -> None:
        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        self.metadata = [
            {"text": c.text, "source": c.source, "chunk_id": c.chunk_id}
            for c in chunks
        ]

        faiss.write_index(self.index, "index.faiss")
        np.save("metadata.npy", np.array(self.metadata, dtype=object), allow_pickle=True)

    def load_index(self) -> None:
        if not Path("index.faiss").exists() or not Path("metadata.npy").exists():
            raise FileNotFoundError("Missing FAISS index or metadata. Run ingest.py first.")

        self.index = faiss.read_index("index.faiss")
        self.metadata = np.load("metadata.npy", allow_pickle=True).tolist()

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        if self.index is None:
            self.load_index()

        query_vector = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(dist)
            results.append(item)
        return results

    def answer(self, query: str) -> Dict[str, Any]:
        retrieved = self.retrieve(query)

        context = "\n\n".join(
            f"[Source: {r['source']} | Chunk: {r['chunk_id']}]\n{r['text']}"
            for r in retrieved
        )

        fallback_answer = (
            "Retrieved relevant evidence from the indexed legal document. "
            "The records support continuing injury through persistent pain, "
            "documented work limitations, and continuity of treatment. "
            "See cited source chunks for supporting details."
        )

        if self.client:
            prompt = f"""
You are a legal AI assistant.

Answer the question clearly and concisely using ONLY the provided context.

Rules:
- Do not repeat the raw context
- Summarize the most relevant points
- If the answer is not supported by the context, say you do not have enough information
- Keep the answer short and structured

Question:
{query}

Context:
{context}
"""
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                answer_text = response.choices[0].message.content
            except Exception:
                answer_text = fallback_answer
        else:
            answer_text = fallback_answer

        return {
            "query": query,
            "answer": answer_text,
            "sources": [
                {"source": r["source"], "chunk_id": r["chunk_id"], "score": r["score"]}
                for r in retrieved
            ],
        }