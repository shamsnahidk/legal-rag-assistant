# Legal RAG Assistant

Production-style RAG system with grounded retrieval and source attribution for legal document workflows.

## What it does
- Ingests `.txt`, `.md`, and `.pdf` files
- Chunks documents for retrieval
- Generates embeddings with SentenceTransformers
- Stores vectors in FAISS
- Retrieves relevant chunks for a user query
- Optionally uses OpenAI to generate grounded answers

## Tech Stack
- Python
- FastAPI
- SentenceTransformers
- FAISS
- OpenAI API
- Pytest

## Project Structure
```text
legal-rag-assistant/
├── app.py
├── ingest.py
├── rag.py
├── config.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── data/
│   └── sample_legal_memo.txt
└── tests/
    └── test_chunking.py
```

## Run Locally
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python ingest.py
pytest
uvicorn app:app --reload

## Example Query
{
  "query": "What evidence supports continuing injury?"
}

## Example Output
The system returns a concise answer grounded in retrieved document chunks, along with cited source references and retrieval scores.

## Why This Project Matters
This project demonstrates end-to-end RAG system design including document ingestion, chunking, embedding generation, vector retrieval, API serving, and grounded answer generation for legal document workflows.

## Design Decisions

- **Embedding Model**: all-MiniLM-L6-v2 chosen for speed vs accuracy tradeoff
- **Chunking Strategy**: 600 tokens with 100 overlap to preserve context continuity
- **Vector Store**: FAISS for fast local similarity search
- **Top-K Retrieval**: 4 documents to balance recall vs noise
- **Fallback Mode**: Supports retrieval-only when LLM is disabled

## Limitations

- No reranking (retrieval quality depends on embeddings)
- No hybrid search (keyword + semantic)
- No metadata filtering
- Not optimized for large-scale datasets

## Future Improvements

- Add reranking (cross-encoder)
- Hybrid search (BM25 + vector)
- Graph-based retrieval (GraphRAG)
- Query expansion techniques (HyDE)
- Evaluation metrics (precision/recall)