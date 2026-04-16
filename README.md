# Legal RAG Assistant

A production-style Retrieval-Augmented Generation (RAG) system for legal document analysis using FAISS, SentenceTransformers, and FastAPI.

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
