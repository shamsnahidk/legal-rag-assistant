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