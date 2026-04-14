from fastapi import FastAPI
from pydantic import BaseModel

from rag import RAGEngine

app = FastAPI(title="Legal RAG Assistant", version="1.0.0")
engine = RAGEngine()


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask")
def ask_question(payload: QueryRequest) -> dict:
    return engine.answer(payload.query)