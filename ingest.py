from __future__ import annotations

from pathlib import Path
from typing import List

from pypdf import PdfReader

from config import CHUNK_SIZE, CHUNK_OVERLAP
from rag import Chunk, RAGEngine


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_documents(data_dir: str = "data") -> List[tuple[str, str]]:
    docs = []
    for path in Path(data_dir).glob("*"):
        if path.suffix.lower() in {".txt", ".md"}:
            docs.append((path.name, read_text_file(path)))
        elif path.suffix.lower() == ".pdf":
            docs.append((path.name, read_pdf_file(path)))
    return docs


def chunk_text(text: str, source: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    chunks: List[Chunk] = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text_value = text[start:end].strip()
        if chunk_text_value:
            chunks.append(Chunk(text=chunk_text_value, source=source, chunk_id=chunk_id))
            chunk_id += 1
        if end == len(text):
            break
        start += chunk_size - overlap

    return chunks


def main() -> None:
    docs = load_documents()
    all_chunks: List[Chunk] = []

    for source, content in docs:
        all_chunks.extend(chunk_text(content, source))

    if not all_chunks:
        raise ValueError("No documents found in the data/ directory.")

    engine = RAGEngine()
    engine.build_index(all_chunks)
    print(f"Indexed {len(all_chunks)} chunks from {len(docs)} documents.")


if __name__ == "__main__":
    main()