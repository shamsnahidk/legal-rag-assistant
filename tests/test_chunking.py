import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingest import chunk_text


def test_chunk_text_creates_multiple_chunks():
    text = "A" * 1500
    chunks = chunk_text(text=text, source="test.txt", chunk_size=500, overlap=100)
    assert len(chunks) >= 3
    assert chunks[0].source == "test.txt"
    assert chunks[0].chunk_id == 0