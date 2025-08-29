"""Unit tests for the Retail Insight Genie RAG pipeline."""

import json
from pathlib import Path

from app.rag import RAG, load_docs


def get_docs_path() -> Path:
    """Return the path to the sample docs.json bundled with the app."""
    return Path(__file__).resolve().parent.parent / "app" / "data" / "docs.json"


def test_load_docs() -> None:
    docs = load_docs(get_docs_path())
    assert isinstance(docs, list)
    assert len(docs) > 0
    for doc in docs:
        assert "title" in doc and "description" in doc


def test_retrieve_non_empty() -> None:
    docs = load_docs(get_docs_path())
    rag = RAG(docs)
    results = rag.retrieve("battery life", k=2)
    # Should return at least one document
    assert len(results) >= 1
    # Each result is a tuple of (index, score)
    idx, score = results[0]
    assert isinstance(idx, int)
    assert isinstance(score, float)


def test_answer_returns_string() -> None:
    docs = load_docs(get_docs_path())
    rag = RAG(docs)
    answer = rag.answer("gaming console graphics")
    assert isinstance(answer, str)
    assert len(answer) > 0