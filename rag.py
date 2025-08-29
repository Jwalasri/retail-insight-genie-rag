"""Retrieval-Augmented Generation (RAG) pipeline implementation.

This module defines a simple RAG class that uses TF‑IDF to retrieve
relevant product documents and composes a basic answer from the top
document.  It also exposes a utility function to load documents from
JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def load_docs(path: Path | str) -> List[Dict[str, Any]]:
    """Load a list of documents from a JSON file.

    Each document must be a mapping with at least `title` and
    `description` fields.  Additional fields are ignored by the RAG
    pipeline.

    Parameters
    ----------
    path:
        Path to a JSON file containing an array of objects.

    Returns
    -------
    list
        A list of document dictionaries.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("documents JSON must contain a list")
    return data


class RAG:
    """A minimal retrieval‑augmented generation pipeline.

    The class loads documents, builds a TF‑IDF vectorizer for the
    textual fields and provides methods to retrieve top documents
    for a given query and to compose a simple answer from the best
    matching document.
    """

    def __init__(self, documents: Iterable[Dict[str, Any]]):
        # Keep a list of documents internally.  We copy the objects to
        # avoid accidental modifications from outside of the class.
        self.docs: List[Dict[str, Any]] = [dict(doc) for doc in documents]
        # Concatenate title and description for vectorisation.
        self.texts: List[str] = [
            f"{doc.get('title', '')} {doc.get('description', '')}"
            for doc in self.docs
        ]
        # Build TF‑IDF model.  We ignore English stop words to
        # emphasise the product terms.
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_vectors = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[int, float]]:
        """Retrieve the top‐k documents for a given query.

        Parameters
        ----------
        query:
            The natural language query describing the information need.
        k:
            The number of documents to return.

        Returns
        -------
        list of tuple(int, float)
            A list of `(index, score)` pairs for the top documents.
        """
        if not query:
            return []
        # Transform the query into the TF‑IDF space.
        query_vec = self.vectorizer.transform([query])
        # Compute cosine similarity between the query and all documents.
        scores = linear_kernel(query_vec, self.doc_vectors).flatten()
        # Get the indices of the top scores.
        ranked_indices = scores.argsort()[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in ranked_indices if scores[idx] > 0]

    def answer(self, query: str, k: int = 1) -> str:
        """Compose a simple answer by returning the title and description
        of the top document.

        If no documents are retrieved, returns a polite fallback.
        """
        results = self.retrieve(query, k)
        if not results:
            return "I'm sorry, I couldn't find information on that topic."
        idx, _score = results[0]
        doc = self.docs[idx]
        title = doc.get("title", "")
        description = doc.get("description", "")
        return f"{title}: {description}"