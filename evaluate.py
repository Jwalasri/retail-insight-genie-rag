"""Simple evaluation harness for the RAG pipeline.

This script loads a small set of sample queries along with the
expected document identifiers and reports precision@k statistics for
the retrieval component.  Because the dataset is tiny, the metrics
are meant only as a demonstration rather than a rigorous benchmark.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from .rag import RAG, load_docs


# Define a handful of queries and the expected document indices.
SAMPLE_QUERIES: List[Tuple[str, int]] = [
    ("battery life of pro laptop", 0),
    ("features of smartphone", 1),
    ("noise cancellation earbuds", 2),
    ("tablet display size", 3),
    ("gaming console graphics", 4),
]


def precision_at_k(ranking: List[int], relevant_index: int, k: int) -> float:
    """Compute precision@k for a single query.

    Parameters
    ----------
    ranking:
        The ranked list of document indices.
    relevant_index:
        The index of the relevant document for this query.
    k:
        The cut‑off for precision.

    Returns
    -------
    float
        1.0 if the relevant document appears in the top‑k results, 0.0 otherwise.
    """
    return 1.0 if relevant_index in ranking[:k] else 0.0


def evaluate(doc_path: Path, k: int = 3) -> float:
    """Evaluate the retrieval precision@k on a small sample of queries.

    Parameters
    ----------
    doc_path:
        Path to the JSON file containing the product documents.
    k:
        The number of top documents to consider for each query.

    Returns
    -------
    float
        The mean precision@k across all sample queries.
    """
    docs = load_docs(doc_path)
    rag = RAG(docs)
    precisions: List[float] = []
    for query, relevant_idx in SAMPLE_QUERIES:
        results = rag.retrieve(query, k)
        ranking = [idx for idx, _score in results]
        precisions.append(precision_at_k(ranking, relevant_idx, k))
    return sum(precisions) / len(precisions) if precisions else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG retrieval component on a few sample queries.")
    parser.add_argument(
        "--docs",
        type=str,
        default=str(Path(__file__).parent / "data" / "docs.json"),
        help="Path to the JSON file with product documents.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of top documents to consider for precision@k.",
    )
    args = parser.parse_args()
    precision = evaluate(Path(args.docs), k=args.k)
    print(f"Precision@{args.k}: {precision:.2f}")


if __name__ == "__main__":
    main()