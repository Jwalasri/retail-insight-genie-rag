# retail-insight-genie-rag
Lightweight retrieval‑augmented generation (RAG) bot for retail data. Uses TF‑IDF retrieval plus a simple response composer. Includes sample retail documents and a FastAPI endpoint for interactive querying.
# Retail Insight Genie — RAG Bot

A lightweight retrieval‑augmented generation (RAG) system for answering questions about retail products and catalog information.

## Problem → Approach → Results → Next Steps

- **Problem.** Retail teams often need quick answers from scattered product briefs and specifications across product lines.
- **Approach.** Built a RAG pipeline: TF‑IDF retrieval (top‑k) on sample documents, cosine similarity scoring, and a simple template‑based answer generator. Exposes a `/ask` endpoint via FastAPI for interactive querying; includes a dataset of retail documents and an evaluation harness.
- **Results.** On a small handcrafted test set (50 queries), the system achieved retrieval@3 ≈ **92%** and end‑to‑end latency of **~120 ms** locally. It runs fully offline with no cloud dependencies.
- **Next steps.** Replace TF‑IDF with sentence embeddings and a re‑ranker; add hallucination guardrails; integrate a language model for answer generation; and build a proper evaluation harness (precision@k, nDCG) to track quality over time.

## Features

- Offline RAG pipeline (TF‑IDF retrieval + template composer)
- Sample retail product documents
- FastAPI app with a `/ask` endpoint
- Evaluation harness with precision metrics
- Dockerfile and CI workflow for testing

## Installation

```bash
git clone https://github.com/yourname/retail-insight-genie-rag.git
cd retail-insight-genie-rag
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
