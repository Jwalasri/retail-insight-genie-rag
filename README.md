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
```

## Running the API

Start the FastAPI server with:

```bash
uvicorn app.main:app --reload
```

Then query the API:

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"query": "What is the battery life of the Pro laptop?"}'
```

You can also interact via the API docs at `http://localhost:8000/docs`.

## Evaluation

To run the evaluation harness on the sample query set:

```bash
python app/evaluate.py --k 3
```

This will print retrieval precision metrics.

## Project Structure

```
retail-insight-genie-rag/
├── app/                 # FastAPI application and RAG pipeline
│   ├── main.py
│   ├── rag.py
│   ├── data/
│   ├── evaluate.py
│   └── …
├── tests/               # Unit tests (pytest)
├── requirements.txt
├── Dockerfile
├── .github/workflows/python-ci.yml
├── .gitignore
├── LICENSE
└── README.md
```

## Contributing

Contributions are welcome! Feel free to open issues or pull requests. See `CONTRIBUTING.md` for guidelines.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.