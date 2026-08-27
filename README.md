# 📚 Ask My Docs — Production RAG Application

> **Domain-specific document Q&A** with hybrid retrieval (BM25 + vector search),
> cross-encoder reranking, full LaTeX math rendering, inline citations, per-document index management, and a CI-gated RAGAS evaluation pipeline.

[![CI](https://github.com/Vivek-6392/Ask-My-Docs/actions/workflows/ci.yml/badge.svg)](https://github.com/Vivek-6392/Ask-My-Docs/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🚀 Key Features

- ⚡ **Hybrid Retrieval Funnel**: Combines dense semantic search (ChromaDB + BAAI/bge-small) and sparse keyword search (BM25) fused with Reciprocal Rank Fusion (RRF).
- 🎯 **Cross-Encoder Reranking**: Uses `ms-marco-MiniLM-L-6-v2` to jointly score and rerank candidate chunks for maximum precision.
- 📐 **Full LaTeX MathJax Support**: Seamlessly renders complex mathematical formulas ($...$ inline and $$...$$ display equations) emitted by reasoning models.
- 📂 **Auto-Index & Document Management**: Direct drag-and-drop file upload with automated chunking/indexing and one-click per-document deletion (`🗑`) in the sidebar.
- 📌 **Inline Citations & Collapsible Sources**: Numbered citation tags linking to collapsible source snippets with match confidence scores.
- 📊 **CI-Gated RAGAS Evaluation**: Automated quality gate testing Faithfulness, Answer Relevancy, Context Precision, and Context Recall on every push.
- 🐳 **Containerized & Production Ready**: Multi-stage Docker build with automated build & push to GitHub Container Registry (`ghcr.io`).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                             │
│       Sidebar (Upload, Docs, Delete, Settings) · Chat View      │
└────────────────────────────┬────────────────────────────────────┘
                             │ query
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       RAG Pipeline                              │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐                          │
│  │ Vector Search │    │  BM25 Search │   (hybrid retrieval)     │
│  │  ChromaDB    │    │  rank-bm25   │                          │
│  └──────┬───────┘    └──────┬───────┘                          │
│         └────────┬──────────┘                                   │
│                  ▼                                              │
│      ┌─────────────────────┐                                    │
│      │ Reciprocal Rank     │  (score fusion, k=60)              │
│      │ Fusion (RRF)        │                                    │
│      └──────────┬──────────┘                                    │
│                 ▼                                               │
│      ┌─────────────────────┐                                    │
│      │  Cross-Encoder      │  (reranking)                       │
│      │  Reranker (MiniLM)  │                                    │
│      └──────────┬──────────┘                                    │
│                 ▼                                               │
│      ┌──────────────────────────────────┐                       │
│      │  Groq LLM (openai/gpt-oss-120b)  │  (answer generation   │
│      │  + Citation & LaTeX Prompts      │   + citations)        │
│      └──────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
                             │
          ┌──────────────────┴───────────────────┐
          ▼                                       ▼
┌──────────────────────────────────┐   ┌─────────────────────────┐
│        CI/CD Pipeline            │   │    RAGAS Evaluation     │
│  GitHub Actions:                 │   │  Judge: compound-mini   │
│  Lint → Unit Tests → RAGAS Gate  │   │  • Faithfulness         │
│  → GHCR Docker Image Push        │   │  • Answer Relevancy     │
│                                  │   │  • Context Precision    │
│                                  │   │  • Context Recall       │
└──────────────────────────────────┘   └─────────────────────────┘
```

---

## File Structure

```
ask-my-docs/
├── app/                        # Streamlit application
│   ├── __init__.py
│   ├── main.py                 # Streamlit entry point
│   ├── config.py               # Centralised AppConfig dataclass + env loading
│   ├── session.py              # Streamlit session state management
│   ├── ui.py                   # UI components (sidebar, chat, sources, citations)
│   └── styles.css              # Custom SaaS theme CSS
│
├── retrieval/                  # RAG core engine
│   ├── __init__.py
│   ├── pipeline.py             # RAGPipeline orchestrator
│   ├── vector_store.py         # ChromaDB dense retrieval & document deletion
│   ├── bm25_store.py           # BM25 sparse keyword retrieval & persistence
│   ├── reranker.py             # Cross-encoder reranking
│   ├── fusion.py               # Reciprocal Rank Fusion (RRF)
│   ├── prompt.py               # Prompts & citation context formatting
│   └── ingestor.py             # Document parser → chunker → embedder
│
├── evaluation/                 # RAGAS evaluation pipeline
│   ├── __init__.py
│   ├── evaluator.py            # CLI evaluator with threshold quality gates
│   ├── eval_dataset.json       # Ground truth evaluation Q&A dataset
│   └── results.json            # Generated evaluation metrics output
│
├── tests/                      # Pytest test suite
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_retrieval.py       # Unit tests (RRF, BM25, prompt builder)
│   ├── test_pipeline.py        # Integration tests (mocked LLM)
│   └── test_evaluator.py       # Evaluator threshold & logic tests
│
├── scripts/
│   ├── ingest.py               # CLI document ingestion tool
│   └── seed_eval_docs.py       # Seeds corpus for CI automated testing
│
├── .github/
│   └── workflows/
│       └── ci.yml              # Lint → Test → RAGAS Eval → GHCR Publish
│
├── .streamlit/
│   └── config.toml             # Streamlit server & theme configuration
│
├── Dockerfile                  # Multi-stage production container build
├── docker-compose.yml          # Local Docker setup with persistent volumes
├── requirements.txt            # Production Python dependencies
├── requirements-dev.txt        # Development & linting dependencies
└── pyproject.toml              # Ruff configuration
```

---

## Quick Start (Local)

### Prerequisites

- Python 3.11+
- A free Groq API key from [console.groq.com](https://console.groq.com)

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/Vivek-6392/Ask-My-Docs.git
cd Ask-My-Docs

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Set your `GROQ_API_KEY`:

```dotenv
GROQ_API_KEY=gsk_your_groq_api_key_here
LLM_MODEL=openai/gpt-oss-120b
RAGAS_LLM_MODEL=groq/compound-mini
```

### 3. Launch the Application

```bash
streamlit run app/main.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Upload & Query

1. Click **📂 Upload Documents** in the sidebar to drop PDF, DOCX, TXT, or Markdown files.
2. Files are **automatically indexed** immediately upon selection.
3. Start asking questions in the chat bar. Expand the **🗎 Sources** dropdown to view source snippets.

---

## Running Tests

```bash
# Run unit & integration test suite
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=retrieval --cov=app -v
```

---

## Running the Evaluation Pipeline

The evaluation pipeline uses **RAGAS** to evaluate the RAG system against ground truth questions and answers:

| Metric | Threshold | Purpose |
|---|---|---|
| **Faithfulness** | ≥ 0.85 | Verifies answers are grounded strictly in retrieved context |
| **Answer Relevancy** | ≥ 0.80 | Ensures the generated response answers the user's prompt |
| **Context Precision** | ≥ 0.75 | Validates high ranking of ground-truth relevant passages |
| **Context Recall** | ≥ 0.70 | Measures if all necessary context was retrieved |

```bash
# 1. Seed sample documents into index
python scripts/seed_eval_docs.py

# 2. Run evaluation
python -m evaluation.evaluator \
  --dataset evaluation/eval_dataset.json \
  --output evaluation/results.json
```

---

## Docker Deployment

### Using Docker Compose (Recommended)

```bash
# 1. Start application with persistent storage
docker compose up -d

# 2. View container logs
docker compose logs -f app
```

### Using Standalone Docker

```bash
# Build image
docker build -t ask-my-docs:latest .

# Run container
docker run -d \
  --name ask-my-docs \
  -p 8501:8501 \
  -e GROQ_API_KEY=gsk_your_key_here \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v $(pwd)/uploaded_docs:/app/uploaded_docs \
  ask-my-docs:latest
```

---

## CI/CD Pipeline

The GitHub Actions workflow ([.github/workflows/ci.yml](file:///.github/workflows/ci.yml)) runs on every push and PR:

1. **Lint**: Code style and formatting checks with `ruff`.
2. **Unit Tests**: Full test suite execution with `pytest`.
3. **RAG Quality Gate**: Automatically runs `seed_eval_docs.py` and `evaluator.py`, verifying RAGAS metric thresholds and uploading `results.json` artifacts.
4. **Build & Publish**: Builds multi-platform Docker container and publishes to GitHub Container Registry (`ghcr.io`) on `main` branch merges.

---

## Configuration Reference

All settings can be configured via `.env` or system environment variables:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | `""` | **Required.** Groq API authentication key |
| `LLM_MODEL` | `openai/gpt-oss-120b` | LLM used for RAG generation & chat responses |
| `RAGAS_LLM_MODEL` | `groq/compound-mini` | LLM used for RAGAS evaluation in CI (70K TPM) |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model (downloaded locally) |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker model |
| `VECTOR_TOP_K` | `40` | Dense vector candidates from ChromaDB |
| `BM25_TOP_K` | `25` | Sparse keyword candidates from BM25 |
| `RERANK_TOP_K` | `10` | Final top chunks passed into LLM prompt context |
| `CHUNK_SIZE` | `512` | Token chunk size for document splitting |
| `CHUNK_OVERLAP` | `64` | Overlap size between adjacent chunks |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Storage path for ChromaDB vector embeddings |
| `UPLOAD_DIR` | `./uploaded_docs` | Storage directory for user uploaded documents |

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more details.
