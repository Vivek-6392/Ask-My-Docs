# 📚 Ask My Docs — Production RAG Application

> **Domain-specific document Q&A** with hybrid retrieval (BM25 + vector search),
> cross-encoder reranking, citation enforcement, and a CI-gated evaluation pipeline.

[![CI](https://github.com/vivek-6392/ask-my-docs/actions/workflows/ci.yml/badge.svg)](https://github.com/vivek-6392/ask-my-docs/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                             │
│          Chat Panel · Upload Panel · Retrieval Debug            │
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
│      │ Reciprocal Rank     │  (score fusion)                    │
│      │ Fusion (RRF k=60)   │                                    │
│      └──────────┬──────────┘                                    │
│                 ▼                                               │
│      ┌─────────────────────┐                                    │
│      │  Cross-Encoder      │  (reranking)                       │
│      │  Reranker (MiniLM)  │                                    │
│      └──────────┬──────────┘                                    │
│                 ▼                                               │
│      ┌─────────────────────┐                                    │
│      │  GPT-4o-mini        │  (generation + citations)         │
│      │  + Citation Prompt  │                                    │
│      └─────────────────────┘                                    │
└─────────────────────────────────────────────────────────────────┘
                             │
          ┌──────────────────┴───────────────────┐
          ▼                                       ▼
┌──────────────────┐                  ┌───────────────────────┐
│  CI Pipeline     │                  │  RAGAS Evaluation     │
│  GitHub Actions  │                  │  Faithfulness         │
│  Lint → Test →   │                  │  Answer Relevancy     │
│  Eval → Deploy   │                  │  Context Precision    │
└──────────────────┘                  │  Context Recall       │
                                      └───────────────────────┘
```

---

## File Structure

```
ask-my-docs/
├── app/                        # Streamlit application
│   ├── __init__.py
│   ├── main.py                 # Entry point
│   ├── config.py               # AppConfig dataclass + env loading
│   ├── session.py              # Streamlit session state helpers
│   ├── ui.py                   # UI components (sidebar, chat, upload)
│   └── styles.css              # Custom dark theme CSS
│
├── retrieval/                  # RAG core
│   ├── __init__.py
│   ├── pipeline.py             # RAGPipeline: orchestrator
│   ├── vector_store.py         # ChromaDB dense retrieval
│   ├── bm25_store.py           # BM25 sparse retrieval (rank-bm25)
│   ├── reranker.py             # Cross-encoder reranking
│   ├── fusion.py               # Reciprocal Rank Fusion
│   ├── prompt.py               # Prompt templates + citation builder
│   └── ingestor.py             # Load → chunk → embed → index
│
├── evaluation/                 # RAGAS evaluation pipeline
│   ├── __init__.py
│   ├── evaluator.py            # CLI evaluator with threshold gates
│   └── eval_dataset.json       # Q&A pairs for evaluation
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_retrieval.py       # Unit tests (RRF, BM25, prompts)
│   └── test_pipeline.py        # Integration tests (mocked LLM)
│
├── scripts/
│   ├── ingest.py               # CLI batch document ingestor
│   └── seed_eval_docs.py       # Seeds index for CI evaluation
│
├── .github/
│   └── workflows/
│       └── ci.yml              # Lint → Test → Eval → Docker CI
│
├── .streamlit/
│   └── config.toml             # Streamlit server + theme config
│
├── .env.example                # Environment variable template
├── .gitignore
├── Dockerfile                  # Multi-stage production build
├── docker-compose.yml          # Local dev with volumes
├── requirements.txt            # Pinned Python dependencies
├── pytest.ini                  # Pytest configuration
└── pyproject.toml              # Ruff + mypy configuration
```

---

## Quick Start (Local)

### Prerequisites

- Python 3.11+
- An OpenAI API key

### Step 1 — Clone & set up environment

```bash
git clone https://github.com/your-org/ask-my-docs.git
cd ask-my-docs

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2 — Configure environment

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_your_key_here
# Get a free key at: https://console.groq.com
```

### Step 3 — Run the app

```bash
streamlit run app/main.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Step 4 — Upload documents

Use the **Upload Documents** tab to drop in PDFs, TXT, DOCX, or Markdown files.
Click **Index Documents**. Then switch to the **Chat** tab to query them.

### Step 5 — (Optional) Batch ingest from CLI

```bash
python scripts/ingest.py --dir ./my_documents
python scripts/ingest.py --file report.pdf
python scripts/ingest.py --dir ./docs --clear   # clears existing index first
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only (no API key needed)
pytest tests/ -v -m "not integration"

# With coverage report
pytest tests/ --cov=retrieval --cov=app --cov-report=term-missing
```

---

## Running the Evaluation Pipeline

The RAGAS evaluation measures 4 metrics against your indexed documents:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Faithfulness | ≥ 0.80 | Answer grounded in retrieved context |
| Answer Relevancy | ≥ 0.75 | Answer addresses the question |
| Context Precision | ≥ 0.70 | Retrieved chunks are relevant |
| Context Recall | ≥ 0.70 | All needed info was retrieved |

```bash
# First seed your index with test documents
python scripts/seed_eval_docs.py

# Then run evaluation (requires OPENAI_API_KEY)
python -m evaluation.evaluator \
  --dataset evaluation/eval_dataset.json \
  --output evaluation/results.json
```

Exit code is `0` if all thresholds pass, `1` if any fail — used by CI to gate merges.

### Adding your own eval questions

Edit `evaluation/eval_dataset.json`:

```json
[
  {
    "question": "What is the cancellation policy?",
    "ground_truth": "Subscriptions can be cancelled any time; access continues until period end."
  }
]
```

---

## Deploy on Streamlit Community Cloud (Free)

1. Push your code to a **public** GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch `main`, and set **Main file path** to `app/main.py`.
4. Under **Advanced settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Click **Deploy**. Done — your app is live at `https://your-app.streamlit.app`.

> **Note:** ChromaDB persists in-memory on Streamlit Cloud (no disk persistence).
> For production persistence, use the Docker deployment below.

---

## Deploy with Docker (Self-hosted)

### Option A — Docker Compose (recommended)

```bash
# 1. Clone repo and set env
git clone https://github.com/your-org/ask-my-docs.git
cd ask-my-docs
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# 2. Build and start
docker-compose up -d

# 3. View logs
docker-compose logs -f app

# 4. Open in browser
open http://localhost:8501
```

Data is persisted in Docker volumes `chroma_data` and `uploaded_docs`.

### Option B — Docker standalone

```bash
docker build -t ask-my-docs .

docker run -d \
  --name ask-my-docs \
  -p 8501:8501 \
  -e GROQ_API_KEY=gsk_your_key_here \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v $(pwd)/uploaded_docs:/app/uploaded_docs \
  ask-my-docs
```

---

## Deploy on AWS / GCP / Azure

### AWS ECS (Fargate)

```bash
# 1. Push image to ECR
aws ecr create-repository --repository-name ask-my-docs
aws ecr get-login-password | docker login --username AWS --password-stdin <ECR_URI>
docker tag ask-my-docs:latest <ECR_URI>/ask-my-docs:latest
docker push <ECR_URI>/ask-my-docs:latest

# 2. Create ECS task definition pointing to the image
# 3. Set OPENAI_API_KEY as an ECS secret (AWS Secrets Manager)
# 4. Create a Fargate service with ALB on port 8501
```

### Google Cloud Run

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT/ask-my-docs
gcloud run deploy ask-my-docs \
  --image gcr.io/YOUR_PROJECT/ask-my-docs \
  --port 8501 \
  --set-env-vars OPENAI_API_KEY=sk-... \
  --allow-unauthenticated
```

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push/PR:

```
Push → Lint (ruff) → Unit Tests (pytest + coverage) → Docker Build
                                    │
                           main branch only
                                    ▼
                        RAGAS Evaluation Gate
                    (fails if metrics < thresholds)
                                    │
                              PR Comment with
                              score table
```

### Setting CI secrets

In your GitHub repo → **Settings → Secrets and variables → Actions**:

| Secret | Value |
|--------|-------|
| `GROQ_API_KEY` | Your Groq API key (free at console.groq.com) |

---

## Configuration Reference

All settings live in `.env` (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | **Required.** Get free key at [console.groq.com](https://console.groq.com) |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace model (downloaded locally, no API key) |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace cross-encoder |
| `CHUNK_SIZE` | `512` | Token chunk size |
| `CHUNK_OVERLAP` | `64` | Chunk overlap tokens |
| `VECTOR_TOP_K` | `20` | Dense retrieval candidates |
| `BM25_TOP_K` | `20` | Sparse retrieval candidates |
| `RERANK_TOP_K` | `5` | Final chunks after reranking |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |

### Available Groq models

| Model | Speed | Context | Best for |
|-------|-------|---------|----------|
| `llama-3.3-70b-versatile` | Fast | 128k | Best quality (default) |
| `llama-3.1-8b-instant` | Ultra-fast | 128k | Low-latency use cases |
| `mixtral-8x7b-32768` | Fast | 32k | Long-document tasks |
| `gemma2-9b-it` | Fast | 8k | Lightweight deployment |

---

## How the Hybrid Retrieval Works

```
Query: "What is the refund policy for digital products?"

1. VECTOR SEARCH (top-20)          2. BM25 SEARCH (top-20)
   Dense semantic similarity           Exact keyword matching
   ┌────────────────────────┐         ┌────────────────────────┐
   │ refund_policy.pdf p3   │ 0.91    │ refund_policy.pdf p3   │ 8.2
   │ terms_of_service.pdf   │ 0.87    │ faq.pdf p12            │ 6.1
   │ faq.pdf p12            │ 0.82    │ pricing.pdf p5         │ 3.4
   │ ...                    │         │ ...                    │
   └────────────────────────┘         └────────────────────────┘
                    │                           │
                    └──────────┬────────────────┘
                               ▼
                   3. RECIPROCAL RANK FUSION
                      score = Σ 1/(60 + rank)
                      Deduplicates & merges lists
                               │
                               ▼
                   4. CROSS-ENCODER RERANKING (top-5)
                      Joint query+passage scoring
                      Much more accurate than bi-encoder
                               │
                               ▼
                   5. GPT-4o-mini GENERATION
                      with numbered citation markers
                      enforced by system prompt
```

---

## License

MIT © 2024 Your Organization
