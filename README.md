# Legal Contract Intelligence (RAG Application)

A Streamlit-based **Legal Contract Intelligence** application for analyzing contract PDFs using Retrieval-Augmented Generation (RAG). The app ingests uploaded contracts, indexes them into a hybrid retrieval pipeline (vector + BM25), reranks results, applies guardrails, and generates grounded answers with citations. It can also route queries to MCP-based standards/compliance workflows when an MCP server is available.

## Features

- **Streamlit chat interface** for contract Q&A (`app.py`)
  - PDF upload from the sidebar
  - Automatic re-indexing when uploaded files change
  - Indexed chunk count display and chat reset
  - Friendly no-document chat mode when no PDFs are indexed
- **Document ingestion & chunking**
  - PDF text/table extraction and metadata parsing (`ingestion/pdf_loader.py`)
  - Section-aware chunking with overlap and clause-reference extraction (`ingestion/chunker.py`)
- **Hybrid retrieval pipeline**
  - Dense embeddings via `sentence-transformers` (`BAAI/bge-base-en-v1.5`)
  - Persistent ChromaDB vector store (`retrieval/vector_store.py`, default `./data/chroma`)
  - BM25 keyword search with persisted corpus (`retrieval/bm25_search.py`, default `./data/bm25_corpus.json`)
  - Reciprocal Rank Fusion (RRF) for hybrid search (`retrieval/hybrid_search.py`)
  - MMR diversification (`retrieval/mmr.py`)
  - Cross-encoder reranking (`reranking/reranker.py`, `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Grounded answer generation**
  - RAG answer generation with Groq-hosted LLM (`generation/generator.py`, `core/llm_client.py`)
  - Context-only prompting with citation-oriented responses (`generation/prompts.py`)
- **Guardrails**
  - Blocks empty, prompt-injection, and drafting requests (`guardrails/checker.py`)
- **Query routing / agent orchestration**
  - LLM-based route classification: `rag`, `mcp`, `both`, `full_doc`, `full_doc_mcp` (`router/classifier.py`)
  - MCP tool-calling and compliance/full-document analysis flows (`router/agent.py`, `router/doc_loader.py`)
  - Graceful fallback when MCP is unavailable
- **MCP standards integration**
  - HTTP JSON-RPC/SSE MCP client with dynamic tool discovery (`mcp/client.py`)
  - Default MCP endpoint: `http://localhost:5002/mcp` (configurable via `MCP_URL`)
- **Observability & logging**
  - Optional Langfuse tracing (`core/observability.py`)
  - Structured JSON logs and request/classification logs under `data/logs/` (`core/logging_setup.py`, `core/request_logger.py`)
- **Evaluation & tests**
  - Evaluation datasets/scripts in `evals/`
  - Unit and integration tests in `tests/`
  - Project reports in `reports/`

## How It Works (High-Level)

1. **Upload PDFs** in the Streamlit sidebar.
2. The app saves uploads to `documents/`, extracts text/pages, chunks content, and rebuilds the index.
3. Chunks are stored in **ChromaDB** and a **BM25 corpus** for hybrid retrieval.
4. When you ask a question, the router classifies the request and selects a handling path (RAG, MCP, or combined/full-document flows).
5. For RAG, the system retrieves candidates, applies **MMR** and **cross-encoder reranking**, then generates a grounded answer using the LLM.
6. Responses can include route/debug information in the UI and are logged for observability/evaluation.

> Note: The Streamlit uploader currently accepts PDFs. Some ingestion utilities support additional formats, but the main UI flow is PDF-based.

## Project Structure

```text
rag_project_app/
├── app.py                     # Streamlit entry point (UI, upload/indexing, chat flow)
├── Dockerfile                 # Container image for running the Streamlit app
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
├── .streamlit/                # Streamlit configuration directory
├── core/                      # Shared config, LLM client, logging, observability, request logging
├── ingestion/                 # PDF/document loading and chunking
├── retrieval/                 # Embeddings, Chroma vector store, BM25, hybrid search, MMR
├── reranking/                 # Cross-encoder reranking
├── generation/                # RAG answer generation and prompts
├── guardrails/                # Query screening/guardrails
├── pipeline/                  # High-level RAG pipeline orchestration
├── router/                    # Query classification and route/agent execution
├── mcp/                       # MCP HTTP client integration
├── ui/                        # Streamlit rendering/history helpers
├── evals/                     # Evaluation datasets, scripts, judge prompts/results
├── tests/                     # Unit/integration tests and fixtures
├── reports/                   # Architecture/evaluation/weakness reports
├── documents/                 # Uploaded/sample contract PDFs
├── data/                      # Runtime data (Chroma, BM25 corpus, logs, etc.)
└── week7/                     # Additional project artifacts
```

## Prerequisites

- **Python**: 3.11 recommended (matches the provided `Dockerfile`)
- **pip** and a virtual environment tool (`venv`)
- **Groq API key** for LLM-powered responses/routing
- **(Optional) MCP server** for standards/compliance/MCP-routed queries
- **(Optional) Langfuse account/keys** for tracing/observability

### MCP Server (External Dependency)

This application can integrate with an MCP standards server for MCP-based routes (`mcp`, `both`, `full_doc_mcp`).

- **MCP server repository:** https://github.com/vinslin/contarct_mcp_server
- **How to get/use it:** Clone and run the MCP server from the repository above, then configure this app to point to it using `MCP_URL`.
- **Default MCP URL in this app:** `http://localhost:5002/mcp`

If the MCP server is not running or unreachable, the router falls back to non-MCP flows where possible.

## Installation

1. **Clone this repository** and open the project directory:
   ```bash
   git clone <this-repo-url>
   cd rag_project_app
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # macOS/Linux
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

> Note: On first run, embedding/reranker models may be downloaded by `sentence-transformers`, so initial startup can take longer and requires internet access (unless models are already cached).

## Configuration

Create a `.env` file in the project root (you can copy from `.env.example`) and configure the required values.

```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# macOS/Linux
cp .env.example .env
```

### Environment Variables

| Variable | Required | Default / Behavior | Description |
|---|---|---|---|
| `GROQ_API_KEY` | **Yes** (for LLM features) | None | Groq API key used for answer generation, routing/classification, and no-document replies. |
| `MCP_URL` | No | `http://localhost:5002/mcp` | MCP standards server endpoint used by MCP-routed flows. Set this to your running MCP server URL. |
| `CHROMA_PATH` | No | `./data/chroma` (documented default) | Intended ChromaDB persistence path (see `.env.example`). |
| `COLLECTION_NAME` | No | `legal_contracts` | Chroma collection name for indexed chunks. |
| `BM25_CORPUS_PATH` | No | `./data/bm25_corpus.json` | BM25 corpus cache file path. |
| `LANGFUSE_PUBLIC_KEY` | No | Disabled if unset | Langfuse public key for observability/tracing. |
| `LANGFUSE_SECRET_KEY` | No | Disabled if unset | Langfuse secret key for observability/tracing. |
| `LANGFUSE_HOST` | No | `https://cloud.langfuse.com` | Langfuse host URL (override for self-hosted). |

> The app can run without Langfuse. MCP features require a reachable MCP server (see: https://github.com/vinslin/contarct_mcp_server).

## Running the Application

### Option 1: Run Locally (Streamlit)

From the project root (with your virtual environment activated and `.env` configured):

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`).

**Usage:**
1. Upload one or more contract PDFs from the sidebar.
2. Wait for indexing to complete (chunk count appears in the sidebar).
3. Ask questions in the chat.
4. Use "Clear Chat" to reset the conversation state.

### Option 2: Run with Docker

A `Dockerfile` is included for containerized runs.

#### Build the image
```bash
docker build -t rag-contract-app .
```

#### Run the container
```bash
docker run --rm -p 8501:8501 --env-file .env rag-contract-app
```

Open: `http://localhost:8501`

#### Persist runtime data (recommended)

To keep indexed data, BM25 corpus, logs, and uploaded documents across container restarts, mount volumes:

```bash
# PowerShell
docker run --rm -p 8501:8501 --env-file .env `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/documents:/app/documents `
  rag-contract-app

# macOS/Linux
docker run --rm -p 8501:8501 --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/documents:/app/documents \
  rag-contract-app
```

#### Docker networking note for MCP
If the MCP server runs on your host machine, `http://localhost:5002/mcp` inside the container will not reach the host by default. Set `MCP_URL` to a container-reachable address (for example, a Docker network service name or an appropriate host alias depending on your setup).

## MCP Server Setup (for MCP-Routed Features)

To enable MCP-based standards/compliance functionality:

1. Get the MCP server from: **https://github.com/vinslin/contarct_mcp_server**
2. Follow that repository's setup/run instructions to start the server.
3. Set `MCP_URL` in this app's `.env` to the server endpoint (default expected: `http://localhost:5002/mcp`).
4. Start/restart this app and ask MCP-related questions (or queries that route to `mcp`, `both`, `full_doc_mcp`).

## Data, Logs, and Runtime Artifacts

- **ChromaDB data:** `data/chroma/` (persistent vector index)
- **BM25 corpus cache:** `data/bm25_corpus.json`
- **Application/request logs:** `data/logs/` (e.g. app/error/request/classification logs)
- **Uploaded documents:** `documents/`

These paths may be created at runtime. Mount/persist them in Docker if you want data to survive container restarts.

## Evaluation

The repository includes evaluation assets and scripts under `evals/` (e.g. `golden_set.json`, `dataset.json`, `run_eval.py`, judge prompts/results).

> **Prerequisites:** Ensure dependencies are installed and your `.env` is configured (at minimum `GROQ_API_KEY` for live recompute/judging). If the Chroma collection is empty, `run_eval.py` can auto-build the index from PDFs in `documents/`.

### Common commands

```bash
# Run evaluation using cached predictions (if available)
python evals/run_eval.py

# Recompute predictions by running the pipeline, then evaluate (live run)
python evals/run_eval.py --recompute

# Recompute predictions and evaluate with a specific judge version (v1 or v2)
python evals/run_eval.py --recompute --judge v1
python evals/run_eval.py --recompute --judge v2
```

**Notes:**
- `--recompute` regenerates predictions through the live RAG/router pipeline and may take time.
- Evaluation/judging may make LLM calls and can be subject to API rate limits/costs.
- Outputs/results are written under `evals/` (e.g. judge-specific result files depending on the selected mode).

## Tests

Tests are available under `tests/` (`unit/`, `integration/`, fixtures).

### Common commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run only unit tests
pytest tests/unit

# Run only integration tests
pytest tests/integration

# Run a specific test file
pytest tests/unit/test_chunker.py

# Run tests matching a keyword/expression
pytest -k "router"
```

> If `pytest` is not installed in your environment, install it first (for example: `pip install pytest`) before running the commands above.

## Troubleshooting

- **No LLM responses / auth errors:** Verify `GROQ_API_KEY` is set correctly in `.env`.
- **MCP routes unavailable:** Ensure the MCP server is running and `MCP_URL` is reachable from the app environment (especially in Docker). MCP server repo: https://github.com/vinslin/contarct_mcp_server
- **Slow first startup:** `sentence-transformers` embedding/reranker models may download on first use.
- **Index/data not persisting in Docker:** Mount `data/` and `documents/` as volumes.
- **Port conflicts:** Streamlit defaults to port `8501`; change the host port mapping in Docker (e.g. `-p 8502:8501`) if needed.
