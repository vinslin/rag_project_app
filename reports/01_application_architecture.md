# Application Architecture
**Legal Contract Intelligence — Soft Suave AI Engineering League, Week 12**

---

## Overview

A Retrieval-Augmented Generation system for legal contract lookup. Given a natural-language question about one or more contracts, the system resolves the amendment chain, retrieves the operative clause, and returns a cited answer — or refers to counsel if the question requires legal characterisation.

**Entry point:** `streamlit run app.py`
**One trace ID from question in to answer out** — logged to `data/logs/`.

---

## System Diagram

```
User (browser)
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  app.py  —  Streamlit frontend                          │
│  • PDF upload & index build                             │
│  • Chat input / message history                         │
│  • Calls router_route(query, full_doc_texts, **kwargs)  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  router/agent.py  —  Route dispatcher                   │
│  • Generates trace_id, sets context var                 │
│  • Calls QueryClassifier → route decision               │
│  • Dispatches to one of 5 handlers                      │
│  • Logs every request to data/logs/requests.jsonl       │
└──────┬────────────────────────────────────┬─────────────┘
       │                                    │
       ▼                                    ▼
┌──────────────────┐              ┌─────────────────────────┐
│ router/           │              │ mcp/client.py           │
│ classifier.py    │              │ MCPHttpClient           │
│ LLM route        │              │ • Discovers tools at    │
│ decision         │              │   runtime (tools/list)  │
│ → rag / mcp /    │              │ • Dispatches tool calls │
│   both /         │              │ • Not hardcoded         │
│   full_doc /     │              └──────────┬──────────────┘
│   full_doc_mcp   │                         │
└──────────────────┘                         │
                                             │
       ┌─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│  pipeline/rag.py  —  RAG pipeline                       │
│  • Hybrid retrieval: vector (BGE-base) + BM25           │
│  • MMR reranking (cross-encoder ms-marco-MiniLM-L-6-v2) │
│  • Generation via Groq LLM                              │
│  • Guardrails: out-of-scope detection                   │
└──────┬──────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│  retrieval/vector_store.py  —  ChromaDB + BM25          │
│  • BGE-base-en-v1.5 embeddings (local, no API cost)     │
│  • BM25 index (rank-bm25)                               │
│  • Hybrid fusion: vector + BM25 scores merged           │
│  • MMR for diversity after reranking                    │
└──────┬──────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│  ingestion/                                             │
│  • pdf_loader.py — PyMuPDF extraction (digital + OCR)  │
│  • chunker.py    — token-level chunking with overlap    │
│  Documents stored in documents/                         │
└─────────────────────────────────────────────────────────┘
```

---

## Route Handlers

| Route | Trigger | What runs | LLM calls |
|-------|---------|-----------|-----------|
| `rag` | Specific clause question from uploaded docs | Hybrid retrieval → rerank → generate | 1 |
| `mcp` | Company standards question | MCP tool-calling loop | 1–8 (tool loop) |
| `both` | Validate contract clause against standards | RAG + MCP loop + synthesis | 2–9 |
| `full_doc` | Whole document analysis | Full doc text → single LLM pass | 1 |
| `full_doc_mcp` | Compliance review | Full doc + all MCP standards | 2–20 |

---

## Logging & Observability

Every request writes one structured JSON line to two files:

| File | What it captures |
|------|-----------------|
| `data/logs/requests.jsonl` | Full request: trace_id, query, answer, route, tokens, cost, spans_ms |
| `data/logs/classifications.jsonl` | Classifier decision: trace_id, route, reason, latency_ms |
| `data/logs/app.jsonl` | All INFO+ log lines from all modules, JSON, with trace_id |
| `data/logs/error.jsonl` | ERROR+ only, rotating, with trace_id |

**The same `trace_id` appears in all four files** — cross-log correlation is exact.

**Cost per query (RAG route, typical):**

| Stage | Model | Cost |
|-------|-------|------|
| Retrieval | BGE-base-en-v1.5 (local) | $0.00 |
| Reranking | ms-marco-MiniLM-L-6-v2 (local) | $0.00 |
| Generation | Groq LLM (~2,159 tokens) | $0.00034 |
| **Total** | | **$0.00034 / query** |

---

## Key Design Decisions

**Amendment resolution before citation** — the prompt instructs the model to prefer
the clause with the latest effective date. When two chunks cover the same section
(base MSA and an amendment), the amendment clause is cited and the base clause is
noted as superseded.

**Tools discovered at runtime** — the MCP client calls `tools/list` on startup and
caches the schema. Nothing is hardcoded. A third-party reviewer from another squad
can call the MCP server without any code changes on their end.

**Guardrails before generation** — out-of-scope detection runs before the LLM
generation call. Questions requiring legal characterisation ("is this standard?",
"is this acceptable?") are refused with an explicit referral to counsel.

**Single entry point** — `app.py` is the only way in. The eval suite (`run_eval.py`)
calls `pipeline.rag.answer_question` directly but through the same retrieval and
generation stack.

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| LLM | Groq (llama-3.3-70b or configured model) |
| Embeddings | sentence-transformers BGE-base-en-v1.5 (local) |
| Vector store | ChromaDB (local) |
| Lexical retrieval | rank-bm25 |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 (local) |
| PDF extraction | PyMuPDF (fitz) |
| MCP | mcp<2 |
| Structured logging | python-json-logger |
| Environment | python-dotenv |

---

## Directory Structure

```
rag_project_app/
├── app.py                    # Entry point
├── core/
│   ├── config.py             # Model names, retrieval params
│   ├── logging_setup.py      # configure_logging(), set_trace_id()
│   └── request_logger.py     # log_request(), iter_logs(), search_logs()
├── ingestion/
│   ├── pdf_loader.py         # PyMuPDF extraction
│   └── chunker.py            # Token-level chunking
├── retrieval/
│   └── vector_store.py       # ChromaDB + BM25 hybrid
├── pipeline/
│   └── rag.py                # End-to-end RAG pipeline
├── router/
│   ├── agent.py              # Route dispatcher, trace_id generation
│   ├── classifier.py         # LLM-based route classification
│   ├── doc_loader.py         # Full document loader
│   └── prompts.py            # All system prompts + version constants
├── generation/
│   └── prompts.py            # Generation prompt + PROMPT_VERSION
├── mcp/
│   └── client.py             # MCPHttpClient
├── guardrails/               # Out-of-scope detection
├── reranking/                # Cross-encoder reranking
├── evals/
│   ├── run_eval.py           # Eval runner (one command)
│   ├── golden_set.json       # 32 ground-truth cases
│   ├── results_v1.json       # Latest eval results
│   └── judge_v1.txt          # LLM judge prompt
├── data/logs/                # Runtime log files (gitignored)
├── documents/                # Uploaded contract PDFs
└── reports/                  # This folder
```
