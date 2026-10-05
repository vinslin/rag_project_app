# Cost Per Query — Breakdown by Stage

Based on the bad-answer trace (`trace_id: a3f1c2d4-...`) and representative RAG queries.

---

## Model Pricing (Groq)

| | Price |
|---|---|
| Input tokens | $0.10 / 1M tokens |
| Output tokens | $0.50 / 1M tokens |

---

## Per-Stage Cost — RAG Route (typical query)

| Stage | What runs | Tokens | Cost (USD) |
|---|---|---|---|
| **Retrieval** | BGE-base embedding (local) + ChromaDB (local) + BM25 (local) | 0 API tokens | $0.0000000 |
| **Reranking** | Cross-encoder ms-marco-MiniLM-L-6-v2 (local) | 0 API tokens | $0.0000000 |
| **Generation** | Groq LLM call (1,847 input + 312 output) | 2,159 tokens | $0.0003407 |
| **Tools** | No MCP tools called on RAG route | 0 | $0.0000000 |
| **Total** | | **2,159 tokens** | **$0.0003407** |

---

## Per-Stage Latency (from trace spans_ms)

| Stage | Latency |
|---|---|
| Retrieval | 118.4 ms |
| Reranking | 43.7 ms |
| Generation (LLM) | 912.1 ms |
| **Total** | **1,074.2 ms** |

---

## Notes

- Retrieval and reranking cost $0.00 because all models run locally (sentence-transformers,
  ChromaDB, BM25) — no external API calls.
- The entire per-query cost is the Groq generation call.
- MCP routes (mcp / both / full_doc_mcp) will be higher: each tool-calling loop
  iteration adds ~800–1,200 input tokens for tool schemas + results.
- Full-doc routes will be significantly higher: a 60,000-token document pass costs
  approximately $0.006 in input tokens alone.
