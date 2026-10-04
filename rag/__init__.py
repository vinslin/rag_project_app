"""RAG package — backward-compatibility shim.

The actual implementation has moved to clean-architecture packages:
    core/        config and constants
    ingestion/   PDF loading and chunking
    retrieval/   vector store, BM25, hybrid search, MMR
    reranking/   cross-encoder reranker
    generation/  LLM generation
    guardrails/  input safety screening
    pipeline/    RAG use-case orchestrator
    mcp/         MCP HTTP adapter
    router/      query routing agent
    ui/          Streamlit components
"""
