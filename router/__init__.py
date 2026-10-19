"""Router package — classifies queries and dispatches to RAG, MCP, or both.

Routes:
    rag          → RAG chunk retrieval
    mcp          → MCP standards server
    both         → RAG + MCP synthesis (clause-level compliance)
    full_doc     → Full document analysis (no MCP)
    full_doc_mcp → Full document + all MCP standards (compliance report)
"""

from router.agent import route, RouteResult
from router.classifier import Route

__all__ = ["route", "RouteResult", "Route"]
