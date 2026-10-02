"""Router package — classifies queries and dispatches to RAG, MCP, or both."""

from router.agent import route, RouteResult

__all__ = ["route", "RouteResult"]
