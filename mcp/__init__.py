"""MCP infrastructure layer — HTTP adapter for the Contract MCP server."""

from mcp.client import MCPHttpClient, MCPClientError

__all__ = ["MCPHttpClient", "MCPClientError"]
