"""HTTP client for the Contract MCP server.

Communicates with the MCP server over HTTP using JSON-RPC 2.0.
The server wraps every response in a Server-Sent Events (SSE) envelope,
so each response is parsed by stripping the "data: " prefix.

Usage:
    client = MCPHttpClient()
    if client.is_available():
        tools = client.to_groq_tools()
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

_MCP_URL = "http://localhost:5002/mcp"
_TIMEOUT = 10  # seconds

_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}

_id_counter = itertools.count(start=1)


def _next_id() -> int:
    return next(_id_counter)


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist={500, 502, 503, 504},
        allowed_methods={"POST"},
    )
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def _parse_sse(raw: str) -> dict:
    """Extract the JSON payload from an SSE-wrapped response."""
    for line in raw.splitlines():
        if line.startswith("data: "):
            return json.loads(line[6:])
    raise ValueError(f"No SSE data line found in MCP response: {raw!r}")


# ── Typed response dataclasses ────────────────────────────────────────────────

@dataclass
class ContractTemplate:
    template_id: int
    template_code: str
    template_name: str
    version: str
    description: str
    effective_date: str


@dataclass
class ContractSection:
    section_code: str
    section_name: str
    order: int
    required: bool
    description: str


@dataclass
class RequiredClause:
    clause_code: str
    clause_name: str
    required: bool
    description: str


@dataclass
class ClauseRequirement:
    requirement_code: str
    name: str
    description: str
    mandatory: bool


# ── Client ────────────────────────────────────────────────────────────────────

class MCPHttpClient:
    """Stateless HTTP client for the Contract MCP server."""

    def __init__(self, url: str = _MCP_URL):
        self._url = url
        self._session = _build_session()

    def _rpc(self, method: str, params: dict) -> dict:
        body = {"jsonrpc": "2.0", "id": _next_id(), "method": method, "params": params}
        try:
            resp = self._session.post(self._url, json=body, headers=_HEADERS, timeout=_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise MCPClientError(f"MCP HTTP error: {exc}") from exc
        try:
            return _parse_sse(resp.text)
        except (ValueError, json.JSONDecodeError) as exc:
            raise MCPClientError(f"MCP response parse error: {exc}") from exc

    def call_tool(self, name: str, arguments: dict) -> str:
        """Call any MCP tool by name and return its text content."""
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        content: list[dict] = result.get("result", {}).get("content", [])
        parts = [block["text"] for block in content if block.get("type") == "text"]
        if not parts:
            raise MCPClientError(f"Tool '{name}' returned no text content.")
        return "\n".join(parts)

    def _call_tool_json(self, name: str, arguments: dict) -> Any:
        raw = self.call_tool(name, arguments)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MCPClientError(f"Tool '{name}' returned non-JSON: {raw!r}") from exc

    def is_available(self) -> bool:
        """Return True if the MCP server responds to an initialize handshake."""
        try:
            self._rpc("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "router-agent", "version": "1.0"},
            })
            return True
        except MCPClientError as exc:
            logger.warning("MCP server unavailable: %s", exc)
            return False

    def get_latest_template(self) -> ContractTemplate:
        data = self._call_tool_json("get_latest_contract_template", {})
        return ContractTemplate(
            template_id=data["templateId"],
            template_code=data["templateCode"],
            template_name=data["templateName"],
            version=data["version"],
            description=data["description"],
            effective_date=data["effectiveDate"],
        )

    def get_contract_structure(self, template_id: int = 1) -> list[ContractSection]:
        data = self._call_tool_json("get_contract_structure", {"templateId": template_id})
        return [
            ContractSection(
                section_code=s["sectionCode"],
                section_name=s["sectionName"],
                order=s["order"],
                required=s["required"],
                description=s["description"],
            )
            for s in data.get("sections", [])
        ]

    def get_required_clauses(self, template_id: int = 1) -> list[RequiredClause]:
        data = self._call_tool_json("get_required_clauses", {"templateId": template_id})
        return [
            RequiredClause(
                clause_code=c["clauseCode"],
                clause_name=c["clauseName"],
                required=c["required"],
                description=c["description"],
            )
            for c in data.get("requiredClauses", [])
        ]

    def get_clause_requirement(self, clause_code: str, template_id: int = 1) -> list[ClauseRequirement]:
        data = self._call_tool_json(
            "get_clause_requirement",
            {"templateId": template_id, "clauseCode": clause_code},
        )
        return [
            ClauseRequirement(
                requirement_code=r["requirementCode"],
                name=r["name"],
                description=r["description"],
                mandatory=r["mandatory"],
            )
            for r in data.get("requirements", [])
        ]

    def list_tools(self) -> list[dict]:
        """Fetch the raw tool list from the MCP server via tools/list."""
        result = self._rpc("tools/list", {})
        tools = result.get("result", {}).get("tools", [])
        logger.info("Discovered %d tools from MCP server", len(tools))
        return tools

    # Fields Groq/OpenAI function-calling accepts at the property level.
    _ALLOWED_PROP_KEYS = frozenset({
        "type", "description", "enum", "items", "properties",
        "required", "anyOf", "oneOf", "allOf",
    })

    @classmethod
    def _sanitize_schema(cls, schema: dict | None) -> dict:
        """Strip fields Groq doesn't accept (notably 'default') from an MCP inputSchema."""
        if not isinstance(schema, dict):
            return {"type": "object", "properties": {}}
        sanitized: dict = {"type": schema.get("type", "object")}
        raw_props: dict = schema.get("properties") or {}
        sanitized["properties"] = {
            k: {pk: pv for pk, pv in v.items() if pk in cls._ALLOWED_PROP_KEYS}
            for k, v in raw_props.items()
            if isinstance(v, dict)
        }
        if "required" in schema:
            sanitized["required"] = schema["required"]
        return sanitized

    def to_groq_tools(self) -> list[dict]:
        """Discover tools from the server and convert to Groq function-calling format.

        MCP inputSchema is sanitized to remove fields Groq does not accept.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name":        tool["name"],
                    "description": tool.get("description", ""),
                    "parameters":  self._sanitize_schema(tool.get("inputSchema")),
                },
            }
            for tool in self.list_tools()
        ]

    def dispatch(self, name: str, args: dict) -> str:
        """Dispatch a tool call by name; used inside the LLM tool-calling loop."""
        try:
            return self.call_tool(name, args)
        except MCPClientError as exc:
            logger.error("MCP tool '%s' failed: %s", name, exc)
            return f"[MCP ERROR] {exc}"


class MCPClientError(Exception):
    """Raised when the MCP HTTP client encounters a protocol or network error."""
