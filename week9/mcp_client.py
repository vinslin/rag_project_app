"""Generic MCP multi-server connector.

Reads a config listing zero or more MCP servers, connects to each over
stdio, discovers their tools via the real tools/list call, and hands back
one merged tool pool (in Groq/OpenAI function-calling format) plus a
dispatch map from tool name -> the server session that owns it.

This module has NO knowledge of what any specific server or tool is
named. That is what makes adding a second server in week9/mcp_config.json
a config-only change: nothing here branches on server identity.
"""

import json
import os
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mcp_tool_to_groq(tool, server_name: str) -> dict:
    """Convert one MCP Tool (name, description, inputSchema) into a
    Groq/OpenAI-style function-calling tool definition. MCP's inputSchema
    is already standard JSON Schema, so no case-conversion is needed
    (unlike week7's Gemini-style OBJECT/STRING declarations)."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": f"[{server_name}] {tool.description or ''}",
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        },
    }


class MCPToolPool:
    """Holds open sessions to every configured server plus the merged,
    Groq-ready tool list and a name -> session dispatch map.

    Usage:
        async with MCPToolPool(config) as pool:
            response = groq_call(..., tools=pool.groq_tools)
            result = await pool.call_tool(name, args)
    """

    def __init__(self, config: dict):
        self.config = config
        self._stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}     # server_name -> session
        self.tool_owner: dict[str, str] = {}              # tool_name -> server_name
        self.groq_tools: list[dict] = []
        self.discovered: dict[str, list[str]] = {}        # server_name -> [tool names]

    async def __aenter__(self):
        for server_cfg in self.config.get("servers", []):
            server_name = server_cfg["name"]
            params = StdioServerParameters(
                command=server_cfg["command"],
                args=server_cfg.get("args", []),
                env=server_cfg.get("env"),
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            tools_result = await session.list_tools()
            names = []
            for tool in tools_result.tools:
                if tool.name in self.tool_owner:
                    raise ValueError(
                        f"Tool name collision: '{tool.name}' exposed by both "
                        f"'{self.tool_owner[tool.name]}' and '{server_name}'"
                    )
                self.tool_owner[tool.name] = server_name
                self.groq_tools.append(mcp_tool_to_groq(tool, server_name))
                names.append(tool.name)

            self.sessions[server_name] = session
            self.discovered[server_name] = names

        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._stack.aclose()

    async def call_tool(self, name: str, arguments: dict) -> str:
        server_name = self.tool_owner.get(name)
        if server_name is None:
            return f"Unknown tool: {name}"
        session = self.sessions[server_name]
        result = await session.call_tool(name, arguments)
        parts = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        text = "\n".join(parts) if parts else "(empty result)"
        if result.isError:
            return f"[TOOL ERROR] {text}"
        return text


def default_config_path() -> str:
    return os.path.join(os.path.dirname(__file__), "mcp_config.json")
