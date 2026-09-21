"""The Week 9 MCP-driven agent.

This module is deliberately generic: it has no import of, or reference
to, any specific MCP server or tool name. It reads week9/mcp_config.json,
connects to whatever servers are listed (via week9.mcp_client.MCPToolPool),
and runs a standard Groq tool-calling loop over the merged, dynamically
discovered tool pool.

This is the file whose line count must NOT change when a second server is
added to mcp_config.json — see week9/agent_diff.txt.
"""

import asyncio
import json
import time

from rag import config as rag_config
from week7.llm_client import get_client, groq_call_with_retry
from week9.mcp_client import MCPToolPool, load_config, default_config_path

MODEL = rag_config.GENERATION_MODEL

SYSTEM_PROMPT = (
    "You are a legal contract analysis agent. Answer the user's question "
    "using whatever tools are available to you. Tools may come from more "
    "than one server — you do not need to know which server a tool comes "
    "from, only what it does, from its description. Call the appropriate "
    "tools to gather information, then give a precise, cited answer."
)


async def run_agent_mcp(
    question: str,
    *,
    config_path: str | None = None,
    max_iters: int = 8,
    log: list | None = None,
) -> dict:
    """Run the MCP tool-calling agent loop and return a structured result."""
    if log is None:
        log = []

    cfg = load_config(config_path or default_config_path())
    client = get_client()

    async with MCPToolPool(cfg) as pool:
        log.append(f"Discovered servers: {list(pool.discovered.keys())}")
        for server_name, names in pool.discovered.items():
            log.append(f"  [{server_name}] tools: {names}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        tools_called: list[dict] = []
        iteration = 0
        final_answer = None
        start = time.time()

        while iteration < max_iters:
            iteration += 1
            response = groq_call_with_retry(
                client, model=MODEL, messages=messages, tools=pool.groq_tools, log=log,
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls
                    ],
                })
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    owner = pool.tool_owner.get(tc.function.name, "?")
                    log.append(f"Tool call: [{owner}] {tc.function.name}({json.dumps(args)})")

                    result = await pool.call_tool(tc.function.name, args)
                    tools_called.append({
                        "name": tc.function.name,
                        "server": owner,
                        "args": args,
                        "result_preview": result[:300],
                    })
                    log.append(f"Tool result: {result[:300]}")
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            else:
                final_answer = msg.content
                break

        elapsed = time.time() - start
        return {
            "answer": final_answer or "[no answer produced within max_iters]",
            "tools_called": tools_called,
            "iterations": iteration,
            "latency_s": round(elapsed, 3),
            "log": log,
        }


def run_agent(question: str, **kwargs) -> dict:
    """Sync convenience wrapper around run_agent_mcp."""
    return asyncio.run(run_agent_mcp(question, **kwargs))


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What is the termination notice period under Section 8.2?"
    r = run_agent(q)
    print("ANSWER:", r["answer"])
    print("TOOLS CALLED:", [(t["server"], t["name"]) for t in r["tools_called"]])
