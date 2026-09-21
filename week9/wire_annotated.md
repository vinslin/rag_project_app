# Week 9 — Raw Wire Capture, Annotated

Source: `week9/wire.json`, produced by `week9/wire_capture.py`, which talks
directly to the `contract_repository` server's stdin/stdout as raw
newline-delimited JSON-RPC 2.0 — no SDK object abstraction in between.
This is the literal protocol traffic for `initialize -> tools/list -> tools/call`.

---

## Message 1 — `client -> server`: `initialize` request

```json
{"jsonrpc": "2.0", "id": 1, "method": "initialize",
 "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {...}}}
```

| Field | Meaning |
|---|---|
| `jsonrpc` | Protocol version marker — MCP rides on JSON-RPC 2.0, every message carries this. |
| `id` | Correlates this request to its response (message 2 echoes `id: 1`). Notifications (message 3) omit this. |
| `method` | The RPC method being invoked — here, the MCP handshake method every session must call first. |
| `params.protocolVersion` | The MCP protocol version the client speaks; server can accept, downgrade, or reject. |
| `params.capabilities` | What the *client* supports (empty here — this capture script is minimal). Real clients advertise things like `sampling` support. |
| `params.clientInfo` | Free-form client identification (name/version), for logging/debugging on the server side. |

## Message 2 — `server -> client`: `initialize` response

```json
{"jsonrpc": "2.0", "id": 1,
 "result": {"protocolVersion": "2024-11-05", "capabilities": {...}, "serverInfo": {...}}}
```

| Field | Meaning |
|---|---|
| `id: 1` | Matches the request — this IS the answer to message 1. |
| `result.protocolVersion` | Server confirming the negotiated protocol version. |
| `result.capabilities.tools` | Server declares it supports the `tools` capability (`listChanged: false` = it won't push list-changed notifications; a static tool set). |
| `result.capabilities.resources` / `.prompts` | Declared but unused here — this server exposes no resources or prompts, only tools. |
| `result.serverInfo` | Identifies the server (`contract_repository`) and the MCP SDK version it's running (`1.30.0`) — this is metadata, not our application version. |

## Message 3 — `client -> server`: `notifications/initialized`

```json
{"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
```

| Field | Meaning |
|---|---|
| No `id` | This is a **notification**, not a request — by JSON-RPC convention, no `id` means "don't send a response." |
| `method` | Tells the server the handshake is complete and the client is ready to make real calls. The server does not reply to this. |

## Message 4 — `client -> server`: `tools/list` request

```json
{"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
```

| Field | Meaning |
|---|---|
| `id: 2` | New request id — independent counter from the client, not tied to `id: 1`. |
| `method` | Asks the server to enumerate every tool it exposes. **This is the dynamic-discovery call** — the client never hardcodes tool names; it asks. |
| `params` | Empty here; would carry a pagination `cursor` on a follow-up call if the server paginates. |

## Message 5 — `server -> client`: `tools/list` response

```json
{"jsonrpc": "2.0", "id": 2, "result": {"tools": [ {...}, {...}, {...} ]}}
```

| Field | Meaning |
|---|---|
| `result.tools` | Array of exactly the 3 tools this server exposes: `lookup_contract`, `get_contract_effective_date`, `get_amendment_chain`. |
| `tools[].name` | The literal string the client must send back in a `tools/call` to invoke this tool — this is what `week9/mcp_client.py` reads to build the Groq tool schema. |
| `tools[].description` | **This is literally the Python docstring** from the `@mcp.tool()`-decorated function in `contract_repo_server.py` — the docstring IS the prompt the model sees, verbatim, including the "if not found, lists valid ids" guidance written into it. |
| `tools[].inputSchema` | Auto-generated JSON Schema from the function's type hints (`contract_id: str` → `{"type": "object", "properties": {"contract_id": {"type": "string"}}, "required": ["contract_id"]}`). This is fed straight into Groq's tool-calling `parameters` field with no reformatting needed. |
| `tools[].outputSchema` | What a structured result looks like (`{"result": <string>}`) — MCP's newer structured-output convention; our client only reads the plain-text `content` block, not this. |

## Message 6 — `client -> server`: `tools/call` request

```json
{"jsonrpc": "2.0", "id": 3, "method": "tools/call",
 "params": {"name": "lookup_contract", "arguments": {"contract_id": "MSA-2026-014"}}}
```

| Field | Meaning |
|---|---|
| `id: 3` | Third request in this session. |
| `method` | Invoke a specific tool by name. |
| `params.name` | Must exactly match a `name` returned by `tools/list` (message 5) — `lookup_contract`. |
| `params.arguments` | The actual arguments, matching the `inputSchema` from message 5 — `{"contract_id": "MSA-2026-014"}` is where the LLM's chosen argument value lands. |

## Message 7 — `server -> client`: `tools/call` response

```json
{"jsonrpc": "2.0", "id": 3,
 "result": {"content": [{"type": "text", "text": "Contract ID: MSA-2026-014..."}], "structuredContent": {...}, "isError": false}}
```

| Field | Meaning |
|---|---|
| `result.content` | Array of content blocks (MCP supports text/image/etc.) — our client (`week9/mcp_client.py`) reads `block.text` from these. |
| `result.structuredContent` | The same answer, also as a typed JSON object per `outputSchema` — unused by our simple text-based client, but available for clients that want structured data instead of prose. |
| `result.isError` | `false` here — the tool executed successfully. If `true`, the *same* content field carries the error message, letting the model see and recover from tool-level failures without the whole call crashing (this is the mechanism `week9/error_before_after.md`'s recoverable-error rewrite relies on). |

---

## Where the model call happens — and where it doesn't

**None of the 7 messages above involve an LLM call.** This entire exchange is pure MCP protocol traffic between `week9/wire_capture.py` (a plain Python client) and the `contract_repository` subprocess — the model is never invoked here. The only place an LLM is called in this whole system is inside `week9/agent.py`'s `groq_call_with_retry(...)` (via `week7/llm_client.py`), which happens in the *host* (our agent process), strictly after tool discovery and strictly outside any MCP server — the server exposes capabilities, the host decides when and how to use a model, exactly as the architecture separates the two.
