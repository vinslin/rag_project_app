# Week 9 — Task Set F Submission

| Checklist item | File |
|---|---|
| `agent_diff.txt` showing 0 changed lines in the agent module | [`agent_diff.txt`](agent_diff.txt) — empty file (0 lines) |
| Config diff adding the second server | [`config_diff.txt`](config_diff.txt) |
| `wire.json` — raw initialize, tools/list, tools/call, annotated | [`wire.json`](wire.json) (raw capture) + [`wire_annotated.md`](wire_annotated.md) (annotations + model-call location statement) |
| Tool count line: N before → M after, with tool names | [`tool_count_report.md`](tool_count_report.md) — **3 → 6**, includes live server-2 tool-call trace |
| `error_before_after.md` — same failing call, old docstring/error vs new | [`error_before_after.md`](error_before_after.md) (see also [`clause_server_error_rewrite_diff.txt`](clause_server_error_rewrite_diff.txt) for the exact code change) |
| `risk_note.md` — exactly 5 lines | [`risk_note.md`](risk_note.md) |

## How it's structured

- `agent.py` + `mcp_client.py` — the generic, server-agnostic agent and connector (never touched after the baseline commit-point; see `agent_diff.txt`).
- `servers/clause_server.py` — our own server (existing week7 tools, MCP-wrapped), with the `search_clause` error path rewritten mid-task.
- `servers/contract_repo_server.py` — the new "knowledge team" contract-repository server, added purely via `mcp_config.json`.
- `mcp_config.json` — the only file edited to go from 1 server to 2.
- `discover_tools.py`, `wire_capture.py`, `error_before_after.py` — the scripts used to produce the evidence files above; all runnable directly (`python -m week9.<name>`).

## Common mistakes checked against

- Tool list was never hardcoded after connecting — `agent.py` reads it from `tools/list` every run, which is exactly what makes the diff empty.
- No LLM call inside either MCP server — both servers are pure deterministic Python; the only Groq call in the whole system is in `agent.py`. Stated explicitly in `wire_annotated.md`.
- `get_definitions` (schedule resolution) stays a tool, not a resource — the model decides when a clause's defined term needs resolving, so it has to be something it can invoke, not context force-attached to every request.
- Not-found paths distinguish "lookup failed" from other states (e.g. `get_amendment_chain` explicitly distinguishes "unknown contract_id" from "valid contract_id, zero amendments on file") rather than collapsing everything into one generic error.

## Bonus (not attempted)

The gateway/audit-log/scoped-token-denial bonus was not built — flagging honestly rather than claiming partial credit. The core 100-point rubric above is complete and verified live (see `server2_trace.json` for the raw trace backing the tool-count report).
