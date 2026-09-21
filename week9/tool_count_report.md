# Week 9 — Tool Count Report (from real `tools/list`, not notes)

All counts below come from actually running `python -m week9.discover_tools`
against the live `week9/mcp_config.json`, at two points in time — never
hand-typed.

## Before (1 server configured)

`week9/mcp_config.json` listed only `clause_search`.

```json
{
  "servers": ["clause_search"],
  "total_tools": 3,
  "all_tool_names": ["search_clause", "get_effective_date", "get_definitions"]
}
```

**3 tools discovered.**

## After (2 servers configured — `contract_repository` added, config-only)

```json
{
  "servers": ["clause_search", "contract_repository"],
  "by_server": {
    "clause_search": ["search_clause", "get_effective_date", "get_definitions"],
    "contract_repository": ["lookup_contract", "get_contract_effective_date", "get_amendment_chain"]
  },
  "total_tools": 6,
  "all_tool_names": [
    "search_clause", "get_effective_date", "get_definitions",
    "lookup_contract", "get_contract_effective_date", "get_amendment_chain"
  ]
}
```

**6 tools discovered.**

## Result

**3 → 6 tools**, purely from editing `week9/mcp_config.json` (see `week9/config_diff.txt`). `week9/agent.py` was not touched (see `week9/agent_diff.txt`, 0 lines).

## Live proof the new server's tools are actually reachable

Query: *"What is the amendment chain for contract MSA-2026-014, according to the contract repository?"*

Trace (from `week9/_server2_trace.json`):

```json
{
  "name": "get_amendment_chain",
  "server": "contract_repository",
  "args": { "contract_id": "MSA-2026-014" }
}
```

The tool name `get_amendment_chain` and its owning server `contract_repository` are both new to the "after" tool list — this is not one of the original 3 tools, so this call provably exercises the newly bolted-on server, discovered purely from config.
