# Week 9 — Supply-Chain Risk Note: `contract_repository` MCP server

1. **Who wrote it**: the knowledge team, a different team in our own org — but from our agent's trust boundary it is still third-party code and infrastructure we don't control or review on our release cycle.
2. **What it can reach**: whatever the token we give it is scoped to — currently, every contract and its full amendment chain, including executed agreements under NDA, since `lookup_contract`/`get_amendment_chain` take an arbitrary `contract_id` with no per-caller filtering visible to us.
3. **What it logs**: unknown to us — we have no visibility into their server's logging, retention, or who else can read those logs; our questions and contract IDs queried are effectively disclosed to a system we don't audit.
4. **What a stolen token could do**: read the entire contract repository (all contracts, all amendments) at whatever rate their API allows, with no additional authentication once the token is valid.
5. **Ship or don't**: ship, but only with a read-only, narrowly-scoped token (ideally limited to the specific contract IDs Legal needs before Thursday), our own audit log of every `tools/call` we make against it, and a follow-up request to the knowledge team for their logging/retention policy before renewal review.
