# Week 10 Verdict — Keep or Kill the Orchestrator?

**Sunk-cost bias in the room**: we already built the orchestrator and MCP servers in weeks 9–10, so there is a temptation to declare it the winner regardless of the numbers. Named and rejected.

The single agent passes **10/10** vs the orchestrator's **8/10** — the multi-agent pattern **loses** accuracy.

The orchestrator costs **$0.001443/question** vs **$0.000532/question** for the single agent — a **2.0x token multiplier** driven by context re-sends across three LLM calls (decompose + two workers + synthesise) instead of one.

p99 latency: single agent **29.121s** vs orchestrator **68.88s**. The client notices p99 during live negotiation.

**VERDICT: KILL (or demote to opt-in)** — the orchestrator delivers no accuracy gain (8/10 vs 10/10) while costing 2.0x more tokens. The complexity budget is not earned.
