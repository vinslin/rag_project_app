## Verdict

No question in this set of 10 strictly requires an agent. The workflow matches or exceeds the agent on all inputs.

Cost and latency are comparable between the two systems.

**Decision rule**: The path varies by input for the defined-term-chase class — the agent must dynamically decide to call get_definitions after seeing the clause text reference a term. For single-clause lookups and date queries, the workflow suffices. A production system should use the workflow as the default path and escalate to the agent only when the clause text contains unresolved defined terms.