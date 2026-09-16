# Week 8 — Trajectory Gap Report

Outcome (did it pass the keyword grader) vs trajectory (did it call the tools that make the answer reliable) for the 4 most tool-dependent questions from the Week 7 question set.

| QID | Passed | Expected Trajectory | Actual Trajectory | Gap Found | Reason |
|---|---|---|---|---|---|
| Q04 | True | search_clause → get_definitions | get_definitions → search_clause | False | none |
| Q07 | True | search_clause → get_definitions | get_definitions | True | answer graded correct but never called expected tool(s): {'search_clause'} |
| Q09 | True | search_clause | search_clause | False | none |
| Q10 | False | search_clause → search_clause | (none) | False | none |

**Summary**: 1/4 questions showed a trajectory gap (right-looking answer, unreliable path).

## Worst example

**Q07**: The payment clause refers to 'Service Period'. Resolve that term and include any schedule it references.

- Expected: search_clause → get_definitions
- Actual: get_definitions
- Why this matters: answer graded correct but never called expected tool(s): {'search_clause'}. The keyword grader scored this as a pass, but the agent reached it without doing the lookup that guarantees correctness — a different phrasing of the same question could easily fail.