# Week 10 Race Table — Single Agent vs Multi-Agent Orchestrator

**Eval set**: 10 Week-6 cases (Q01–Q10), unchanged from Week 7 race.  
**Judge**: keyword match (same `_grade()` function as `week7/race.py`).  
**Model**: `openai/gpt-oss-20b` via Groq for both arms.  

| Case ID | Class | Single Pass | Orch Pass | Single Tok | Orch Tok | Single Lat (s) | Orch Lat (s) |
|---------|-------|-------------|-----------|------------|----------|----------------|---------------|
| Q01 | simple_clause_lookup | ✓ | ✓ | 2,043 | 4,452 | 1.375 | 4.191 |
| Q02 | amendment_override | ✓ | ✓ | 2,639 | 5,633 | 1.756 | 23.501 |
| Q03 | definition_lookup | ✓ | ✓ | 1,343 | 3,936 | 5.092 | 15.514 |
| Q04 | defined_term_chase_DEP | ✓ | ✓ | 3,425 | 9,840 | 20.298 | 68.88 |
| Q05 | date_lookup | ✓ | ✓ | 1,300 | 1,926 | 5.961 | 7.896 |
| Q06 | date_computation_DEP | ✓ | ✓ | 6,485 | 5,245 | 41.366 | 24.96 |
| Q07 | defined_term_schedule_chase_DEP | ✓ | ✓ | 1,916 | 8,379 | 9.068 | 51.956 |
| Q08 | version_comparison | ✓ | ✗ | 4,326 | 12,782 | 29.121 | 86.483 |
| Q09 | clause_with_definition | ✓ | ✓ | 1,296 | 2,381 | 7.026 | 8.108 |
| Q10 | amendment_tracking | ✓ | ✗ | 2,078 | 0 | 8.823 | 0 |

## Aggregate Metrics (4 × 2)

| Metric | Single Agent | Orchestrator |
|--------|-------------|---------------|
| Pass rate | 10/10 | 8/10 |
| p50 latency (s) | 7.925 | 19.508 |
| p99 latency (s) | 29.121 | 68.88 |
| Total tokens (10 qs) | 26,851 | 54,574 |
| Cost per question (USD) | $0.000532 | $0.001443 |

**Context re-send multiplier**: 2.0x  
**Dominant hand-off**: orchestrator → clause_worker (Q04) — 6,679 tokens (12% of all orchestrator tokens)
