# Week 8 — Fix Report: Missing Tool-Chain Gap

**Failure mode fixed**: the agent sometimes gives a final answer using only the raw clause text returned by `search_clause`, without calling `get_definitions` on a capitalized defined term the clause references (e.g. 'Business Day'). This is a trajectory gap, not necessarily an outcome failure -- the keyword grader can still mark it a pass if the clause text happens to contain the expected words.

**Fix applied** (`week8/fixed_agent.py`): (1) an explicit MUST-rule in the system prompt naming this contract's defined terms, and (2) a deterministic guard that scans gathered clause text for any known defined term left unresolved and forces one corrective `get_definitions` call before accepting the final answer.

## Before

| QID | Gap present | Passed |
|---|---|---|
| Q04 | False | True |
| Q07 | True | True |
| Q09 | True | True |
| Q10 | True | True |

**3/4 questions showed the gap.**

## After

| QID | Gap present | Passed | Forced corrections |
|---|---|---|---|
| Q04 | False | True | 1 |
| Q07 | False | True | 0 |
| Q09 | False | True | 0 |
| Q10 | False | True | 0 |

**0/4 questions showed the gap.**

## Result

Missing tool-chain gap: **3/4 → 0/4**.

## What could still get through

- The guard only recognizes the defined-term names hardcoded in this corpus (`week7.contracts.DEFINED_TERMS`). A new or renamed defined term (e.g. an Amendment 3 that introduces 'Force Majeure Event' as a defined term) would not be caught until the list is updated by hand.
- The guard checks whether get_definitions was *called* for the term, not whether the model actually *used* the returned definition correctly in its final answer -- a model could call the tool and still ignore the result.
- This fixes one specific failure mode (missing tool chaining). It does not address other modes from the same family (loops, wrong tool choice, made-up contract_version values) -- those would need their own guards.
