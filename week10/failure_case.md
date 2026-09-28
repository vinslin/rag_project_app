# Week 10 — Worker Failure Case

**Case**: Q04 — `N/A`  
**Question**: The termination clause in Section 8.2 refers to 'Business Day'. What does that term mean, and does it include a holiday schedule?  

## Injected failure

The `defined_terms_worker` was configured to raise a simulated `500 Internal Server Error` on its **first** call for this question_id.  
The orchestrator was not told this would happen — it had to discover and handle it at runtime.

## Orchestrator actual behaviour

**RETRIED** — the orchestrator caught the 500, logged it, and immediately re-dispatched the same `defterm_task` to the defined-terms worker. The second call succeeded (the one-shot failure flag was already consumed). The synthesis step received the full definition and produced a complete answer.

**One-line verdict**: orchestrator **retried** — answer quality preserved.

## Single agent result (no failure injected)

Pass: `True`  
Answer (first 400 chars): **Definition of “Business Day”**

The term *Business Day* is defined in the **Master Service Agreement (MSA‑2026‑014, v1_master)** as follows:

> “**Business Day** – Any day other than a Saturday, Sunday, or public holiday observed in the jurisdiction specified in Section 15 (Governing Law). For the avoidance of doubt, the holiday calendar applicable to this Agreement is set forth in **Schedule B 

## Orchestrator result (failure injected)

Pass: `True`  
Answer (first 400 chars): **Answer**

| Fact | Contract version & section | Citation |
|------|----------------------------|----------|
| “Business Day” is defined as any day that is **not** a Saturday, Sunday or a public holiday observed in the jurisdiction specified in Section 15 (Governing Law). The definition also states that the applicable holiday calendar is set out in Schedule B (Holiday Calendar). | **v1_master** –

## Execution log (failure case only)

```

[orchestrator] DECOMPOSE  qid=Q04
[orchestrator] decompose tokens: input=241 output=368
[orchestrator] decompose plan raw: {"clause_task":"Retrieve the text of Section 8.2 and its effective dates.","defterm_task":"Provide the definition of 'Business Day' in this contract and indicate whether it includes a holiday schedule."}
[orchestrator] clause_task:  Retrieve the text of Section 8.2 and its effective dates.
[orchestrator] defterm_task: Provide the definition of 'Business Day' in this contract and indicate whether it includes a holiday schedule.

[orchestrator → clause_worker] sending: Retrieve the text of Section 8.2 and its effective dates.
[clause_worker] START
  [worker iter 1] tokens: input=334 output=160
  [worker tool] search_clause({"clause_name": "Section 8.2"})
  [worker result] [v1_master] Master Service Agreement MSA-2026-014
  Section 4.1 - Payment Terms:
  The Client shall pay all undisputed invoices within thirty (30) calendar days from the invoice date. Payment shall be
  [worker iter 2] tokens: input=1015 output=88
  [worker tool] get_effective_date({"contract_version": "v1_master"})
  [worker result] Contract: Master Service Agreement MSA-2026-014
  Execution Date: 2025-12-15
  Effective Date:  2026-01-01
  Expiration Date: 2027-12-31
  [worker iter 3] tokens: input=1096 output=245
  [worker tool] get_effective_date({"contract_version": "v2_amendment_1"})
  [worker result] Contract: Amendment No. 1 to MSA-2026-014
  Execution Date: 2026-03-20
  Effective Date:  2026-04-01
  Expiration Date: 2027-12-31
  [worker iter 4] tokens: input=1184 output=461
  [worker tool] get_effective_date({"contract_version": "v3_amendment_2"})
  [worker result] Contract: Amendment No. 2 to MSA-2026-014
  Execution Date: 2026-06-25
  Effective Date:  2026-07-01
  Expiration Date: 2027-12-31
  [worker iter 5] tokens: input=1272 output=824
[clause_worker] END  tokens=6679
[clause_worker → orchestrator] tokens=6679  answer_len=1077

[orchestrator → defined_terms_worker] sending: Provide the definition of 'Business Day' in this contract and indicate whether it includes a holiday schedule.
[orchestrator] defined_terms_worker 500: 500 Internal Server Error — defined_terms_worker: simulated upstream failure (injected for test)
[orchestrator] RETRY defined_terms_worker (attempt 2/2) ...
[defined_terms_worker → orchestrator] RETRY OK  tokens=1180
[defined_terms_worker] START
  [worker iter 1] tokens: input=272 output=43
  [worker tool] get_definitions({"contract_version": "v1_master", "term_name": "Business Day"})
  [worker result] Term: Business Day
Version: v1_master
Definition: Any day other than a Saturday, Sunday, or public holiday observed in the jurisdiction specified in Section 15 (Governing Law). For the avoidance of do
  [worker iter 2] tokens: input=443 output=422
[defined_terms_worker] END  tokens=1180
[defined_terms_worker] START

[orchestrator] SYNTHESISE  qid=Q04
[orchestrator] synthesise tokens: input=662 output=710
```
