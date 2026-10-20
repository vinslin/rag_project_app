# Known Weaknesses — Error Taxonomy
**Legal Contract Intelligence — Derived from Eval Results**
Written: 2026-10-20, before the demo, before anyone asked.

Source: 7 failures from `evals/results_v1.json` (32-case golden set, judge v1).
Overall pass rate: **78.12%** — below the 85% gate.

---

## Error Mode Summary

| # | Mode | Failures | % of Failures | % of Total Cases | Severity |
|---|------|----------|---------------|-----------------|----------|
| 1 | `multi_hop_retrieval_failure` | 2 | 29% | 6.3% | High |
| 2 | `retrieval_gap` | 2 | 29% | 6.3% | High |
| 3 | `supersession_citation_gap` | 1 | 14% | 3.1% | Medium |
| 4 | `computation_refusal` | 1 | 14% | 3.1% | Medium |
| 5 | `refusal_quality` | 1 | 14% | 3.1% | Low–Medium |
| 6 | `integration_context_overflow` ⚑ | observed | not in eval | unknown | High |
| 7 | `ocr_extraction_noise` | not measured | not in eval | unknown | High |

⚑ = new mode created by integration, not inherited from any individual component.

---

## Mode 1 — `multi_hop_retrieval_failure`

**Cases:** GS-013, GS-016
**Frequency:** 2 / 7 failures · 6.3% of all 32 cases
**Severity:** High — returns null on questions with a definitive answer
**Trace IDs:** GS-013 · GS-016

**What happens:**
Questions requiring three or more documents in sequence (MSA → Amendment No.1 →
Amendment No.2 → Schedule) return "I could not find this information." The retriever
surfaces chunks from one or two documents but the third is either not retrieved or
ranked below the context window limit.

**GS-013 example:**
```
Q: Full sequence before Northwind may terminate for cause after a Service Level breach?
Expected: Notice within 15 Business Days (Schedule B-2) → 45-day Cure Period
          (Amendment No.1, Section 1.4) → immediate termination (Section 7.4(a))
Got:      "I could not find this information in the provided documents."
Score:    0.0
```

**GS-016 example:**
```
Q: How did the Cure Period change between original and current Northwind agreement?
Expected: 30 days (MSA-2026-014) → 45 days (AMD-2026-014-01, effective 1 Apr 2026)
Got:      "I could not find this information in the provided documents."
Score:    0.0
```

**Root cause:**
Single-pass hybrid retrieval does not issue follow-up queries when a retrieved chunk
references a section in a second document. The agent has no cross-document resolution
loop.

**Proposed fix:**
When a retrieved chunk contains a cross-document reference ("as defined in Schedule B",
"as amended by Amendment No.1, Section…"), issue a second targeted retrieval for that
document before passing context to the generator.

---

## Mode 2 — `retrieval_gap`

**Cases:** GS-002, GS-025
**Frequency:** 2 / 7 failures · 6.3% of all 32 cases
**Severity:** High — partial answer looks complete; dangerous in legal context
**Trace IDs:** GS-002 · GS-025

**What happens:**
The retriever finds one sub-clause in a section and misses the sibling sub-clause
in the same section. The answer is factually correct for what it found, and wrong
by omission. A partial answer that looks confident is more dangerous than a null
response.

**GS-002 example:**
```
Q: What is the governing law and which courts have jurisdiction?
Expected: India (Section 15.1) + Chennai, Tamil Nadu (Section 15.2)
Got:      Chennai jurisdiction found. Governing law: "I could not find a governing-law clause."
Score:    0.2
```
Section 15.1 and 15.2 are adjacent clauses split across a page boundary by the token
chunker. The retriever ranked the jurisdiction chunk (Section 15.2) first because
the query contained "courts" — the governing-law chunk (Section 15.1) ranked below
the retrieval cutoff.

**GS-025 example:**
```
Q: How are disputes resolved under the Vertex agreement?
Expected: SIAC arbitration + Singapore governing law + no Indian court jurisdiction
Got:      SIAC, Singapore seat, 1 arbitrator correct. Governing law and Indian court
          exclusion clause missing.
Score:    0.4
```

**Root cause:**
Token-level page-boundary chunking splits logically adjacent clauses. The reranker
rewards the chunk with higher keyword overlap, not the set of chunks needed for
completeness.

**Proposed fix:**
Semantic section chunking — keep all sub-clauses of a numbered parent section together
(e.g., all of Section 15 in one chunk), then split at the section level rather than
the page boundary.

---

## Mode 3 — `supersession_citation_gap`

**Cases:** GS-005
**Frequency:** 1 / 7 failures · 3.1% of all 32 cases
**Severity:** Medium — correct for lookup; fails the audit trail requirement
**Trace ID:** GS-005

**What happens:**
The system correctly retrieves and cites the post-amendment clause value but never
states what the original clause said or which amendment superseded it. A lawyer
cannot tell from the output whether the amendment chain was resolved or the base
MSA was quoted.

**GS-005 example:**
```
Q: What is the liability cap in Section 8.3?
Expected: USD 5,000,000 or 12 months' fees (greater). Amendment No.2 replaced
          the original USD 2,000,000 cap.
Got:      Correct cap cited. No mention of original USD 2,000,000 or Amendment No.2.
Score:    0.8
```

**Root cause:**
Prompt v1.x instructs the model to prefer amendments but does not require it to
state the superseded value. The model stops at the first correct answer.

**Proposed fix:**
Add explicit prompt instruction: "When citing an amended clause, always state
(a) the current operative text, (b) which amendment superseded it and when,
and (c) the value that was superseded."

---

## Mode 4 — `computation_refusal`

**Cases:** GS-019
**Frequency:** 1 / 7 failures · 3.1% of all 32 cases
**Severity:** Medium — deterministic answer returned as null
**Trace ID:** GS-019

**What happens:**
Questions requiring arithmetic over retrieved values return "I could not find this
information" instead of performing the calculation. Both source chunks (the SLA
credit table and the monthly fee schedule) were in the index.

**GS-019 example:**
```
Q: If Soft Suave achieves 98.4% availability on SVC-01, what service credit is owed?
Expected: 5% of USD 42,000 = USD 2,100
          (SLA: <99.0% triggers 5% credit; SVC-01 monthly fee: USD 42,000)
Got:      "I could not find this information in the provided documents."
Score:    0.0
```

**Root cause:**
Two separate retrieval calls would be needed (SLA schedule + fee schedule). The
single-pass retriever returned neither in a form the model would calculate from. Even
if both were retrieved, the prompt does not grant the model permission to do arithmetic.

**Proposed fix:**
(1) Expand retrieval to fetch both the SLA schedule and the fee schedule when the
query contains "credit", "service level", or "availability". (2) Add a prompt
instruction explicitly permitting step-by-step arithmetic on retrieved numeric values.

---

## Mode 5 — `refusal_quality`

**Cases:** GS-032
**Frequency:** 1 / 7 failures · 3.1% of all 32 cases
**Severity:** Low–Medium — correct to refuse, wrong form
**Trace ID:** GS-032

**What happens:**
In-corpus absence questions (e.g. "which amendments were executed?", where the answer
is "none" as stated in Section 20.1) return a generic "I could not find this
information" rather than citing the specific clause that confirms the absence.

**GS-032 example:**
```
Q: Which amendments have been executed to the Vertex MSA, and what did they change?
Expected: Refusal citing MSA-2026-022, Section 20.1: "no amendments have been executed"
Got:      "I could not find this information in the provided documents."
Score:    0.2
```
The section title is "Entire Agreement / No Amendments" — the word "amendments"
appears in the query but the section's heading does not match the query well enough
to rank it high.

**Root cause:**
The refusal path handles unknown counterparties but not in-corpus confirmed absences.
The model falls to a generic null response instead of retrieving the clause that
affirmatively states the absence.

**Proposed fix:**
Add an eval assertion that in-corpus refusals must cite a clause, not just return a
null string. A specific prompt instruction: "When stating that something does not
exist in the corpus, cite the clause or section that confirms its absence."

---

## Mode 6 — `integration_context_overflow` ⚑ *(new — created by integration)*

**Cases:** Not isolated in the 32-case eval. Observed manually on `full_doc_mcp` and `both` routes.
**Frequency:** Estimated 1–2 of every 5 full_doc_mcp queries exceed effective context
**Severity:** High — silently degrades answer quality with no error raised
**Trace:** Any production trace in `data/logs/app.jsonl` where `route = full_doc_mcp` and `tokens.input > 55000`

**What happens:**
In the isolated pipeline, RAG was tested without MCP tool schemas in context.
In the isolated MCP loop, document chunks were not in context. Only the integrated
system combines them. On a `full_doc_mcp` route: MCP tool schemas (~800–1,200 tokens
each × multiple tools) + system prompt + document text can exceed the model's
effective context window. The model answers from whatever fits, silently dropping
the tail of the document. No error is raised; the answer looks complete.

**This mode did not exist in any individual component.** It is produced only by integration.

**Proposed fix:**
(1) Enforce a hard token budget per stage before assembling the prompt.
(2) Strip MCP tool schema descriptions to names + required parameters only when
context is under pressure.
(3) Force the section-by-section analysis path (already implemented in `router/agent.py`)
for all documents over 50,000 tokens rather than 60,000.

---

## Mode 7 — `ocr_extraction_noise`

**Cases:** Not in the 32-case golden set (all cases use digitally-native PDFs)
**Frequency:** Not measured. Estimated 3–8% character error rate on scanned pages
**Severity:** High — garbled values enter the index silently; cited with confidence
**Trace:** None from current eval; would appear in production on scanned contract queries

**What happens:**
Scanned PDFs processed through PyMuPDF produce OCR artefacts: clause numbers
garbled ("11.3" → "l1.3"), numeric values corrupted ("USD 2,000,000" → "USD 2.000.000"),
defined terms split across lines. The index stores the garbled text. The model cites it
with full confidence.

**Current eval coverage:** Zero. All 32 cases use clean digital PDFs.

**Proposed fix:**
(1) Post-extraction validation: check that numeric patterns and defined terms from
a known reference list are present verbatim in the extracted text.
(2) Flag low-confidence extractions with a visible warning in the citation metadata.
(3) Run a pre-indexing OCR quality check and quarantine suspect pages.

---

## Top 3 Known Weaknesses

Written 2026-10-20, before the demo, before anyone asked.

**1. Amendment chains longer than two hops are not resolved.**
The current retriever makes one pass. Any contract with three or more amendments
to the same clause will surface the wrong version if the latest amendment's chunk
ranks below the context cutoff. This is the single highest-risk failure mode in
a legal context — the answer will be confidently wrong and citable.

**2. The eval suite has zero scanned-PDF coverage.**
All 32 golden-set cases use digitally-native PDFs. The real corpus includes scanned
contracts and faxed amendments. OCR extraction errors are a named failure mode with
zero test coverage — the 78.12% pass rate does not apply to scanned documents.

**3. Computation questions are under-represented.**
Only 2 of 32 cases involve arithmetic. The suite does not test pro-rata fee
calculations, penalty accumulation across multiple SLA breaches, or compounded
interest. These are the numbers lawyers actually dispute in redline negotiations.

---

## Closed Failure → Test Loop (demonstrated)

**Failure found:** Trace `a3f1c2d4-87b5-4e91-bc30-9f2e1a4d8c77`
System cited Section 8.2 (60 days' notice — base MSA) while Amendment No.2
(15 Business Days) was already retrieved at rank 3 and ignored.

**Fix applied:** Prompt v1.1 added Rules 7 and 8:
> Rule 7 — When an amendment clause and a base clause cover the same section,
> always cite the amendment clause and note that it supersedes the base clause.
> Rule 8 — Always include the amendment document ID, section number, and
> effective date in the citation.

**Test added:** GS-007 (`amendment_supersession`) now permanently guards this failure.

**Before → after:**
- Before (prompt v1.0): termination class 0 / 7 = 0%
- After (prompt v1.1): termination class 6 / 7 = 85.7%

**Cost of this failure in production:** A salesperson told a counterparty they could
exit the contract with 60 days' notice. The operative notice period was 15 Business Days.
The difference is 6–7 weeks of contract exposure on a USD 973,200/year account.
