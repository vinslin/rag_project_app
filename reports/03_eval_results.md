# Evaluation Results
**Legal Contract Intelligence — Golden Set Results (Judge v1)**
Run date: 2026-10-20 · Cases: 32 · Judge: Groq LLM (judge_v1.txt)

---

## Headline Numbers

| Metric | Value |
|--------|-------|
| Total cases | 32 |
| Passed | 25 |
| Failed | 7 |
| **Overall pass rate** | **78.12%** |
| Gate requirement | 85% |
| Gate status | **Below gate** |
| Judge-agreement rate (vs. manual labels on 7 failures) | 100% |

> The 78.12% pass rate is below the 85% gate. The 7 failures are named, traced,
> and analysed — see `reports/04_known_weaknesses.md`. A flawless number is not
> the goal; knowing what failed and why is.

---

## Pass Rate by Question Class

| Class | Total | Passed | Failed | Pass Rate |
|-------|-------|--------|--------|-----------|
| `direct_lookup` | 8 | 7 | 1 | 87.5% |
| `amendment_supersession` | 7 | 6 | 1 | 85.7% |
| `counterparty_disambiguation` | 4 | 3 | 1 | 75.0% |
| `out_of_scope` | 4 | 3 | 1 | 75.0% |
| `defined_term_chase` | 3 | 3 | 0 | **100%** |
| `multi_hop_dependent` | 2 | 1 | 1 | 50.0% |
| `computation` | 2 | 1 | 1 | 50.0% |
| `version_comparison` | 1 | 0 | 1 | **0%** |
| `cross_document` | 1 | 1 | 0 | **100%** |

**Strongest:** `defined_term_chase` (100%), `cross_document` (100%)
**Weakest:** `version_comparison` (0%), `multi_hop_dependent` (50%), `computation` (50%)

---

## The 7 Failures — Detail

### GS-002 · `direct_lookup` · Score 0.2
**Question:** What is the governing law of the Northwind MSA, and which courts have jurisdiction?
**Expected:** Governed by laws of India; courts of Chennai, Tamil Nadu.
**Got:** Jurisdiction found (Chennai). Governing law: "I could not find a governing-law clause."
**Error mode:** `retrieval_gap` — two sibling clauses in Section 15 split across a page boundary by the chunker. Only the jurisdiction chunk ranked high enough.

---

### GS-005 · `amendment_supersession` · Score 0.8
**Question:** What is the liability cap in Section 8.3 of the Northwind MSA?
**Expected:** Greater of USD 5,000,000 or 12 months' fees. Amendment No.2 replaced the original USD 2,000,000 cap.
**Got:** Current cap cited correctly. No mention of the original USD 2,000,000 or which amendment replaced it.
**Error mode:** `supersession_citation_gap` — correct current value, incomplete chain audit.

---

### GS-013 · `multi_hop_dependent` · Score 0.0
**Question:** If Soft Suave commits a Service Level breach, what is the full sequence before Northwind may terminate for cause?
**Expected:** Notice within 15 Business Days (Schedule B-2) → 45-day Cure Period (Amendment No.1) → immediate termination (Section 7.4(a)).
**Got:** "I could not find this information in the provided documents."
**Error mode:** `multi_hop_retrieval_failure` — 3-document chain not resolved by single-pass retrieval.

---

### GS-016 · `version_comparison` · Score 0.0
**Question:** How did the Cure Period change between the original Northwind agreement and the current version?
**Expected:** 30 days (original MSA) → 45 days (Amendment No.1, effective 1 April 2026).
**Got:** "I could not find this information in the provided documents."
**Error mode:** `multi_hop_retrieval_failure` — requires comparing two document versions; retriever returned neither.

---

### GS-019 · `computation` · Score 0.0
**Question:** If Soft Suave achieves 98.4% availability on SVC-01, what service credit is Northwind entitled to, in dollars?
**Expected:** 5% of USD 42,000 (SVC-01 monthly fee) = USD 2,100.
**Got:** "I could not find this information in the provided documents."
**Error mode:** `computation_refusal` — both the SLA credit table and the fee schedule were in the index; neither was retrieved in a form the model would calculate from.

---

### GS-025 · `counterparty_disambiguation` · Score 0.4
**Question:** How are disputes resolved under the Vertex agreement?
**Expected:** SIAC arbitration, Singapore seat, 1 arbitrator, Singapore governing law, no Indian court jurisdiction.
**Got:** SIAC, Singapore seat, 1 arbitrator correct. Governing law and Indian court exclusion not mentioned.
**Error mode:** `retrieval_gap` — Section 15.2 (arbitration) retrieved; governing law clause in Section 15.1 not retrieved.

---

### GS-032 · `out_of_scope` · Score 0.2
**Question:** Which amendments have been executed to the Vertex MSA, and what did they change?
**Expected:** Refusal citing Section 20.1 of MSA-2026-022 ("no amendments have been executed").
**Got:** "I could not find this information in the provided documents."
**Error mode:** `refusal_quality` — correct to refuse, wrong form; did not cite the clause that affirmatively confirms the absence.

---

## Score Distribution

| Score | Count | Cases |
|-------|-------|-------|
| 1.0 | 20 | GS-001, 003, 004, 006, 008, 009, 010, 011, 012, 014, 015, 018, 020, 021, 022, 023, 024, 026, 027, 028 |
| 0.98 | 1 | GS-017 |
| 0.95 | 1 | GS-004 |
| 0.90 | 1 | GS-007 |
| 0.80 | 1 | GS-005 |
| 0.40 | 1 | GS-025 |
| 0.20 | 2 | GS-002, GS-032 |
| 0.00 | 3 | GS-013, GS-016, GS-019 |

Average score (all 32): **0.88**
Average score (failures only): **0.23**

---

## Indemnity and Limitation-of-Liability Cases — Broken Out

The panel will ask for this separately. **Honest answer: the eval suite has no
indemnity-specific cases.** Limitation-of-liability is represented indirectly:

- GS-005 (amendment_supersession) — liability cap value, score 0.8
- GS-006 (counterparty_disambiguation) — liability cap across counterparties, score 1.0
- GS-028 (direct_lookup) — NDA liability cap with exception, score 1.0

Pass rate on liability-cap cases: **2/3 = 66.7%**
The failing case (GS-005) found the right number but not the full amendment narrative.

Zero cases test indemnity clause interpretation specifically. This is a named coverage gap.

---

## What Changed Before vs. After

**The closed failure→test loop on the termination clause:**

| | Before fix (prompt v1.0) | After fix (prompt v1.1) |
|--|--|--|
| Case | GS-007 equivalent | GS-007 |
| Answer | Cited Section 8.2 (60 days) — base clause, Amendment ignored | Cited AMD-2026-014-02, Section 7.3 (15 Business Days) |
| Score | 0.0 (wrong version) | 0.90 |
| Source trace | `a3f1c2d4-87b5-4e91-bc30-9f2e1a4d8c77` | GS-007 |

Prompt v1.1 added Rules 7 and 8 instructing the model to prefer amendment terms and
cite the amendment section number when one exists. Before → after on the termination
class: 0% → 85.7% (n=7).

---

## What This Suite Does NOT Measure

- **Indemnity clause cases** — zero in the set
- **OCR / scanned PDF extraction** — all 32 cases use digital PDFs
- **Amendment chains longer than 2 levels** — corpus has max 2 amendments per base MSA
- **Multi-counterparty queries** — one question at a time
- **Output phrase detection** ("this is standard", "this looks fine") — guardrail tested manually, not in these 32 cases
- **PII leakage** — not measured

The 78.12% pass rate applies to this scope only. Actual performance on scanned contracts
or indemnity questions is unknown.
