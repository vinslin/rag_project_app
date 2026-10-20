# Evaluation Dataset
**Legal Contract Intelligence — Eval Suite Documentation**

---

## Dataset Overview

| Property | Value |
|----------|-------|
| Dataset name | Soft Suave AI League — Shared Legal Contract Golden Set |
| Version | 1.0 |
| Total cases | 32 |
| Corpus | 5 documents (MSA-2026-014, AMD-2026-014-01, AMD-2026-014-02, MSA-2026-022, NDA-2026-007, SCH-2026-014) |
| Purpose | Shared ground-truth set — all squad members run the same cases so results are directly comparable |
| File | `evals/golden_set.json` |

> **Do not edit questions or expected answers.** A changed set voids every cross-member comparison.
> Raise ground-truth disputes with the trainer.

---

## Supporting Labels File

A second dataset (`evals/labels_25.json`) contains 30 manually annotated cases from the earlier
single-contract test corpus (`hybrid_search_legal_contract_test.pdf`). These were used for
pipeline development and taxonomy validation before the shared golden set was created.

| Property | Value |
|----------|-------|
| Cases | 30 |
| Contracts | 4 (hybrid_search_legal_contract_test.pdf, Master_Service_Agreement.pdf, Vendor_Agreement.pdf, Chunking_Test_Amendment.pdf) |
| Taxonomy | retrieval_failure (16), generation_failure (7), amendment_override_failure (2), guardrail_trigger (5) |

These cases are **not** part of the official eval run. They informed the error taxonomy in the
weak spots report but are not included in the 32-case pass-rate numbers.

---

## Question Classes

The golden set uses 9 question classes designed to test distinct failure modes:

| Class | Count | Description |
|-------|-------|-------------|
| `direct_lookup` | 8 | Single clause, single document — the baseline |
| `amendment_supersession` | 7 | The original value was replaced by one or more amendments. Tests version precedence |
| `counterparty_disambiguation` | 4 | The same section number exists in two agreements with different values. Tests metadata filtering |
| `out_of_scope` | 4 | The corpus cannot answer this. A correct system refuses |
| `defined_term_chase` | 3 | The answer requires resolving a defined term that points elsewhere in the documents |
| `multi_hop_dependent` | 2 | Step 3 depends on what step 2 returned. These are the cases that justify an agent |
| `computation` | 2 | Requires arithmetic or date computation on retrieved values |
| `version_comparison` | 1 | Requires comparing a value across document versions |
| `cross_document` | 1 | Requires facts from two different agreements |

---

## Corpus Documents

| Document ID | Type | Counterparty | Effective Date | Hops involved |
|-------------|------|-------------|----------------|---------------|
| MSA-2026-014 | Master Services Agreement | Northwind Logistics Pvt. Ltd. | 14 Jan 2026 | Base for 7 amendment_supersession cases |
| AMD-2026-014-01 | Amendment No. 1 | Northwind | 1 Apr 2026 | Supersedes MSA on payment terms, audit, cure period |
| AMD-2026-014-02 | Amendment No. 2 | Northwind | 1 Aug 2026 | Supersedes AMD-01 on liability cap, termination, data residency, confidentiality |
| SCH-2026-014 | Schedules (SLA, fees, defined terms) | Northwind | 14 Jan 2026 | Referenced by 6 cases |
| MSA-2026-022 | Master Services Agreement | Vertex Retail Solutions | 1 Mar 2026 | Used for counterparty_disambiguation cases |
| NDA-2026-007 | Non-Disclosure Agreement | Halcyon Materials GmbH | 1 Feb 2026 | 2 direct_lookup cases |

---

## Scoring Criteria

A case **PASS**es when the answer:
1. Contains the expected value (or values from `expected_values` list), AND
2. Cites the correct source document and clause number

A case **FAIL**s with one of four error categories:

| Error category | Meaning |
|----------------|---------|
| `fail_hallucination` | Answer states a value not supported by any cited source |
| `fail_wrong_version` | Answer gives a superseded value (e.g. original MSA cap after Amendment No.2 changed it) |
| `fail_wrong_counterparty` | Answer gives the correct section from the wrong agreement |
| `fail_false_refusal` | System refuses a question the corpus can answer |

---

## Judge

**Primary judge:** Groq LLM (model: `config.GENERATION_MODEL`) using prompt in `evals/judge_v1.txt`.
The judge returns `status` (PASS/FAIL), `score` (0.0–1.0), `reasoning`, and `error_category`.

**Fallback (deterministic):** Used when `GROQ_API_KEY` is unset. Checks:
- For `refusal_expected` cases: presence of refusal phrases in the predicted answer
- For all other cases: expected value substrings present in the predicted answer
- Applies Unicode NFKC normalisation to handle fancy punctuation from LLM output

**Judge agreement rate:** The judge's PASS/FAIL verdict was hand-verified against manual
labels on 7 failing cases. Agreement on those 7: **100%** (all judge failures match manual
assessment). The judge has not been validated on the 25 passing cases — it may be lenient.

---

## What This Eval Suite Does NOT Cover

These gaps are acknowledged explicitly and must not be read into the headline pass rate:

| Gap | Reason |
|----|---------|
| Indemnity clause interpretation | No indemnity-specific cases in the 32-case set |
| Scanned PDF extraction errors | All 32 cases use digitally-native PDFs |
| Amendment chains longer than 2 levels | Corpus has at most 2 amendments per base MSA |
| PII / privilege leakage via MCP | Operational risk, not eval-covered |
| Concurrent multi-user sessions | Single-user system tested only |
| Computation with multiple retrieved values | Only 2 computation cases; both require simple arithmetic |
| Output phrase "this is standard" / legal advice detection | Guardrail tested manually, not in the 32 cases |

---

## How to Run

```bash
# Use cached predictions (fast — re-runs judge only)
python evals/run_eval.py

# Regenerate all predictions from the live pipeline (slow — 3-min gaps between calls)
python evals/run_eval.py --recompute

# Use judge v2
python evals/run_eval.py --judge v2
```

Results are written to `evals/results_v1.json` (or `results_v2.json` for `--judge v2`).

---

## Rate-Limit Note

The eval runner inserts a **3-minute sleep** between prediction calls (`--recompute` mode)
to stay within the Groq free-tier rolling token window. A full 32-case recompute takes
approximately **1.5–2 hours**. Cached-prediction runs (judge only, with 10-second gaps
between judge calls) take approximately **6–8 minutes**.
