# Week 9 — Error Path Before/After

Same failing question run twice against `search_clause` in `week9/servers/clause_server.py` — once with the original terse error, once after rewriting it to be descriptive and recoverable.

**Question (both runs)**: What does the warranty disclaimer clause say?

## Before (terse error)

**Raw tool result from `search_clause`**: `Error: not found`

**Model's final answer to the user**:

> I’m looking for the exact text of the “Warranty Disclaimer” clause, but I’m not finding a clause with that name in the contract database. Could you let me know:

1. The exact name of the clause as it appears in the contract (e.g., “Warranty Disclaimer,” “Disclaimer of Warranties,” etc.)  
2. Which contract version you’re referring to (v1_master, v2_amendment_1, or v3_amendment_2).  

With that information I can pull the precise wording for you.

## After (recoverable error)

**Raw tool result from `search_clause`**: `No clause matching 'Warranty Disclaimer' was found in any version of MSA-2026-014. This tool matches by clause name or section number substring, not free-text description — try one of the exact clause names below instead:
  [v1_master] Master Service Agreement MSA-2026-014: ['Section 4.1 - Payment T`

**Model's final answer to the user**:

> I looked through all versions of the MSA‑2026‑014 contract (master and both amendments) for a clause titled or containing “Warranty Disclaimer” or “Warranty.”  The clause‑search tool returned:

*No clause matching ‘Warranty Disclaimer’ or ‘Warranty’ was found in any version of MSA‑2026‑014. The tool lists all existing clause headings for each version so you can see what’s actually there.*  

The available clauses are:

**v1_master (Master Service Agreement MSA‑2026‑014)**  
- Section 4.1 – Payment Terms  
- Section 4.2 – Late Payment Interest  
- Section 6 – Confidentiality  
- Section 7.3 – Force Majeure  
- Section 8.2 – Termination for Convenience  
- Section 11.4 – Intellectual Property  

**v2_amendment_1 (Amendment No. 1 to MSA‑2026‑014)**  
- Section 1 – Termination Notice Period  
- Section 3 – Payment Terms  

**v3_amendment_2 (Amendment No. 2 to MSA‑2026‑014)**  
- Section 1 – Termination Notice Period  
- Section 2 – Payment Terms Confirmation  

Because there is no Warranty or Warranty Disclaimer clause in any version, the contract does not provide a statement about warranties or a disclaimer thereof. If you need to confirm whether a warranty provision exists, you would need to check an earlier draft or another related agreement.

