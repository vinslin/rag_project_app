"""Fixed workflow — three hard-coded steps, same tools, same model,
same output contract.  No loop hiding inside it.

Steps (always in this order, no branching):
  1.  search_clause      → find the relevant clause text
  2.  get_definitions    → resolve defined terms found in the clause/question
  3.  get_effective_date → get temporal data for the latest version
  4.  Single LLM call    → generate the answer from all gathered context
"""

import os
import time

from google import genai
from dotenv import load_dotenv

from rag import config
from week7.tools import search_clause, get_effective_date, get_definitions
from week7.utils import gemini_call_with_retry

load_dotenv()

MODEL = config.GENERATION_MODEL

INPUT_PRICE_PER_M  = 0.075
OUTPUT_PRICE_PER_M = 0.30

SYSTEM_PROMPT = (
    "You are a legal contract analysis assistant.  Answer the user's "
    "question using ONLY the provided context gathered from the contract "
    "tools.\n\n"
    "Rules:\n"
    "1. Use only the information provided below.\n"
    "2. Cite the specific version, section, and clause.\n"
    "3. If the information is insufficient, say so.\n"
    "4. Be precise and concise."
)

# ── Keyword maps for the fixed routing (no LLM decision) ────────────────

_CLAUSE_KEYWORDS = {
    "termination":           "termination",
    "payment":               "payment",
    "confidential":          "confidentiality",
    "force majeure":         "force majeure",
    "intellectual property": "intellectual property",
    "ip ownership":          "intellectual property",
    "interest":              "late payment",
    "late payment":          "late payment",
}

_TERM_KEYWORDS = {
    "business day":             "Business Day",
    "confidential information": "Confidential Information",
    "service period":           "Service Period",
    "material breach":          "Material Breach",
    "effective date":           "Effective Date",
}


def _extract_clause_keyword(question: str) -> str:
    q = question.lower()
    for kw, clause in _CLAUSE_KEYWORDS.items():
        if kw in q:
            return clause
    return "termination"                       # safe default


def _extract_terms(question: str, clause_result: str = "") -> list[str]:
    combined = (question + " " + clause_result).lower()
    found = [term for kw, term in _TERM_KEYWORDS.items() if kw in combined]
    return found if found else ["Business Day"]  # always resolve at least one


# ── Fixed workflow ───────────────────────────────────────────────────────

def run_workflow(question: str) -> dict:
    """Run the fixed three-step workflow.

    Returns the same output contract as ``run_agent`` so the race
    compares like-for-like.
    """
    start = time.time()

    # ── Step 1: search_clause ────────────────────────────────────────────
    clause_kw    = _extract_clause_keyword(question)
    clause_text  = search_clause(clause_kw)

    # ── Step 2: get_definitions ──────────────────────────────────────────
    terms        = _extract_terms(question, clause_text)
    def_parts    = []
    for term in terms[:2]:
        def_parts.append(get_definitions(term, "v1_master"))
    definitions_text = "\n\n".join(def_parts)

    # ── Step 3: get_effective_date ───────────────────────────────────────
    date_text    = get_effective_date("v3_amendment_2")

    # ── Step 4: single LLM call ─────────────────────────────────────────
    context = (
        f"CLAUSE SEARCH RESULTS:\n{clause_text}\n\n"
        f"DEFINED TERMS:\n{definitions_text}\n\n"
        f"EFFECTIVE DATES (Latest Version):\n{date_text}"
    )

    prompt = f"{SYSTEM_PROMPT}\n\nGATHERED CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}"

    client   = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    response = gemini_call_with_retry(client, model=MODEL, contents=prompt)

    elapsed = time.time() - start

    # Token bookkeeping
    i_tok = o_tok = 0
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        i_tok = response.usage_metadata.prompt_token_count or 0
        o_tok = response.usage_metadata.candidates_token_count or 0

    total_tok = i_tok + o_tok
    cost = (i_tok * INPUT_PRICE_PER_M / 1_000_000
            + o_tok * OUTPUT_PRICE_PER_M / 1_000_000)

    return {
        "answer":              response.text,
        "tools_called": [
            {"name": "search_clause",
             "args": {"clause_name": clause_kw}},
            {"name": "get_definitions",
             "args": {"terms": terms[:2],
                      "contract_version": "v1_master"}},
            {"name": "get_effective_date",
             "args": {"contract_version": "v3_amendment_2"}},
        ],
        "iterations":          1,               # always exactly 1 LLM call
        "total_tokens":        total_tok,
        "total_input_tokens":  i_tok,
        "total_output_tokens": o_tok,
        "latency_s":           round(elapsed, 3),
        "cost_usd":            round(cost, 6),
        "budget_termination":  None,
        "log": [
            f"Step 1: search_clause('{clause_kw}')",
            f"Step 2: get_definitions({terms[:2]}, 'v1_master')",
            f"Step 3: get_effective_date('v3_amendment_2')",
            f"Step 4: LLM generation ({MODEL})",
        ],
    }
