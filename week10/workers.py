"""Week 10 — Narrow workers for the multi-agent orchestrator.

Two workers with deliberately constrained tool sets:

  clause_worker       — search_clause + get_effective_date ONLY
                        (no get_definitions — keeps the narrow-prompt-fewer-tools
                        constraint that is the only plausible source of a win over
                        the single agent)

  defined_terms_worker — get_definitions ONLY
                        (cannot retrieve clauses or dates — single responsibility)

Each worker runs its own mini tool-calling loop, returns a structured dict that
includes the LLM answer AND the token counts consumed, so the orchestrator can
log every hand-off token count for the context re-send multiplier.

Failure injection
-----------------
Set INJECT_FAILURE_CASE = <question_id> before calling defined_terms_worker and it
will raise a simulated 500 HTTPError on that call (once — subsequent calls succeed
so the orchestrator's retry path is exercised).
"""

import json
import time
from typing import Optional

from week7.contracts import CONTRACTS, DEFINED_TERMS, SCHEDULES
from week7.llm_client import get_client, groq_call_with_retry, to_groq_tools
from rag import config as rag_config

MODEL = rag_config.GENERATION_MODEL

# ---------------------------------------------------------------------------
# Pricing (same constants as week7/agent.py)
# ---------------------------------------------------------------------------
INPUT_PRICE_PER_M  = 0.10   # $/1 M input tokens
OUTPUT_PRICE_PER_M = 0.50   # $/1 M output tokens


def _estimate_cost(inp: int, out: int) -> float:
    return inp * INPUT_PRICE_PER_M / 1_000_000 + out * OUTPUT_PRICE_PER_M / 1_000_000


# ---------------------------------------------------------------------------
# Failure injection state
# ---------------------------------------------------------------------------
# Set to a question_id string to make defined_terms_worker raise a 500 once.
INJECT_FAILURE_CASE: Optional[str] = None
_failure_already_fired: set = set()


def _maybe_inject_500(question_id: Optional[str]) -> None:
    """Raise a simulated 500 if this question_id matches the injection target
    and has not already fired for this question_id."""
    if question_id and INJECT_FAILURE_CASE == question_id:
        if question_id not in _failure_already_fired:
            _failure_already_fired.add(question_id)
            raise RuntimeError(
                "500 Internal Server Error — defined_terms_worker: "
                "simulated upstream failure (injected for test)"
            )


def reset_failure_state() -> None:
    """Call between questions to reset the one-shot failure flag."""
    _failure_already_fired.clear()


# ---------------------------------------------------------------------------
# Tool implementations (deterministic Python — no LLM inside the workers'
# tool layer; only the worker's *reasoning* loop calls the LLM)
# ---------------------------------------------------------------------------

def _search_clause(clause_name: str) -> str:
    results = []
    query = clause_name.lower().strip()
    for version_key, contract in CONTRACTS.items():
        for clause_key, clause_text in contract["clauses"].items():
            if (query in clause_key.lower()
                    or any(w in clause_key.lower()
                           for w in query.split() if len(w) > 3)):
                results.append({
                    "version": version_key,
                    "contract": contract["name"],
                    "clause": clause_key,
                    "text": clause_text,
                })
    if not results:
        lines = [f"No clause matching '{clause_name}' found. Available clauses:"]
        for vk, ct in CONTRACTS.items():
            lines.append(f"  [{vk}] {ct['name']}: {list(ct['clauses'].keys())}")
        return "\n".join(lines)
    parts = []
    for r in results:
        parts.append(
            f"[{r['version']}] {r['contract']}\n"
            f"  {r['clause']}:\n"
            f"  {r['text']}"
        )
    return "\n\n".join(parts)


def _get_effective_date(contract_version: str) -> str:
    if contract_version not in CONTRACTS:
        return f"Unknown contract version: {contract_version}. Valid: {list(CONTRACTS.keys())}"
    c = CONTRACTS[contract_version]
    return (
        f"Contract: {c['name']}\n"
        f"  Execution Date: {c['execution_date']}\n"
        f"  Effective Date:  {c['effective_date']}\n"
        f"  Expiration Date: {c['expiration_date']}"
    )


def _get_definitions(term_name: str, contract_version: str) -> str:
    if contract_version not in DEFINED_TERMS:
        return f"No definitions found for version: {contract_version}"
    terms = DEFINED_TERMS[contract_version]
    query = term_name.lower().strip()
    matched_term = matched_definition = None
    for term_key, definition in terms.items():
        if query in term_key.lower() or term_key.lower() in query:
            matched_term = term_key
            matched_definition = definition
            break
    if not matched_definition:
        return (f"Term '{term_name}' not found in {contract_version} definitions. "
                f"Available terms: {list(terms.keys())}")
    result = (f"Term: {matched_term}\n"
              f"Version: {contract_version}\n"
              f"Definition: {matched_definition}")
    for schedule_name, schedule_text in SCHEDULES.items():
        if schedule_name.lower() in matched_definition.lower():
            result += f"\n\nReferenced {schedule_name}:\n  {schedule_text}"
    return result


# ---------------------------------------------------------------------------
# Narrow tool declarations (Gemini-style, converted via to_groq_tools)
# ---------------------------------------------------------------------------

_CLAUSE_TOOL_DECLARATIONS = [
    {
        "name": "search_clause",
        "description": (
            "Locate a specific clause by name or section number and return "
            "its full text. Searches across all contract versions (master + amendments)."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "clause_name": {
                    "type": "STRING",
                    "description": "Clause name or section number (e.g. 'Section 8.2', 'termination')",
                },
            },
            "required": ["clause_name"],
        },
    },
    {
        "name": "get_effective_date",
        "description": (
            "Return the effective date and temporal metadata for a specific contract version. "
            "Valid values: v1_master, v2_amendment_1, v3_amendment_2."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "contract_version": {
                    "type": "STRING",
                    "description": "Which contract version to check",
                    "enum": ["v1_master", "v2_amendment_1", "v3_amendment_2"],
                },
            },
            "required": ["contract_version"],
        },
    },
]

_DEFTERM_TOOL_DECLARATIONS = [
    {
        "name": "get_definitions",
        "description": (
            "Resolve a defined contractual term to its full definition, following "
            "any references to schedules. Use this to find what a defined term means."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "term_name": {
                    "type": "STRING",
                    "description": "The defined term to look up (e.g. 'Business Day', 'Service Period')",
                },
                "contract_version": {
                    "type": "STRING",
                    "description": "Which contract version's definitions to check",
                    "enum": ["v1_master", "v2_amendment_1", "v3_amendment_2"],
                },
            },
            "required": ["term_name", "contract_version"],
        },
    },
]

_CLAUSE_TOOLS  = to_groq_tools(_CLAUSE_TOOL_DECLARATIONS)
_DEFTERM_TOOLS = to_groq_tools(_DEFTERM_TOOL_DECLARATIONS)

_CLAUSE_TOOL_FUNCTIONS  = {"search_clause": _search_clause, "get_effective_date": _get_effective_date}
_DEFTERM_TOOL_FUNCTIONS = {"get_definitions": _get_definitions}


# ---------------------------------------------------------------------------
# Generic mini tool-calling loop (reused by both workers)
# ---------------------------------------------------------------------------

def _run_worker_loop(
    system_prompt: str,
    user_message: str,
    tools: list,
    tool_functions: dict,
    max_iters: int = 5,
    log: Optional[list] = None,
) -> dict:
    """Run a compact tool-calling loop and return answer + token counts."""
    if log is None:
        log = []

    client = get_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]

    total_input_tokens  = 0
    total_output_tokens = 0
    tools_called: list[dict] = []
    final_answer = None
    start = time.time()

    for iteration in range(1, max_iters + 1):
        response = groq_call_with_retry(
            client, model=MODEL, messages=messages, tools=tools, log=log
        )
        if getattr(response, "usage", None):
            i_tok = response.usage.prompt_tokens or 0
            o_tok = response.usage.completion_tokens or 0
            total_input_tokens  += i_tok
            total_output_tokens += o_tok
            log.append(f"  [worker iter {iteration}] tokens: input={i_tok} output={o_tok}")

        msg = response.choices[0].message
        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                log.append(f"  [worker tool] {tc.function.name}({json.dumps(args)})")
                fn = tool_functions.get(tc.function.name)
                result = fn(**args) if fn else f"Unknown tool: {tc.function.name}"
                tools_called.append({"name": tc.function.name, "args": args,
                                     "result_preview": result[:200]})
                log.append(f"  [worker result] {result[:200]}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
            final_answer = msg.content
            break

    elapsed = time.time() - start
    return {
        "answer":               final_answer or "[worker: no answer within max_iters]",
        "tools_called":         tools_called,
        "total_input_tokens":   total_input_tokens,
        "total_output_tokens":  total_output_tokens,
        "total_tokens":         total_input_tokens + total_output_tokens,
        "latency_s":            round(elapsed, 3),
        "cost_usd":             round(_estimate_cost(total_input_tokens, total_output_tokens), 6),
        "log":                  log,
    }


# ---------------------------------------------------------------------------
# Public worker entry points
# ---------------------------------------------------------------------------

CLAUSE_SYSTEM_PROMPT = (
    "You are the Clause-Retrieval Worker for a legal contract analysis system. "
    "Your ONLY job is to find and return relevant clause text and effective dates. "
    "You have two tools: search_clause and get_effective_date. "
    "Do NOT attempt to resolve defined terms — that is a different worker's job. "
    "Return the raw clause text with version and section citations. "
    "Be concise and precise."
)

DEFTERM_SYSTEM_PROMPT = (
    "You are the Defined-Terms Worker for a legal contract analysis system. "
    "Your ONLY job is to resolve defined terms and follow schedule references. "
    "You have one tool: get_definitions. "
    "Do NOT retrieve clause text or dates. "
    "Return the full definition including any referenced schedule text."
)


def clause_worker(question: str, log: Optional[list] = None) -> dict:
    """Clause-Retrieval Worker — search_clause + get_effective_date ONLY."""
    if log is None:
        log = []
    log.append("[clause_worker] START")
    result = _run_worker_loop(
        system_prompt=CLAUSE_SYSTEM_PROMPT,
        user_message=question,
        tools=_CLAUSE_TOOLS,
        tool_functions=_CLAUSE_TOOL_FUNCTIONS,
        log=log,
    )
    log.append(f"[clause_worker] END  tokens={result['total_tokens']}")
    return result


def defined_terms_worker(
    question: str,
    question_id: Optional[str] = None,
    log: Optional[list] = None,
) -> dict:
    """Defined-Terms Worker — get_definitions ONLY.

    Raises RuntimeError('500 ...') if INJECT_FAILURE_CASE matches question_id
    and has not already fired for this question_id.
    """
    if log is None:
        log = []
    log.append("[defined_terms_worker] START")
    _maybe_inject_500(question_id)   # raises on injected failure
    result = _run_worker_loop(
        system_prompt=DEFTERM_SYSTEM_PROMPT,
        user_message=question,
        tools=_DEFTERM_TOOLS,
        tool_functions=_DEFTERM_TOOL_FUNCTIONS,
        log=log,
    )
    log.append(f"[defined_terms_worker] END  tokens={result['total_tokens']}")
    return result
