"""Fix for the "missing tool chain" trajectory gap found in
week8/trajectory_eval.py: the agent sometimes answers from search_clause
text alone without resolving a defined term it references.

Two additive changes on top of week7.agent.run_agent (implemented as a
standalone loop here, not an edit to week7/agent.py, so Week 7's graded
behavior stays intact):

  1. Prompt strengthening — an explicit, checkable MUST-rule about calling
     get_definitions whenever retrieved clause text contains a known
     defined term.
  2. Deterministic guard — after the model produces what it thinks is a
     final answer, scan the clause text gathered via search_clause for any
     of the corpus's known defined terms. If one appears but
     get_definitions was never called for it, force one corrective
     iteration instead of accepting the answer.
"""

import json
import time

from rag import config
from week7.contracts import DEFINED_TERMS
from week7.tools import TOOL_DECLARATIONS, TOOL_FUNCTIONS
from week7.llm_client import get_client, groq_call_with_retry, to_groq_tools

MODEL = config.GENERATION_MODEL

# Union of defined-term names across all contract versions in this corpus.
KNOWN_DEFINED_TERMS = sorted({
    term for version_terms in DEFINED_TERMS.values() for term in version_terms
})

FIXED_SYSTEM_PROMPT = (
    "You are a legal contract analysis agent.  Answer the user's question "
    "about the Master Service Agreement MSA-2026-014 and its amendments.\n\n"
    "You have three tools:\n"
    "  1. search_clause — find clause text by name/section number\n"
    "  2. get_effective_date — get dates for a contract version\n"
    "  3. get_definitions — resolve a defined term, following schedule refs\n\n"
    "MUST-RULE: If any clause text you retrieve contains one of this "
    "contract's capitalized defined terms (for example 'Business Day', "
    "'Service Period', 'Confidential Information', 'Material Breach', "
    "'Effective Date'), you MUST call get_definitions for that exact term "
    "before giving your final answer, even if you think you already know "
    "what it means. Do not answer from the clause text alone when it "
    "references a defined term.\n\n"
    "If you need dates, call get_effective_date with the specific version "
    "enum. Always cite which contract version and section your answer "
    "comes from."
)


def _dispatch_tool(name: str, args: dict) -> str:
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return f"Unknown tool: {name}"
    try:
        return func(**args)
    except Exception as exc:
        return f"Tool error: {exc}"


def _find_unresolved_terms(tools_called: list[dict]) -> list[str]:
    """Defined terms that appear in search_clause results but were never
    passed to get_definitions."""
    clause_text = " ".join(
        t["result_preview"] for t in tools_called if t["name"] == "search_clause"
    )
    resolved = {
        t["args"].get("term_name") for t in tools_called if t["name"] == "get_definitions"
    }
    return [
        term for term in KNOWN_DEFINED_TERMS
        if term in clause_text and term not in resolved
    ]


_TOOLS = to_groq_tools(TOOL_DECLARATIONS)


def run_agent_fixed(question: str, *, max_iters: int = 8) -> dict:
    """Same return contract as week7.agent.run_agent, plus a
    'forced_corrections' count showing how many times the guard fired."""
    client = get_client()

    messages = [
        {"role": "system", "content": FIXED_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tools_called: list[dict] = []
    log: list[str] = []
    final_answer = None
    iteration = 0
    forced_corrections = 0
    start = time.time()

    while iteration < max_iters:
        iteration += 1
        response = groq_call_with_retry(
            client, model=MODEL, messages=messages, tools=_TOOLS, log=log,
        )
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
                result = _dispatch_tool(tc.function.name, args)
                tools_called.append({"name": tc.function.name, "args": args, "result_preview": result[:300]})
                log.append(f"Tool call: {tc.function.name}({json.dumps(args)})")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            continue

        # Model produced a text answer — run the deterministic guard before accepting it.
        candidate_answer = msg.content
        unresolved = _find_unresolved_terms(tools_called)

        if unresolved and iteration < max_iters:
            forced_corrections += 1
            log.append(f"[GUARD] Unresolved defined term(s) {unresolved} found in clause "
                       f"text but never resolved via get_definitions. Forcing correction.")
            messages.append({"role": "assistant", "content": candidate_answer})
            messages.append({
                "role": "user",
                "content": (
                    f"Before finalizing, you must call get_definitions for: "
                    f"{', '.join(unresolved)}. Do this now, then give your final answer."
                ),
            })
            continue

        final_answer = candidate_answer
        break

    elapsed = time.time() - start
    return {
        "answer": final_answer or "[no answer produced within max_iters]",
        "tools_called": tools_called,
        "iterations": iteration,
        "forced_corrections": forced_corrections,
        "latency_s": round(elapsed, 3),
        "log": log,
    }
