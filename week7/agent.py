"""Agent loop with four enforced budgets for contract question answering.

Budgets enforced every iteration:
  MAX_ITERATIONS  — hard cap on tool-calling loop turns
  MAX_TOKENS      — cumulative input + output tokens across ALL turns
  MAX_COST        — estimated USD cost (input + output pricing)
  MAX_WALL_CLOCK  — wall-clock seconds

Each iteration re-sends the full message history to the model, so per-lap
tokens are summed — not just the final call.

Uses Groq (OpenAI-compatible chat completions + tool calling) as the LLM.
"""

import json
import time

from rag import config
from week7.tools import TOOL_DECLARATIONS, TOOL_FUNCTIONS
from week7.llm_client import get_client, groq_call_with_retry, to_groq_tools

# ── Budget defaults ──────────────────────────────────────────────────────────

MAX_ITERATIONS = 10
MAX_TOKENS     = 8_000
MAX_COST       = 0.05      # USD
MAX_WALL_CLOCK = 30.0      # seconds

# ── Pricing (Groq openai/gpt-oss-20b, approximate — same rate for both systems) ──

INPUT_PRICE_PER_M  = 0.10   # $/1 M input tokens
OUTPUT_PRICE_PER_M = 0.50   # $/1 M output tokens

MODEL = config.GENERATION_MODEL

SYSTEM_PROMPT = (
    "You are a legal contract analysis agent.  Answer the user's question "
    "about the Master Service Agreement MSA-2026-014 and its amendments.\n\n"
    "You have three tools:\n"
    "  1. search_clause — find clause text by name/section number\n"
    "  2. get_effective_date — get dates for a contract version\n"
    "  3. get_definitions — resolve a defined term, following schedule refs\n\n"
    "Call the appropriate tools to gather information, then provide a precise "
    "answer.  If a clause references a defined term (e.g. 'Business Day'), "
    "call get_definitions to resolve it.  If you need dates, call "
    "get_effective_date with the specific version enum.\n"
    "Always cite which contract version and section your answer comes from."
)

_TOOLS = to_groq_tools(TOOL_DECLARATIONS)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * INPUT_PRICE_PER_M / 1_000_000
            + output_tokens * OUTPUT_PRICE_PER_M / 1_000_000)


def _dispatch_tool(name: str, args: dict) -> str:
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return f"Unknown tool: {name}"
    try:
        return func(**args)
    except Exception as exc:
        return f"Tool error: {exc}"


# ── Agent loop ───────────────────────────────────────────────────────────────

def run_agent(
    question: str,
    *,
    max_iters: int = MAX_ITERATIONS,
    max_tokens: int = MAX_TOKENS,
    max_cost: float = MAX_COST,
    max_wall_clock: float = MAX_WALL_CLOCK,
    log: list | None = None,
) -> dict:
    """Run the agent loop and return a structured result.

    Returns
    -------
    dict
        answer, tools_called, iterations, total_tokens,
        total_input_tokens, total_output_tokens, latency_s,
        cost_usd, budget_termination, log
    """
    if log is None:
        log = []

    client = get_client()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    total_input_tokens  = 0
    total_output_tokens = 0
    tools_called: list[dict] = []
    iteration          = 0
    budget_termination = None
    final_answer       = None

    start = time.time()

    while iteration < max_iters:
        # ── Check wall-clock budget ──────────────────────────────────────
        elapsed = time.time() - start
        if elapsed >= max_wall_clock:
            budget_termination = (
                f"WALL_CLOCK ({elapsed:.1f}s >= {max_wall_clock}s)"
            )
            log.append(f"[BUDGET HIT] {budget_termination}")
            break

        # ── Check token budget ───────────────────────────────────────────
        cumulative = total_input_tokens + total_output_tokens
        if cumulative >= max_tokens:
            budget_termination = (
                f"MAX_TOKENS ({cumulative} >= {max_tokens})"
            )
            log.append(f"[BUDGET HIT] {budget_termination}")
            break

        # ── Check cost budget ────────────────────────────────────────────
        cost_so_far = _estimate_cost(total_input_tokens, total_output_tokens)
        if cost_so_far >= max_cost:
            budget_termination = (
                f"MAX_COST (${cost_so_far:.6f} >= ${max_cost})"
            )
            log.append(f"[BUDGET HIT] {budget_termination}")
            break

        iteration += 1
        log.append(f"\n--- Iteration {iteration} ---")

        # ── Call Groq (with retry on rate-limit / transient errors) ──────
        try:
            response = groq_call_with_retry(
                client,
                model=MODEL,
                messages=messages,
                tools=_TOOLS,
                log=log,
            )
        except Exception as exc:
            log.append(f"[ERROR] Groq call failed: {exc}")
            budget_termination = f"ERROR: {exc}"
            break

        # ── Track tokens (cumulative — every lap re-sends history) ───────
        if getattr(response, "usage", None):
            i_tok = response.usage.prompt_tokens or 0
            o_tok = response.usage.completion_tokens or 0
            total_input_tokens  += i_tok
            total_output_tokens += o_tok
            log.append(f"Tokens this turn:  input={i_tok}  output={o_tok}")
            log.append(
                f"Cumulative tokens: input={total_input_tokens}  "
                f"output={total_output_tokens}  "
                f"total={total_input_tokens + total_output_tokens}"
            )

        # ── Process response ─────────────────────────────────────────────
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                log.append(f"Tool call: {tc.function.name}({json.dumps(args)})")

                result = _dispatch_tool(tc.function.name, args)
                tools_called.append({
                    "name": tc.function.name,
                    "args": args,
                    "result_preview": result[:300],
                })
                log.append(f"Tool result: {result[:300]}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            final_answer = msg.content
            log.append(f"Final answer received (len={len(final_answer or '')})")
            break

    # ── Check iteration-limit budget ─────────────────────────────────────
    if iteration >= max_iters and not final_answer and not budget_termination:
        budget_termination = (
            f"MAX_ITERATIONS ({iteration} >= {max_iters})"
        )
        log.append(f"[BUDGET HIT] {budget_termination}")

    elapsed = time.time() - start
    total_tok = total_input_tokens + total_output_tokens
    cost_final = _estimate_cost(total_input_tokens, total_output_tokens)

    if not final_answer:
        # Produce a partial answer from whatever was gathered
        gathered = (tools_called[-1]["result_preview"]
                    if tools_called else "No information gathered.")
        final_answer = (
            f"[Agent terminated: {budget_termination or 'unknown'}] "
            f"Based on information gathered so far: {gathered}"
        )

    return {
        "answer":             final_answer,
        "tools_called":       tools_called,
        "iterations":         iteration,
        "total_tokens":       total_tok,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "latency_s":          round(elapsed, 3),
        "cost_usd":           round(cost_final, 6),
        "budget_termination": budget_termination,
        "log":                log,
    }
