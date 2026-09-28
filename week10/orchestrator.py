"""Week 10 — Multi-agent orchestrator.

Architecture
------------
  User question
       │
       ▼
  Orchestrator LLM call 1: DECOMPOSE
    - Decides which sub-tasks to send to each worker.
    - Outputs a JSON plan: {clause_task, defterm_task}.
       │
       ├─→ clause_worker(clause_task)        [narrow: search_clause + get_effective_date]
       │       └─→ clause_result (answer + tokens)
       │
       ├─→ defined_terms_worker(defterm_task) [narrow: get_definitions]
       │       └─→ defterm_result (answer + tokens)   ← may 500; orchestrator retries once, then degrades
       │
       ▼
  Orchestrator LLM call 2: SYNTHESISE
    - Merges clause_result + defterm_result into a final answer.

Per-hand-off token tracking
---------------------------
Every crossing of a worker boundary is logged with its token count.
The orchestrator accumulates ORCHESTRATOR_TOKENS (decompose call + synthesise call)
plus WORKER_TOKENS per worker. The total is orchestrator + all workers combined.

handoffs.log columns
--------------------
  timestamp | hand-off label | tokens_this_hop | cumulative_tokens
"""

import json
import time
import os
import sys
from typing import Optional

from week7.llm_client import get_client, groq_call_with_retry
from rag import config as rag_config
from week10.workers import (
    clause_worker,
    defined_terms_worker,
    reset_failure_state,
    _estimate_cost,
)

MODEL = rag_config.GENERATION_MODEL


# ---------------------------------------------------------------------------
# Orchestrator system prompts
# ---------------------------------------------------------------------------

DECOMPOSE_SYSTEM = (
    "You are a contract-question orchestrator. "
    "Your job is to decompose the user's question into exactly two sub-tasks:\n"
    "  1. clause_task — what to ask the clause-retrieval worker "
    "(it can find clause text and effective dates, but NOT defined terms).\n"
    "  2. defterm_task — what to ask the defined-terms worker "
    "(it can resolve defined terms and follow schedule references, "
    "but NOT retrieve clause text or dates). "
    "If the question does not require defined terms, set defterm_task to null.\n\n"
    "Respond ONLY with valid JSON, no prose, no markdown fences:\n"
    '{"clause_task": "<string>", "defterm_task": "<string or null>"}'
)

SYNTHESISE_SYSTEM = (
    "You are a contract-question synthesiser. "
    "You will receive the original user question plus the findings from two specialist workers. "
    "Combine those findings into a single, precise, cited answer. "
    "Do NOT add information that was not in the worker answers. "
    "Cite the contract version and section for every fact."
)


# ---------------------------------------------------------------------------
# Hand-off log helper
# ---------------------------------------------------------------------------

class HandoffLogger:
    """Accumulates per-hand-off records for handoffs.log."""

    def __init__(self):
        self.records: list[dict] = []
        self._cumulative = 0

    def log(self, label: str, tokens: int):
        self._cumulative += tokens
        self.records.append({
            "label":       label,
            "tokens_hop":  tokens,
            "cumulative":  self._cumulative,
        })

    def total(self) -> int:
        return self._cumulative

    def to_lines(self) -> list[str]:
        lines = [
            f"{'HAND-OFF LABEL':<45} {'TOKENS (HOP)':>12} {'CUMULATIVE':>12}",
            "-" * 72,
        ]
        for r in self.records:
            lines.append(
                f"{r['label']:<45} {r['tokens_hop']:>12,} {r['cumulative']:>12,}"
            )
        lines.append("-" * 72)
        lines.append(f"{'TOTAL':>57} {self._cumulative:>12,}")
        return lines


# ---------------------------------------------------------------------------
# Core orchestrator
# ---------------------------------------------------------------------------

def run_orchestrator(
    question: str,
    question_id: Optional[str] = None,
    log: Optional[list] = None,
    handoff_logger: Optional[HandoffLogger] = None,
    inject_failure: bool = False,
) -> dict:
    """Run the multi-agent orchestrator for one question.

    Parameters
    ----------
    question        : the user's question
    question_id     : Q01..Q10 — used for failure injection routing
    log             : accumulated log lines (mutated in-place)
    handoff_logger  : HandoffLogger shared across all questions in a race
    inject_failure  : if True, defined_terms_worker will 500 once then succeed on retry

    Returns
    -------
    dict with keys:
        answer, total_tokens, total_input_tokens, total_output_tokens,
        latency_s, cost_usd, handoffs, log,
        defterm_behaviour  (None | 'retry_succeeded' | 'degraded' | 'lied')
    """
    if log is None:
        log = []
    if handoff_logger is None:
        handoff_logger = HandoffLogger()

    start = time.time()
    client = get_client()

    total_input  = 0
    total_output = 0

    # ── LLM call 1: DECOMPOSE ──────────────────────────────────────────────
    log.append(f"\n[orchestrator] DECOMPOSE  qid={question_id}")
    decompose_messages = [
        {"role": "system",  "content": DECOMPOSE_SYSTEM},
        {"role": "user",    "content": question},
    ]
    decompose_resp = groq_call_with_retry(
        client, model=MODEL, messages=decompose_messages, tools=None, log=log
    )
    d_in  = (decompose_resp.usage.prompt_tokens     or 0) if decompose_resp.usage else 0
    d_out = (decompose_resp.usage.completion_tokens or 0) if decompose_resp.usage else 0
    total_input  += d_in
    total_output += d_out
    handoff_logger.log(f"orchestrator → decompose LLM call ({question_id})", d_in + d_out)
    log.append(f"[orchestrator] decompose tokens: input={d_in} output={d_out}")

    raw_plan = decompose_resp.choices[0].message.content or ""
    log.append(f"[orchestrator] decompose plan raw: {raw_plan[:400]}")

    # Parse plan JSON (graceful fallback)
    try:
        # Strip possible markdown fences
        clean = raw_plan.strip()
        if clean.startswith("```"):
            lines = clean.splitlines()
            clean = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        plan = json.loads(clean)
        clause_task  = plan.get("clause_task")  or question
        defterm_task = plan.get("defterm_task")  # may be None/null
    except Exception as exc:
        log.append(f"[orchestrator] plan parse error ({exc}) — falling back to full question for both")
        clause_task  = question
        defterm_task = question

    log.append(f"[orchestrator] clause_task:  {clause_task}")
    log.append(f"[orchestrator] defterm_task: {defterm_task}")

    # ── Call clause_worker ─────────────────────────────────────────────────
    log.append(f"\n[orchestrator → clause_worker] sending: {clause_task[:120]}")
    c_worker_log: list = []
    c_result = clause_worker(clause_task, log=c_worker_log)
    log.extend(c_worker_log)

    c_tokens = c_result["total_tokens"]
    total_input  += c_result["total_input_tokens"]
    total_output += c_result["total_output_tokens"]
    handoff_logger.log(f"orchestrator → clause_worker ({question_id})", c_tokens)
    log.append(f"[clause_worker → orchestrator] tokens={c_tokens}  answer_len={len(c_result['answer'])}")

    # ── Call defined_terms_worker (with failure injection + retry) ─────────
    defterm_answer    = None
    defterm_behaviour = None   # None | 'not_needed' | 'retry_succeeded' | 'degraded' | 'lied'
    dt_tokens         = 0

    if defterm_task:
        log.append(f"\n[orchestrator → defined_terms_worker] sending: {defterm_task[:120]}")

        if inject_failure:
            import week10.workers as _workers_mod
            _workers_mod.INJECT_FAILURE_CASE = question_id
        reset_failure_state()

        dt_worker_log: list = []
        try:
            dt_result = defined_terms_worker(defterm_task, question_id=question_id, log=dt_worker_log)
            defterm_answer    = dt_result["answer"]
            defterm_behaviour = None  # success on first try
            dt_tokens = dt_result["total_tokens"]
            total_input  += dt_result["total_input_tokens"]
            total_output += dt_result["total_output_tokens"]
            log.append(f"[defined_terms_worker → orchestrator] tokens={dt_tokens}")

        except RuntimeError as err:
            log.append(f"[orchestrator] defined_terms_worker 500: {err}")
            log.append("[orchestrator] RETRY defined_terms_worker (attempt 2/2) ...")

            # Retry once (failure flag already set — second call succeeds)
            dt_worker_log2: list = []
            try:
                dt_result2 = defined_terms_worker(defterm_task, question_id=question_id, log=dt_worker_log2)
                defterm_answer    = dt_result2["answer"]
                defterm_behaviour = "retry_succeeded"
                dt_tokens = dt_result2["total_tokens"]
                total_input  += dt_result2["total_input_tokens"]
                total_output += dt_result2["total_output_tokens"]
                log.append(f"[defined_terms_worker → orchestrator] RETRY OK  tokens={dt_tokens}")
            except Exception as err2:
                # Retry also failed → degrade (partial answer from clause_worker only)
                log.append(f"[orchestrator] RETRY also failed: {err2} — DEGRADING to partial answer")
                defterm_answer    = "[defined-terms worker unavailable — omitted from synthesis]"
                defterm_behaviour = "degraded"

            log.extend(dt_worker_log2)

        log.extend(dt_worker_log)

        if dt_tokens:
            handoff_logger.log(f"orchestrator → defined_terms_worker ({question_id})", dt_tokens)
    else:
        defterm_answer    = "No defined-term resolution required for this question."
        defterm_behaviour = "not_needed"
        log.append("[orchestrator] defterm_task=null — skipping defined_terms_worker")

    # ── LLM call 2: SYNTHESISE ─────────────────────────────────────────────
    log.append(f"\n[orchestrator] SYNTHESISE  qid={question_id}")
    synthesise_messages = [
        {"role": "system", "content": SYNTHESISE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"ORIGINAL QUESTION:\n{question}\n\n"
                f"CLAUSE-RETRIEVAL WORKER ANSWER:\n{c_result['answer']}\n\n"
                f"DEFINED-TERMS WORKER ANSWER:\n{defterm_answer}"
            ),
        },
    ]
    synth_resp = groq_call_with_retry(
        client, model=MODEL, messages=synthesise_messages, tools=None, log=log
    )
    s_in  = (synth_resp.usage.prompt_tokens     or 0) if synth_resp.usage else 0
    s_out = (synth_resp.usage.completion_tokens or 0) if synth_resp.usage else 0
    total_input  += s_in
    total_output += s_out
    handoff_logger.log(f"orchestrator → synthesise LLM call ({question_id})", s_in + s_out)
    log.append(f"[orchestrator] synthesise tokens: input={s_in} output={s_out}")

    final_answer = synth_resp.choices[0].message.content or "[no synthesis produced]"

    elapsed = time.time() - start
    total_tokens = total_input + total_output

    return {
        "answer":               final_answer,
        "total_tokens":         total_tokens,
        "total_input_tokens":   total_input,
        "total_output_tokens":  total_output,
        "latency_s":            round(elapsed, 3),
        "cost_usd":             round(_estimate_cost(total_input, total_output), 6),
        "handoffs":             handoff_logger.records[:],  # snapshot
        "defterm_behaviour":    defterm_behaviour,
        "log":                  log,
    }
