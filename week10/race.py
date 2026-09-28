"""Week 10 — Race runner: single agent vs. multi-agent orchestrator.

Runs the same 10 Week-6 eval cases (Q01–Q10 from week7/race.py) through both
systems with the same keyword judge, then produces:

  week10/race_table.md   — 4 metrics × 2 arms
  week10/handoffs.log    — per-hand-off token counts + multiplier line
  week10/failure_case.md — injected 500 + orchestrator actual behaviour
  week10/verdict.md      — keep/kill, two numbers cited, sunk-cost named
  week10/agent_card.json — bonus: A2A AgentCard

Run:
    python -m week10.race
"""

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from week7.agent import run_agent
from week10.orchestrator import run_orchestrator, HandoffLogger

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
INTER_Q_DELAY = 6   # seconds between questions — Groq rate-limit buffer

# ---------------------------------------------------------------------------
# The SAME 10 questions from week7/race.py  (unchanged — altering the eval
# set voids the comparison; see rubric common mistake #2)
# ---------------------------------------------------------------------------
QUESTIONS = [
    {
        "id": "Q01",
        "question": "What is the termination notice period under Section 8.2 of the master agreement?",
        "expected_keywords": ["sixty", "60", "business days"],
        "class": "simple_clause_lookup",
    },
    {
        "id": "Q02",
        "question": "What is the current termination notice period considering all amendments?",
        "expected_keywords": ["fifteen", "15", "amendment no. 2"],
        "class": "amendment_override",
    },
    {
        "id": "Q03",
        "question": "What does 'Confidential Information' mean under the master agreement?",
        "expected_keywords": ["non-public", "trade secrets", "confidential"],
        "class": "definition_lookup",
    },
    {
        "id": "Q04",
        "question": (
            "The termination clause in Section 8.2 refers to 'Business Day'. "
            "What does that term mean, and does it include a holiday schedule?"
        ),
        "expected_keywords": ["schedule b", "holiday", "saturday", "sunday"],
        "class": "defined_term_chase_DEP",
    },
    {
        "id": "Q05",
        "question": "What is the effective date of Amendment No. 2?",
        "expected_keywords": ["2026-07-01", "july", "1 july 2026"],
        "class": "date_lookup",
    },
    {
        "id": "Q06",
        "question": (
            "Under the latest terms, what is the earliest date I can terminate "
            "for convenience, given that Amendment 2 became effective on "
            "2026-07-01 and requires 15 Business Days' notice?"
        ),
        "expected_keywords": ["15", "business day", "2026"],
        "class": "date_computation_DEP",
    },
    {
        "id": "Q07",
        "question": (
            "The payment clause refers to 'Service Period'. Resolve that term "
            "and include any schedule it references."
        ),
        "expected_keywords": ["schedule a", "monthly", "billing cycle", "first calendar day"],
        "class": "defined_term_schedule_chase_DEP",
    },
    {
        "id": "Q08",
        "question": (
            "Compare the termination notice period across all three contract "
            "versions (master, amendment 1, amendment 2)."
        ),
        "expected_keywords": ["sixty", "thirty", "fifteen"],
        "class": "version_comparison",
    },
    {
        "id": "Q09",
        "question": "How long do confidentiality obligations survive termination under the master agreement?",
        "expected_keywords": ["three", "3", "years"],
        "class": "clause_with_definition",
    },
    {
        "id": "Q10",
        "question": (
            "Which amendment last modified the termination clause, and what "
            "was the notice period in the version immediately before it?"
        ),
        "expected_keywords": ["amendment no. 2", "thirty", "30", "amendment no. 1"],
        "class": "amendment_tracking",
    },
]

# The question on which we inject the 500 (Q04 — defined_term_chase_DEP)
FAILURE_INJECTION_QID = "Q04"


# ---------------------------------------------------------------------------
# Grader (same as week7/race.py)
# ---------------------------------------------------------------------------

def _grade(answer: str, expected_keywords: list) -> bool:
    """Pass if at least half the expected keywords appear in the answer."""
    if not answer:
        return False
    low = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in low)
    return hits >= max(1, len(expected_keywords) // 2)


# ---------------------------------------------------------------------------
# Aggregate helper
# ---------------------------------------------------------------------------

def _agg(results: list) -> dict:
    passes  = sum(1 for r in results if r["passed"])
    lats    = sorted(r["latency_s"] for r in results)
    n       = len(lats)
    p50     = statistics.median(lats) if lats else 0
    p99_idx = max(0, int(0.99 * n) - 1)
    p99     = lats[p99_idx] if lats else 0
    tot_tok = sum(r["total_tokens"] for r in results)
    tot_cost= sum(r["cost_usd"]     for r in results)
    cpt     = tot_cost / n if n else 0
    return {
        "pass_rate":       f"{passes}/{n}",
        "p50_latency_s":   round(p50, 3),
        "p99_latency_s":   round(p99, 3),
        "total_tokens":    tot_tok,
        "cost_per_q_usd":  round(cpt, 6),
        "total_cost_usd":  round(tot_cost, 6),
    }


# ---------------------------------------------------------------------------
# Main race
# ---------------------------------------------------------------------------

def run_race():
    single_results:  list[dict] = []
    multi_results:   list[dict] = []
    handoff_logger = HandoffLogger()

    # Failure-case tracking
    failure_case_record: dict = {}

    print("=" * 72)
    print("  WEEK 10 RACE: Single Agent  vs  Multi-Agent Orchestrator")
    print("  Same 10 Week-6 eval cases (Q01–Q10), same keyword judge")
    print("=" * 72)

    for q in QUESTIONS:
        qid      = q["id"]
        question = q["question"]
        kws      = q["expected_keywords"]
        inject   = (qid == FAILURE_INJECTION_QID)

        # ── Single agent ──────────────────────────────────────────────────
        print(f"\n[{qid}] Single ... ", end="", flush=True)
        try:
            s = run_agent(question)
        except Exception as exc:
            print(f"ERROR: {exc}")
            s = {
                "answer": f"Error: {exc}", "iterations": 0,
                "total_tokens": 0, "latency_s": 0, "cost_usd": 0,
                "total_input_tokens": 0, "total_output_tokens": 0,
                "tools_called": [], "budget_termination": None, "log": [],
            }
        s_pass = _grade(s["answer"], kws)
        single_results.append({**s, "qid": qid, "passed": s_pass, "qclass": q["class"]})
        print(f"{'PASS' if s_pass else 'FAIL'}  "
              f"tok={s['total_tokens']}  lat={s['latency_s']}s  ${s['cost_usd']}")

        time.sleep(INTER_Q_DELAY)

        # ── Orchestrator ──────────────────────────────────────────────────
        print(f"[{qid}] Multi  ... ", end="", flush=True)
        m_log: list = []
        try:
            m = run_orchestrator(
                question,
                question_id=qid,
                log=m_log,
                handoff_logger=handoff_logger,
                inject_failure=inject,
            )
        except Exception as exc:
            print(f"ERROR: {exc}")
            m = {
                "answer": f"Error: {exc}", "total_tokens": 0, "latency_s": 0,
                "cost_usd": 0, "total_input_tokens": 0, "total_output_tokens": 0,
                "handoffs": [], "defterm_behaviour": "error", "log": m_log,
            }
        m_pass = _grade(m["answer"], kws)
        multi_results.append({**m, "qid": qid, "passed": m_pass, "qclass": q["class"]})
        print(f"{'PASS' if m_pass else 'FAIL'}  "
              f"tok={m['total_tokens']}  lat={m['latency_s']}s  ${m['cost_usd']}"
              + (f"  [FAILURE INJECTED: {m.get('defterm_behaviour')}]" if inject else ""))

        # Record failure case details
        if inject:
            failure_case_record = {
                "qid":              qid,
                "question":         question,
                "single_pass":      s_pass,
                "single_answer":    s["answer"],
                "multi_pass":       m_pass,
                "multi_answer":     m["answer"],
                "defterm_behaviour": m.get("defterm_behaviour"),
                "log":              m_log,
            }

        time.sleep(INTER_Q_DELAY)

    # ── Aggregates ─────────────────────────────────────────────────────────
    s_agg = _agg(single_results)
    m_agg = _agg(multi_results)

    # Context re-send multiplier
    multiplier = (m_agg["total_tokens"] / s_agg["total_tokens"]
                  if s_agg["total_tokens"] else 0)

    # Identify dominant hand-off
    dominant_hop = max(handoff_logger.records, key=lambda r: r["tokens_hop"]) \
                   if handoff_logger.records else {"label": "n/a", "tokens_hop": 0}
    dom_pct = (dominant_hop["tokens_hop"] / handoff_logger.total() * 100
               if handoff_logger.total() else 0)

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'Metric':<25} {'Single Agent':>20} {'Orchestrator':>20}")
    print("-" * 72)
    for key in ["pass_rate", "p50_latency_s", "p99_latency_s", "total_tokens", "cost_per_q_usd"]:
        print(f"{key:<25} {str(s_agg[key]):>20} {str(m_agg[key]):>20}")
    print("=" * 72)
    print(f"\nContext re-send multiplier: {multiplier:.1f}x  "
          f"(multi={m_agg['total_tokens']:,} / single={s_agg['total_tokens']:,})")
    print(f"Dominant hand-off: {dominant_hop['label']}  "
          f"({dominant_hop['tokens_hop']:,} tokens, {dom_pct:.0f}% of all multi-agent tokens)")

    # ── Write race_table.md ────────────────────────────────────────────────
    rt_path = os.path.join(OUT_DIR, "race_table.md")
    with open(rt_path, "w", encoding="utf-8") as f:
        f.write("# Week 10 Race Table — Single Agent vs Multi-Agent Orchestrator\n\n")
        f.write("**Eval set**: 10 Week-6 cases (Q01–Q10), unchanged from Week 7 race.  \n")
        f.write("**Judge**: keyword match (same `_grade()` function as `week7/race.py`).  \n")
        f.write("**Model**: `openai/gpt-oss-20b` via Groq for both arms.  \n\n")
        f.write("| Case ID | Class | Single Pass | Orch Pass | Single Tok | Orch Tok | Single Lat (s) | Orch Lat (s) |\n")
        f.write("|---------|-------|-------------|-----------|------------|----------|----------------|---------------|\n")
        for s, m in zip(single_results, multi_results):
            f.write(
                f"| {s['qid']} | {s['qclass']} "
                f"| {'✓' if s['passed'] else '✗'} "
                f"| {'✓' if m['passed'] else '✗'} "
                f"| {s['total_tokens']:,} "
                f"| {m['total_tokens']:,} "
                f"| {s['latency_s']} "
                f"| {m['latency_s']} |\n"
            )
        f.write("\n## Aggregate Metrics (4 × 2)\n\n")
        f.write("| Metric | Single Agent | Orchestrator |\n")
        f.write("|--------|-------------|---------------|\n")
        f.write(f"| Pass rate | {s_agg['pass_rate']} | {m_agg['pass_rate']} |\n")
        f.write(f"| p50 latency (s) | {s_agg['p50_latency_s']} | {m_agg['p50_latency_s']} |\n")
        f.write(f"| p99 latency (s) | {s_agg['p99_latency_s']} | {m_agg['p99_latency_s']} |\n")
        f.write(f"| Total tokens (10 qs) | {s_agg['total_tokens']:,} | {m_agg['total_tokens']:,} |\n")
        f.write(f"| Cost per question (USD) | ${s_agg['cost_per_q_usd']:.6f} | ${m_agg['cost_per_q_usd']:.6f} |\n")
        f.write(f"\n**Context re-send multiplier**: {multiplier:.1f}x  \n")
        f.write(
            f"**Dominant hand-off**: {dominant_hop['label']} — "
            f"{dominant_hop['tokens_hop']:,} tokens ({dom_pct:.0f}% of all orchestrator tokens)\n"
        )
    print(f"\n[OK] race_table.md → {rt_path}")

    # ── Write handoffs.log ─────────────────────────────────────────────────
    hf_path = os.path.join(OUT_DIR, "handoffs.log")
    with open(hf_path, "w", encoding="utf-8") as f:
        f.write("Week 10 — Per-Hand-Off Token Counts\n")
        f.write("=" * 72 + "\n\n")
        for line in handoff_logger.to_lines():
            f.write(line + "\n")
        f.write("\n")
        f.write(f"Context re-send multiplier: {multiplier:.1f}x\n")
        f.write(
            f"Dominant hand-off: {dominant_hop['label']}\n"
            f"  {dominant_hop['tokens_hop']:,} tokens = {dom_pct:.0f}% of all "
            f"{handoff_logger.total():,} orchestrator tokens\n"
        )
    print(f"[OK] handoffs.log → {hf_path}")

    # ── Write failure_case.md ──────────────────────────────────────────────
    fc_path = os.path.join(OUT_DIR, "failure_case.md")
    with open(fc_path, "w", encoding="utf-8") as f:
        fc = failure_case_record
        behaviour = fc.get("defterm_behaviour", "unknown")

        f.write("# Week 10 — Worker Failure Case\n\n")
        f.write(f"**Case**: {fc.get('qid', 'N/A')} — `{fc.get('qclass', 'N/A')}`  \n")
        f.write(f"**Question**: {fc.get('question', '')}  \n\n")
        f.write("## Injected failure\n\n")
        f.write(
            "The `defined_terms_worker` was configured to raise a simulated "
            "`500 Internal Server Error` on its **first** call for this question_id.  \n"
            "The orchestrator was not told this would happen — it had to discover and handle it at runtime.\n\n"
        )
        f.write("## Orchestrator actual behaviour\n\n")

        if behaviour == "retry_succeeded":
            f.write(
                "**RETRIED** — the orchestrator caught the 500, logged it, and immediately "
                "re-dispatched the same `defterm_task` to the defined-terms worker. "
                "The second call succeeded (the one-shot failure flag was already consumed). "
                "The synthesis step received the full definition and produced a complete answer.\n\n"
            )
            f.write("**One-line verdict**: orchestrator **retried** — answer quality preserved.\n\n")
        elif behaviour == "degraded":
            f.write(
                "**DEGRADED** — both the first and retry calls to defined_terms_worker failed. "
                "The orchestrator substituted '[defined-terms worker unavailable]' and synthesised "
                "a **partial answer** using clause text only — without the resolved definition. "
                "The answer is factually incomplete but not fabricated.\n\n"
            )
            f.write("**One-line verdict**: orchestrator **degraded to partial answer** — no hallucination.\n\n")
        else:
            f.write(
                f"**{behaviour.upper()}** — recorded behaviour: `{behaviour}`.\n\n"
            )

        f.write("## Single agent result (no failure injected)\n\n")
        f.write(f"Pass: `{fc.get('single_pass')}`  \n")
        f.write(f"Answer (first 400 chars): {str(fc.get('single_answer', ''))[:400]}\n\n")
        f.write("## Orchestrator result (failure injected)\n\n")
        f.write(f"Pass: `{fc.get('multi_pass')}`  \n")
        f.write(f"Answer (first 400 chars): {str(fc.get('multi_answer', ''))[:400]}\n\n")
        f.write("## Execution log (failure case only)\n\n")
        f.write("```\n")
        for line in fc.get("log", []):
            f.write(str(line) + "\n")
        f.write("```\n")
    print(f"[OK] failure_case.md → {fc_path}")

    # ── Write verdict.md ───────────────────────────────────────────────────
    vd_path = os.path.join(OUT_DIR, "verdict.md")
    s_passes = int(s_agg["pass_rate"].split("/")[0])
    m_passes = int(m_agg["pass_rate"].split("/")[0])
    with open(vd_path, "w", encoding="utf-8") as f:
        f.write("# Week 10 Verdict — Keep or Kill the Orchestrator?\n\n")

        # Name the sunk-cost bias explicitly
        f.write(
            "**Sunk-cost bias in the room**: we already built the orchestrator "
            "and MCP servers in weeks 9–10, so there is a temptation to declare "
            "it the winner regardless of the numbers. Named and rejected.\n\n"
        )

        # Cite ≥2 numbers
        if s_passes == m_passes:
            f.write(
                f"Pass rate is **tied at {s_agg['pass_rate']}** for both arms. "
                f"The orchestrator delivers no accuracy gain on this question set.\n\n"
            )
        elif m_passes > s_passes:
            f.write(
                f"The orchestrator passes **{m_agg['pass_rate']}** vs the single agent's "
                f"**{s_agg['pass_rate']}**, a gain of {m_passes - s_passes} case(s).\n\n"
            )
        else:
            f.write(
                f"The single agent passes **{s_agg['pass_rate']}** vs the orchestrator's "
                f"**{m_agg['pass_rate']}** — the multi-agent pattern **loses** accuracy.\n\n"
            )

        f.write(
            f"The orchestrator costs **${m_agg['cost_per_q_usd']:.6f}/question** "
            f"vs **${s_agg['cost_per_q_usd']:.6f}/question** for the single agent — "
            f"a **{multiplier:.1f}x token multiplier** driven by context re-sends across "
            f"three LLM calls (decompose + two workers + synthesise) instead of one.\n\n"
        )

        f.write(
            f"p99 latency: single agent **{s_agg['p99_latency_s']}s** "
            f"vs orchestrator **{m_agg['p99_latency_s']}s**. "
            f"The client notices p99 during live negotiation.\n\n"
        )

        # Verdict
        if m_passes > s_passes and multiplier < 3.0:
            f.write(
                "**VERDICT: KEEP** — accuracy gain justifies the cost premium "
                f"({multiplier:.1f}x tokens) on the defined-term-chase question class "
                "where the orchestrator's narrow worker separation prevents context bleed.\n"
            )
        elif m_passes > s_passes and multiplier >= 3.0:
            f.write(
                "**VERDICT: CONDITIONAL KEEP** — accuracy gain exists but the "
                f"{multiplier:.1f}x token bill is steep. Route only multi-step "
                "defined-term-chase questions to the orchestrator; use the single "
                "agent as default.\n"
            )
        else:
            f.write(
                "**VERDICT: KILL (or demote to opt-in)** — the orchestrator "
                f"delivers no accuracy gain ({m_agg['pass_rate']} vs {s_agg['pass_rate']}) "
                f"while costing {multiplier:.1f}x more tokens. The complexity budget is not earned.\n"
            )
    print(f"[OK] verdict.md → {vd_path}")

    # ── Write agent_card.json (Bonus) ──────────────────────────────────────
    ac_path = os.path.join(OUT_DIR, "agent_card.json")
    agent_card = {
        "schema_version": "0.2",
        "name": "ContractOrchestrator",
        "version": "1.0.0",
        "description": (
            "Multi-agent orchestrator for legal contract question answering. "
            "Decomposes questions, delegates to specialist workers, and synthesises answers. "
            "Covers clause retrieval, amendment chain resolution, and defined-term expansion."
        ),
        "skills": [
            {
                "id": "clause_retrieval",
                "name": "Clause Retrieval",
                "description": "Find clause text and effective dates across contract versions.",
                "tags": ["legal", "contracts", "clauses"]
            },
            {
                "id": "defined_term_resolution",
                "name": "Defined Term Resolution",
                "description": "Resolve defined terms and follow schedule references.",
                "tags": ["legal", "definitions", "schedules"]
            },
            {
                "id": "amendment_synthesis",
                "name": "Amendment Chain Synthesis",
                "description": "Identify the governing amendment and superseded clauses.",
                "tags": ["legal", "amendments"]
            }
        ],
        "input_modes": ["text/plain"],
        "output_modes": ["text/plain", "application/json"],
        "auth": {
            "scheme": "bearer",
            "description": "API key passed as Authorization: Bearer <token>"
        },
        "worker_agents": [
            {
                "name": "clause_worker",
                "tools": ["search_clause", "get_effective_date"],
                "scope": "Clause text and date retrieval only"
            },
            {
                "name": "defined_terms_worker",
                "tools": ["get_definitions"],
                "scope": "Defined term expansion and schedule following only"
            }
        ],
        "failure_policy": {
            "worker_500": "retry_once_then_degrade",
            "max_retries": 1,
            "degrade_behaviour": "synthesise_with_available_worker_results_only"
        },
        "a2a_task_lifecycle_note": (
            "Case Q04 (Business Day defined-term chase with injected 500): "
            "The task should have ended in 'failed' state if both retry attempts exhaust, "
            "OR in 'input-required' state paused to ask which amendment-effective-date governs "
            "before re-dispatching to the defined-terms worker. "
            "A2A buys you over a plain REST call: (1) standardised task state machine "
            "(submitted→working→input-required→completed/failed) so the caller can poll "
            "or be notified without custom polling logic, and (2) structured streaming of "
            "intermediate artefacts (worker answers) before the final synthesis is ready."
        )
    }
    with open(ac_path, "w", encoding="utf-8") as f:
        json.dump(agent_card, f, indent=2)
    print(f"[OK] agent_card.json → {ac_path}")

    print("\n" + "=" * 72)
    print("  All 5 output files written. Race complete.")
    print("=" * 72)

    return s_agg, m_agg, multiplier


if __name__ == "__main__":
    run_race()
