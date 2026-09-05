"""Race runner — agent vs fixed workflow over 10 contract questions.

Produces:
  week7/race.csv         — 8 numbers (4 per system)
  week7/budget_log.txt   — log of one budget-triggered termination
  week7/verdict.md       — under-150-word verdict
  week7/tool_diff.md     — third tool description diff

Run:
    python -m week7.race
"""

import csv
import json
import os
import statistics
import sys
import time

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from week7.agent import run_agent
from week7.workflow import run_workflow

# Delay between questions to respect API rate limits (seconds)
INTER_QUESTION_DELAY = 5


# ── 10 Questions ─────────────────────────────────────────────────────────
# At least 3 where step 3 depends on what step 2 found (marked DEP).

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


# ── Grading ──────────────────────────────────────────────────────────────

def _grade(answer: str, expected_keywords: list[str]) -> bool:
    """Pass if at least half the expected keywords appear in the answer."""
    if not answer:
        return False
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits >= max(1, len(expected_keywords) // 2)


# ── Race ─────────────────────────────────────────────────────────────────

def run_race():
    out_dir = os.path.join(os.path.dirname(__file__))

    agent_results   = []
    workflow_results = []

    print("=" * 70)
    print("  RACE: Agent Loop  vs  Fixed Workflow")
    print("=" * 70)

    for q in QUESTIONS:
        qid = q["id"]
        question = q["question"]
        kws = q["expected_keywords"]

        # ── Run agent ────────────────────────────────────────────────────
        print(f"\n[{qid}] Agent  ... ", end="", flush=True)
        try:
            a = run_agent(question)
        except Exception as exc:
            print(f"ERROR: {exc}")
            a = {
                "answer": f"Error: {exc}", "iterations": 0,
                "total_tokens": 0, "latency_s": 0, "cost_usd": 0,
                "budget_termination": None, "log": [],
                "total_input_tokens": 0, "total_output_tokens": 0,
                "tools_called": [],
            }
        a_pass = _grade(a["answer"], kws)
        agent_results.append({**a, "qid": qid, "passed": a_pass, "qclass": q["class"]})
        print(f"{'PASS' if a_pass else 'FAIL'}  "
              f"iters={a['iterations']}  tok={a['total_tokens']}  "
              f"lat={a['latency_s']}s  ${a['cost_usd']}")

        time.sleep(INTER_QUESTION_DELAY)   # rate-limit cooldown

        # ── Run workflow ─────────────────────────────────────────────────
        print(f"[{qid}] Wflow ... ", end="", flush=True)
        try:
            w = run_workflow(question)
        except Exception as exc:
            print(f"ERROR: {exc}")
            w = {
                "answer": f"Error: {exc}", "iterations": 1,
                "total_tokens": 0, "latency_s": 0, "cost_usd": 0,
                "budget_termination": None, "log": [],
                "total_input_tokens": 0, "total_output_tokens": 0,
                "tools_called": [],
            }
        w_pass = _grade(w["answer"], kws)
        workflow_results.append({**w, "qid": qid, "passed": w_pass, "qclass": q["class"]})
        print(f"{'PASS' if w_pass else 'FAIL'}  "
              f"iters={w['iterations']}  tok={w['total_tokens']}  "
              f"lat={w['latency_s']}s  ${w['cost_usd']}")

        time.sleep(INTER_QUESTION_DELAY)   # rate-limit cooldown

    # ── Aggregates ───────────────────────────────────────────────────────

    def _agg(results):
        passes   = sum(1 for r in results if r["passed"])
        lats     = sorted(r["latency_s"] for r in results)
        p50_lat  = statistics.median(lats) if lats else 0
        tot_tok  = sum(r["total_tokens"] for r in results)
        tot_cost = sum(r["cost_usd"] for r in results)
        cpt      = tot_cost / len(results) if results else 0
        return {
            "pass_rate":      f"{passes}/{len(results)}",
            "p50_latency_s":  round(p50_lat, 3),
            "total_tokens":   tot_tok,
            "cost_per_q_usd": round(cpt, 6),
        }

    agent_agg    = _agg(agent_results)
    workflow_agg = _agg(workflow_results)

    # ── race.csv ─────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "race.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "agent", "workflow"])
        writer.writerow(["pass_rate",
                          agent_agg["pass_rate"],
                          workflow_agg["pass_rate"]])
        writer.writerow(["p50_latency_s",
                          agent_agg["p50_latency_s"],
                          workflow_agg["p50_latency_s"]])
        writer.writerow(["total_tokens",
                          agent_agg["total_tokens"],
                          workflow_agg["total_tokens"]])
        writer.writerow(["cost_per_question_usd",
                          agent_agg["cost_per_q_usd"],
                          workflow_agg["cost_per_q_usd"]])

    print(f"\n[OK] race.csv written -> {csv_path}")

    # ── Per-question detail CSV ──────────────────────────────────────────
    detail_path = os.path.join(out_dir, "race_detail.csv")
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "class", "system", "passed", "iterations",
                          "total_tokens", "latency_s", "cost_usd"])
        for a, w in zip(agent_results, workflow_results):
            writer.writerow([a["qid"], a["qclass"], "agent",
                              a["passed"], a["iterations"],
                              a["total_tokens"], a["latency_s"], a["cost_usd"]])
            writer.writerow([w["qid"], w["qclass"], "workflow",
                              w["passed"], w["iterations"],
                              w["total_tokens"], w["latency_s"], w["cost_usd"]])

    print(f"[OK] race_detail.csv written -> {detail_path}")

    # ── Print summary table ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Metric':<25} {'Agent':>15} {'Workflow':>15}")
    print("-" * 70)
    for key in ["pass_rate", "p50_latency_s", "total_tokens", "cost_per_q_usd"]:
        print(f"{key:<25} {str(agent_agg[key]):>15} {str(workflow_agg[key]):>15}")
    print("=" * 70)

    # ── Budget-termination log ───────────────────────────────────────────
    print("\n>>> Running budget-termination demo (MAX_ITERATIONS=2) ...")
    budget_log: list[str] = []
    budget_q = QUESTIONS[7]["question"]  # Q08: version comparison -- needs multiple tools
    budget_result = run_agent(
        budget_q,
        max_iters=2,
        max_tokens=8000,
        max_cost=0.05,
        max_wall_clock=30.0,
        log=budget_log,
    )

    log_path = os.path.join(out_dir, "budget_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("BUDGET-TERMINATION LOG\n")
        f.write(f"Question: {budget_q}\n")
        f.write(f"Budget overrides: MAX_ITERATIONS=2\n")
        f.write("=" * 70 + "\n\n")
        for line in budget_log:
            f.write(line + "\n")
        f.write(f"\n--- Result ---\n")
        f.write(f"Answer:             {budget_result['answer'][:300]}\n")
        f.write(f"Iterations:         {budget_result['iterations']}\n")
        f.write(f"Total tokens:       {budget_result['total_tokens']}\n")
        f.write(f"Latency:            {budget_result['latency_s']}s\n")
        f.write(f"Cost:               ${budget_result['cost_usd']}\n")
        f.write(f"Budget termination: {budget_result['budget_termination']}\n")
    print(f"[OK] budget_log.txt written -> {log_path}")

    # ── Verdict ──────────────────────────────────────────────────────────
    # Determine which question classes the agent won/lost
    agent_only_pass = []
    workflow_only_pass = []
    both_pass = []
    both_fail = []
    for a, w in zip(agent_results, workflow_results):
        if a["passed"] and not w["passed"]:
            agent_only_pass.append(a["qclass"])
        elif w["passed"] and not a["passed"]:
            workflow_only_pass.append(a["qclass"])
        elif a["passed"] and w["passed"]:
            both_pass.append(a["qclass"])
        else:
            both_fail.append(a["qclass"])

    # Build verdict based on actual numbers
    dep_classes = [c for c in agent_only_pass if "DEP" in c]
    agent_wins_on_accuracy = (int(agent_agg["pass_rate"].split("/")[0]) >
                              int(workflow_agg["pass_rate"].split("/")[0]))
    workflow_wins_on_cost = (workflow_agg["cost_per_q_usd"] <
                             agent_agg["cost_per_q_usd"])

    verdict_lines = []
    verdict_lines.append("## Verdict\n")

    if dep_classes:
        verdict_lines.append(
            f"The agent outperforms the workflow on the "
            f"**defined-term-chase** question class ({', '.join(dep_classes)}), "
            f"where step 3 depends on what step 2 found. The workflow's fixed "
            f"routing cannot adapt when a clause references an unexpected "
            f"defined term that itself points to a schedule."
        )
    elif agent_only_pass:
        verdict_lines.append(
            f"The agent outperforms the workflow on question classes: "
            f"{', '.join(agent_only_pass)}."
        )
    else:
        verdict_lines.append(
            "No question in this set of 10 strictly requires an agent. "
            "The workflow matches or exceeds the agent on all inputs."
        )

    verdict_lines.append("")

    if workflow_wins_on_cost:
        verdict_lines.append(
            f"The workflow is cheaper (${workflow_agg['cost_per_q_usd']}/q vs "
            f"${agent_agg['cost_per_q_usd']}/q) and faster "
            f"(p50 {workflow_agg['p50_latency_s']}s vs {agent_agg['p50_latency_s']}s) "
            f"because it makes exactly one LLM call instead of re-sending the "
            f"growing message history on every loop iteration."
        )
    else:
        verdict_lines.append(
            f"Cost and latency are comparable between the two systems."
        )

    verdict_lines.append("")
    verdict_lines.append(
        f"**Decision rule**: The path varies by input for the defined-term-chase "
        f"class — the agent must dynamically decide to call get_definitions "
        f"after seeing the clause text reference a term. For single-clause "
        f"lookups and date queries, the workflow suffices. A production system "
        f"should use the workflow as the default path and escalate to the agent "
        f"only when the clause text contains unresolved defined terms."
    )

    verdict_text = "\n".join(verdict_lines)
    verdict_path = os.path.join(out_dir, "verdict.md")
    with open(verdict_path, "w", encoding="utf-8") as f:
        f.write(verdict_text)
    print(f"[OK] verdict.md written -> {verdict_path}")

    # ── Tool diff ────────────────────────────────────────────────────────
    tool_diff_path = os.path.join(out_dir, "tool_diff.md")
    with open(tool_diff_path, "w", encoding="utf-8") as f:
        f.write("## Third Tool: `get_definitions`\n\n")
        f.write("### Description\n")
        f.write("```\n")
        f.write(
            "Resolve a defined contractual term to its full definition, "
            "following any references to schedules. Use this to find what "
            "a defined term means (e.g. 'Business Day', 'Service Period').\n"
        )
        f.write("```\n\n")
        f.write("### Parameters\n")
        f.write("```json\n")
        f.write(json.dumps({
            "term_name": {
                "type": "STRING",
                "description": "The defined term to look up",
            },
            "contract_version": {
                "type": "STRING",
                "description": "Which contract version's definitions to check",
                "enum": ["v1_master", "v2_amendment_1", "v3_amendment_2"],
            },
        }, indent=2))
        f.write("\n```\n\n")
        f.write("### Diff vs existing tools\n\n")
        f.write("| Aspect | search_clause | get_effective_date | **get_definitions** (NEW) |\n")
        f.write("|--------|--------------|-------------------|-------------------------|\n")
        f.write("| Job | Find clause TEXT | Return DATES | Resolve DEFINED TERMS |\n")
        f.write("| Enum param | -- | contract_version Y | contract_version Y |\n")
        f.write("| Input | clause name (str) | version (enum) | term name (str) + version (enum) |\n")
        f.write("| Schedule follow | No | No | **Yes** -- auto-resolves |\n")
        f.write("| Overlap | None with other two | None with other two | None with other two |\n")
    print(f"[OK] tool_diff.md written -> {tool_diff_path}")

    return agent_agg, workflow_agg


# ── Entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_race()
