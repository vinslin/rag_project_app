"""Before/after measurement for the "missing tool chain" fix.

Baseline ("before") reuses week7.agent.run_agent — the same 4 questions and
same gap definition as week8/trajectory_eval.py, so no extra API calls are
spent re-establishing the baseline beyond what that script already did.
"After" runs the same 4 questions through week8.fixed_agent.run_agent_fixed
and reports how many still exhibit the gap.

Run:
    python -m week8.fix_eval
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from week7.agent import run_agent
from week7.race import QUESTIONS, _grade
from week8.fixed_agent import run_agent_fixed, KNOWN_DEFINED_TERMS

INTER_QUESTION_DELAY = 10
SELECTED_QIDS = ["Q04", "Q07", "Q09", "Q10"]


def _find_question(qid: str) -> dict:
    for q in QUESTIONS:
        if q["id"] == qid:
            return q
    raise ValueError(f"Unknown qid: {qid}")


def _missing_tool_chain(tools_called: list[dict]) -> bool:
    """True if clause text references a known defined term that was never
    passed to get_definitions -- the exact gap week8/fixed_agent.py targets."""
    clause_text = " ".join(
        t["result_preview"] for t in tools_called if t["name"] == "search_clause"
    )
    resolved = {
        t["args"].get("term_name") for t in tools_called if t["name"] == "get_definitions"
    }
    return any(term in clause_text and term not in resolved for term in KNOWN_DEFINED_TERMS)


def run_fix_eval():
    out_dir = os.path.dirname(__file__)

    print("=" * 70)
    print("  WEEK 8 · Fix Measurement — missing tool-chain gap")
    print("=" * 70)

    before_rows = []
    after_rows = []

    print("\n--- BEFORE (original week7.agent.run_agent) ---")
    for i, qid in enumerate(SELECTED_QIDS):
        q = _find_question(qid)
        print(f"[{qid}] running...", end=" ", flush=True)
        try:
            result = run_agent(q["question"])
        except Exception as exc:
            print(f"ERROR: {exc}")
            result = {"answer": f"Error: {exc}", "tools_called": []}
        gap = _missing_tool_chain(result["tools_called"])
        passed = _grade(result["answer"], q["expected_keywords"])
        before_rows.append({"qid": qid, "gap": gap, "passed": passed})
        print(f"gap={gap} passed={passed}")
        if i < len(SELECTED_QIDS) - 1:
            time.sleep(INTER_QUESTION_DELAY)

    time.sleep(INTER_QUESTION_DELAY)

    print("\n--- AFTER (week8.fixed_agent.run_agent_fixed) ---")
    for i, qid in enumerate(SELECTED_QIDS):
        q = _find_question(qid)
        print(f"[{qid}] running...", end=" ", flush=True)
        try:
            result = run_agent_fixed(q["question"])
        except Exception as exc:
            print(f"ERROR: {exc}")
            result = {"answer": f"Error: {exc}", "tools_called": [], "forced_corrections": 0}
        gap = _missing_tool_chain(result["tools_called"])
        passed = _grade(result["answer"], q["expected_keywords"])
        after_rows.append({
            "qid": qid, "gap": gap, "passed": passed,
            "forced_corrections": result.get("forced_corrections", 0),
        })
        print(f"gap={gap} passed={passed} forced_corrections={result.get('forced_corrections', 0)}")
        if i < len(SELECTED_QIDS) - 1:
            time.sleep(INTER_QUESTION_DELAY)

    n_before_gap = sum(1 for r in before_rows if r["gap"])
    n_after_gap = sum(1 for r in after_rows if r["gap"])

    # ── Report ───────────────────────────────────────────────────────────
    report_path = os.path.join(out_dir, "fix_report.md")
    lines = ["# Week 8 — Fix Report: Missing Tool-Chain Gap\n"]
    lines.append(
        "**Failure mode fixed**: the agent sometimes gives a final answer "
        "using only the raw clause text returned by `search_clause`, without "
        "calling `get_definitions` on a capitalized defined term the clause "
        "references (e.g. 'Business Day'). This is a trajectory gap, not "
        "necessarily an outcome failure -- the keyword grader can still mark "
        "it a pass if the clause text happens to contain the expected words.\n"
    )
    lines.append(
        "**Fix applied** (`week8/fixed_agent.py`): (1) an explicit MUST-rule "
        "in the system prompt naming this contract's defined terms, and "
        "(2) a deterministic guard that scans gathered clause text for any "
        "known defined term left unresolved and forces one corrective "
        "`get_definitions` call before accepting the final answer.\n"
    )

    lines.append("## Before\n")
    lines.append("| QID | Gap present | Passed |")
    lines.append("|---|---|---|")
    for r in before_rows:
        lines.append(f"| {r['qid']} | {r['gap']} | {r['passed']} |")
    lines.append(f"\n**{n_before_gap}/{len(before_rows)} questions showed the gap.**\n")

    lines.append("## After\n")
    lines.append("| QID | Gap present | Passed | Forced corrections |")
    lines.append("|---|---|---|---|")
    for r in after_rows:
        lines.append(f"| {r['qid']} | {r['gap']} | {r['passed']} | {r['forced_corrections']} |")
    lines.append(f"\n**{n_after_gap}/{len(after_rows)} questions showed the gap.**\n")

    lines.append(f"## Result\n")
    lines.append(f"Missing tool-chain gap: **{n_before_gap}/{len(before_rows)} → "
                 f"{n_after_gap}/{len(after_rows)}**.\n")

    lines.append("## What could still get through\n")
    lines.append(
        "- The guard only recognizes the defined-term names hardcoded in "
        "this corpus (`week7.contracts.DEFINED_TERMS`). A new or renamed "
        "defined term (e.g. an Amendment 3 that introduces 'Force Majeure "
        "Event' as a defined term) would not be caught until the list is "
        "updated by hand.\n"
        "- The guard checks whether get_definitions was *called* for the "
        "term, not whether the model actually *used* the returned "
        "definition correctly in its final answer -- a model could call the "
        "tool and still ignore the result.\n"
        "- This fixes one specific failure mode (missing tool chaining). It "
        "does not address other modes from the same family (loops, wrong "
        "tool choice, made-up contract_version values) -- those would need "
        "their own guards.\n"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[OK] fix_report.md written -> {report_path}")
    return before_rows, after_rows


if __name__ == "__main__":
    run_fix_eval()
