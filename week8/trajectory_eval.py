"""Trajectory-gap analysis for the Week 7 contract agent.

Runs the agent (not the workflow) on the 4 most tool-dependent questions,
compares the ACTUAL tool-call sequence against a hand-defined EXPECTED
sequence, and flags cases where the answer was graded correct but the
path taken would not generalize (a "trajectory gap").

Run:
    python -m week8.trajectory_eval
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from week7.agent import run_agent
from week7.race import QUESTIONS, _grade

INTER_QUESTION_DELAY = 10

# ── Questions selected for trajectory analysis ──────────────────────────────
# Picked because a correct answer here plausibly requires chaining tools,
# so they are the ones most likely to expose "right answer, wrong path".

SELECTED_QIDS = ["Q04", "Q07", "Q09", "Q10"]

EXPECTED_TRAJECTORIES = {
    "Q04": ["search_clause", "get_definitions"],
    "Q07": ["search_clause", "get_definitions"],
    "Q09": ["search_clause"],
    "Q10": ["search_clause", "search_clause"],
}


def _find_question(qid: str) -> dict:
    for q in QUESTIONS:
        if q["id"] == qid:
            return q
    raise ValueError(f"Unknown qid: {qid}")


def _has_missing_tool_chain(actual: list[str], expected: list[str]) -> bool:
    """True if an expected tool never appears in the actual sequence."""
    return any(tool not in actual for tool in set(expected))


def _has_redundant_call(tools_called: list[dict]) -> bool:
    seen = set()
    for t in tools_called:
        key = (t["name"], tuple(sorted(t["args"].items())))
        if key in seen:
            return True
        seen.add(key)
    return False


def run_trajectory_eval():
    out_dir = os.path.dirname(__file__)
    rows = []

    print("=" * 70)
    print("  WEEK 8 · Trajectory Gap Analysis")
    print("=" * 70)

    for i, qid in enumerate(SELECTED_QIDS):
        q = _find_question(qid)
        expected = EXPECTED_TRAJECTORIES[qid]

        print(f"\n[{qid}] {q['question'][:70]}...")
        try:
            result = run_agent(q["question"])
        except Exception as exc:
            print(f"  ERROR: {exc}")
            result = {
                "answer": f"Error: {exc}", "tools_called": [],
                "iterations": 0, "log": [],
            }

        actual = [t["name"] for t in result["tools_called"]]
        passed = _grade(result["answer"], q["expected_keywords"])
        missing = _has_missing_tool_chain(actual, expected)
        redundant = _has_redundant_call(result["tools_called"])

        gap_found = passed and (missing or redundant)
        gap_reason = []
        if passed and missing:
            gap_reason.append(
                f"answer graded correct but never called expected tool(s): "
                f"{set(expected) - set(actual)}"
            )
        if passed and redundant:
            gap_reason.append("called the same tool with the same args more than once")
        if not gap_reason:
            gap_reason.append("none")

        rows.append({
            "qid": qid,
            "question": q["question"],
            "expected": expected,
            "actual": actual,
            "passed": passed,
            "gap_found": gap_found,
            "gap_reason": "; ".join(gap_reason),
        })

        print(f"  Passed: {passed}  Actual: {actual}  Expected: {expected}")
        print(f"  Gap found: {gap_found} ({rows[-1]['gap_reason']})")

        if i < len(SELECTED_QIDS) - 1:
            time.sleep(INTER_QUESTION_DELAY)

    # ── Write report ─────────────────────────────────────────────────────
    report_path = os.path.join(out_dir, "trajectory_report.md")
    lines = ["# Week 8 — Trajectory Gap Report\n"]
    lines.append(
        "Outcome (did it pass the keyword grader) vs trajectory (did it call "
        "the tools that make the answer reliable) for the 4 most "
        "tool-dependent questions from the Week 7 question set.\n"
    )
    lines.append("| QID | Passed | Expected Trajectory | Actual Trajectory | Gap Found | Reason |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['qid']} | {r['passed']} | {' → '.join(r['expected'])} | "
            f"{' → '.join(r['actual']) if r['actual'] else '(none)'} | "
            f"{r['gap_found']} | {r['gap_reason']} |"
        )

    n_gaps = sum(1 for r in rows if r["gap_found"])
    lines.append(f"\n**Summary**: {n_gaps}/{len(rows)} questions showed a trajectory gap "
                  f"(right-looking answer, unreliable path).\n")

    if n_gaps:
        gap_rows = [r for r in rows if r["gap_found"]]
        lines.append("## Worst example\n")
        worst = gap_rows[0]
        lines.append(f"**{worst['qid']}**: {worst['question']}\n")
        lines.append(f"- Expected: {' → '.join(worst['expected'])}")
        lines.append(f"- Actual: {' → '.join(worst['actual']) if worst['actual'] else '(none)'}")
        lines.append(f"- Why this matters: {worst['gap_reason']}. The keyword grader "
                      f"scored this as a pass, but the agent reached it without doing "
                      f"the lookup that guarantees correctness — a different phrasing "
                      f"of the same question could easily fail.")
    else:
        lines.append(
            "## No gap found in this batch\n\n"
            "All passing answers in this sample called the tools that make them "
            "reliable. See `week8/fix_report.md` for a gap found in a larger "
            "or repeated sample.\n"
        )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[OK] trajectory_report.md written -> {report_path}")
    return rows


if __name__ == "__main__":
    run_trajectory_eval()
