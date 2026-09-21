"""Captures the model's handling of the SAME failing tool call, once
before and once after rewriting search_clause's error path in
week9/servers/clause_server.py from a terse "Error: not found" into a
descriptive, recoverable message.

This script is run TWICE by hand, in order:
  1. Before the rewrite  -> python -m week9.error_before_after before
  2. Make the rewrite in week9/servers/clause_server.py
  3. After the rewrite   -> python -m week9.error_before_after after

Each run appends its transcript to week9/error_before_after.md so both
halves land in one file, in the order they actually happened.

Run:
    python -m week9.error_before_after before
    python -m week9.error_before_after after
"""

import os
import sys

from week9.agent import run_agent

FAILING_QUESTION = "What does the warranty disclaimer clause say?"


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "before"
    assert label in ("before", "after"), "usage: python -m week9.error_before_after before|after"

    log: list = []
    result = run_agent(FAILING_QUESTION, log=log)

    tool_result = None
    for t in result["tools_called"]:
        if t["name"] == "search_clause":
            tool_result = t["result_preview"]
            break

    report_path = os.path.join(os.path.dirname(__file__), "error_before_after.md")
    section = []
    if label == "before":
        section.append("# Week 9 — Error Path Before/After\n")
        section.append(
            "Same failing question run twice against `search_clause` in "
            "`week9/servers/clause_server.py` — once with the original terse "
            "error, once after rewriting it to be descriptive and recoverable.\n"
        )
        section.append(f"**Question (both runs)**: {FAILING_QUESTION}\n")

    heading = "## Before (terse error)" if label == "before" else "## After (recoverable error)"
    section.append(f"{heading}\n")
    section.append(f"**Raw tool result from `search_clause`**: `{tool_result}`\n")
    section.append(f"**Model's final answer to the user**:\n\n> {result['answer']}\n")

    mode = "w" if label == "before" else "a"
    with open(report_path, mode, encoding="utf-8") as f:
        f.write("\n".join(section) + "\n")

    print(f"[OK] Wrote '{label}' section -> {report_path}")
    print(f"Tool result: {tool_result}")
    print(f"Model answer: {result['answer'][:300]}")


if __name__ == "__main__":
    main()
