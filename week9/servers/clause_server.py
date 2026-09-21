"""MCP server exposing the Week 7 contract tools over stdio.

This is "our" server — the one the Week 9 practical says already exists.
It wraps the same contract corpus used throughout weeks 7-8
(week7.contracts) but re-implements the tool functions as MCP tools
(decorated, with MCP-visible docstrings) rather than the old hardcoded
Gemini/Groq function declarations in week7/tools.py.

Run standalone for manual testing:
    python -m week9.servers.clause_server
"""

from mcp.server.fastmcp import FastMCP

from week7.contracts import CONTRACTS, DEFINED_TERMS, SCHEDULES

mcp = FastMCP("clause_search")


@mcp.tool()
def search_clause(clause_name: str) -> str:
    """Locate a specific clause by name or section number and return its
    full text. Searches across ALL contract versions (master + amendments).

    If no clause matches, the result names every real clause that DOES
    exist (grouped by version) so the caller can retry with a valid name
    instead of guessing blindly — this tool never returns a bare "not
    found"; it always tells you what to try next.
    """
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
        # Recoverable error: name what's actually available instead of a
        # bare failure code, so the model (or a human) can self-correct
        # on the next call rather than guessing blindly.
        lines = [
            f"No clause matching '{clause_name}' was found in any version "
            f"of MSA-2026-014. This tool matches by clause name or section "
            f"number substring, not free-text description — try one of the "
            f"exact clause names below instead:"
        ]
        for version_key, contract in CONTRACTS.items():
            clause_names = list(contract["clauses"].keys())
            lines.append(f"  [{version_key}] {contract['name']}: {clause_names}")
        return "\n".join(lines)

    parts = []
    for r in results:
        parts.append(
            f"[{r['version']}] {r['contract']}\n"
            f"  {r['clause']}:\n"
            f"  {r['text']}"
        )
    return "\n\n".join(parts)


@mcp.tool()
def get_effective_date(contract_version: str) -> str:
    """Return the effective date and temporal metadata (execution date,
    effective date, expiration date) for a specific contract version.
    Valid contract_version values: v1_master, v2_amendment_1, v3_amendment_2.
    Does NOT return clause text or defined terms — use the other tools for those.
    """
    if contract_version not in CONTRACTS:
        valid = list(CONTRACTS.keys())
        return f"Unknown contract version: {contract_version}. Valid: {valid}"

    c = CONTRACTS[contract_version]
    return (
        f"Contract: {c['name']}\n"
        f"  Execution Date: {c['execution_date']}\n"
        f"  Effective Date:  {c['effective_date']}\n"
        f"  Expiration Date: {c['expiration_date']}"
    )


@mcp.tool()
def get_definitions(term_name: str, contract_version: str) -> str:
    """Resolve a defined contractual term to its full definition, following
    any references to schedules. Looks up the term in the specified
    contract version's definitions and automatically appends the text of
    any Schedule referenced in the definition.
    Valid contract_version values: v1_master, v2_amendment_1, v3_amendment_2.
    Does NOT return clause text or dates — use the other tools for those.
    """
    if contract_version not in DEFINED_TERMS:
        return f"No definitions found for version: {contract_version}"

    terms = DEFINED_TERMS[contract_version]
    query = term_name.lower().strip()

    matched_term = None
    matched_definition = None
    for term_key, definition in terms.items():
        if query in term_key.lower() or term_key.lower() in query:
            matched_term = term_key
            matched_definition = definition
            break

    if not matched_definition:
        available = list(terms.keys())
        return (f"Term '{term_name}' not found in {contract_version} "
                f"definitions. Available terms: {available}")

    result = (f"Term: {matched_term}\n"
              f"Version: {contract_version}\n"
              f"Definition: {matched_definition}")

    for schedule_name, schedule_text in SCHEDULES.items():
        if schedule_name.lower() in matched_definition.lower():
            result += f"\n\nReferenced {schedule_name}:\n  {schedule_text}"

    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
