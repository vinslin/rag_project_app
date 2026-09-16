"""Three non-overlapping tools for contract analysis.

Each tool has exactly one job and does not overlap with the others:
  search_clause     — finds clause TEXT by name/number
  get_effective_date — returns DATES by contract version (enum)
  get_definitions   — resolves DEFINED TERMS by name + version (enum)
"""

from enum import Enum
from week7.contracts import CONTRACTS, DEFINED_TERMS, SCHEDULES


# ---------------------------------------------------------------------------
# Typed enum for contract versions
# ---------------------------------------------------------------------------

class ContractVersion(str, Enum):
    """Contract versions available for querying."""
    V1_MASTER = "v1_master"
    V2_AMENDMENT_1 = "v2_amendment_1"
    V3_AMENDMENT_2 = "v3_amendment_2"


# ---------------------------------------------------------------------------
# Tool 1: search_clause
# ---------------------------------------------------------------------------

def search_clause(clause_name: str) -> str:
    """Locate a specific clause by name or section number and return its full
    text. Searches across ALL contract versions (master + amendments)."""

    results = []
    query = clause_name.lower().strip()

    for version_key, contract in CONTRACTS.items():
        for clause_key, clause_text in contract["clauses"].items():
            # Match if query appears in clause key OR any significant word matches
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
        return f"No clause matching '{clause_name}' found in any contract version."

    parts = []
    for r in results:
        parts.append(
            f"[{r['version']}] {r['contract']}\n"
            f"  {r['clause']}:\n"
            f"  {r['text']}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Tool 2: get_effective_date
# ---------------------------------------------------------------------------

def get_effective_date(contract_version: str) -> str:
    """Return the effective date and temporal metadata (execution date,
    effective date, expiration date) for a specific contract version.

    Does NOT return clause text or defined terms — use the other tools
    for those.
    """
    version = (contract_version.value
               if isinstance(contract_version, ContractVersion)
               else contract_version)

    if version not in CONTRACTS:
        valid = [v.value for v in ContractVersion]
        return f"Unknown contract version: {version}. Valid: {valid}"

    c = CONTRACTS[version]
    return (
        f"Contract: {c['name']}\n"
        f"  Execution Date: {c['execution_date']}\n"
        f"  Effective Date:  {c['effective_date']}\n"
        f"  Expiration Date: {c['expiration_date']}"
    )


# ---------------------------------------------------------------------------
# Tool 3: get_definitions  (NEW — the third tool)
# ---------------------------------------------------------------------------

def get_definitions(term_name: str, contract_version: str) -> str:
    """Resolve a defined contractual term to its full definition, following
    any references to schedules.

    Looks up the term in the specified contract version's definitions and
    automatically appends the text of any Schedule referenced in the
    definition.

    Does NOT return clause text or dates — use the other tools for those.
    """
    version = (contract_version.value
               if isinstance(contract_version, ContractVersion)
               else contract_version)

    if version not in DEFINED_TERMS:
        return f"No definitions found for version: {version}"

    terms = DEFINED_TERMS[version]
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
        return (f"Term '{term_name}' not found in {version} definitions. "
                f"Available terms: {available}")

    result = (f"Term: {matched_term}\n"
              f"Version: {version}\n"
              f"Definition: {matched_definition}")

    # Follow schedule references automatically
    for schedule_name, schedule_text in SCHEDULES.items():
        if schedule_name.lower() in matched_definition.lower():
            result += f"\n\nReferenced {schedule_name}:\n  {schedule_text}"

    return result


# ---------------------------------------------------------------------------
# Dispatch map
# ---------------------------------------------------------------------------

TOOL_FUNCTIONS = {
    "search_clause": search_clause,
    "get_effective_date": get_effective_date,
    "get_definitions": get_definitions,
}


# ---------------------------------------------------------------------------
# Gemini function declarations for the agent's tool-calling API
# ---------------------------------------------------------------------------

TOOL_DECLARATIONS = [
    {
        "name": "search_clause",
        "description": (
            "Locate a specific clause by name or section number and return "
            "its full text. Searches across all contract versions (master "
            "agreement and amendments). Use this to find what a clause says."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "clause_name": {
                    "type": "STRING",
                    "description": (
                        "The name or section number of the clause to find "
                        "(e.g. 'Section 8.2', 'termination', 'payment terms')"
                    ),
                },
            },
            "required": ["clause_name"],
        },
    },
    {
        "name": "get_effective_date",
        "description": (
            "Return the effective date and temporal metadata (execution date, "
            "effective date, expiration date) for a specific contract version. "
            "Use this to find when a contract or amendment took effect."
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
    {
        "name": "get_definitions",
        "description": (
            "Resolve a defined contractual term to its full definition, "
            "following any references to schedules. Use this to find what "
            "a defined term means (e.g. 'Business Day', 'Service Period')."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "term_name": {
                    "type": "STRING",
                    "description": (
                        "The defined term to look up "
                        "(e.g. 'Business Day', 'Service Period')"
                    ),
                },
                "contract_version": {
                    "type": "STRING",
                    "description": (
                        "Which contract version's definitions to check"
                    ),
                    "enum": ["v1_master", "v2_amendment_1", "v3_amendment_2"],
                },
            },
            "required": ["term_name", "contract_version"],
        },
    },
]
