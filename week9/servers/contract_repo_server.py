"""MCP server simulating the knowledge team's contract-repository server.

This is "their" server — third-party from our agent's point of view. It
models the same underlying contract (MSA-2026-014 + two amendments) but
addresses it the way a repository/registry system would: by contract_id,
not by the version-key scheme week7/contracts.py uses internally. This is
realistic: two systems built by different teams model the same reality
differently.

Run standalone for manual testing:
    python -m week9.servers.contract_repo_server
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("contract_repository")

# Repository's own view of the world -- deliberately NOT the same shape as
# week7.contracts.CONTRACTS, to keep this a believable independent system.
CONTRACT_REPO = {
    "MSA-2026-014": {
        "title": "Master Service Agreement MSA-2026-014",
        "parties": ["Northstar Technologies Pvt. Ltd.", "Meridian Software Services Pvt. Ltd."],
        "effective_date": "2026-01-01",
        "amendments": [
            {"id": "A1", "title": "Amendment No. 1", "effective_date": "2026-04-01",
             "summary": "Reduced termination notice 60->30 Business Days; extended payment window 30->45 days."},
            {"id": "A2", "title": "Amendment No. 2", "effective_date": "2026-07-01",
             "summary": "Reduced termination notice 30->15 Business Days; confirmed 45-day payment window."},
        ],
    },
}


@mcp.tool()
def lookup_contract(contract_id: str) -> str:
    """Look up a contract's basic metadata (title, parties, effective date)
    by its repository contract_id (e.g. 'MSA-2026-014').

    If the contract_id is not found, the result lists every contract_id
    that DOES exist in the repository, so the caller can retry with a
    valid id instead of assuming the repository is down.
    """
    record = CONTRACT_REPO.get(contract_id)
    if not record:
        return (
            f"No contract with id '{contract_id}' exists in the repository. "
            f"Known contract ids: {list(CONTRACT_REPO.keys())}"
        )
    return (
        f"Contract ID: {contract_id}\n"
        f"Title: {record['title']}\n"
        f"Parties: {', '.join(record['parties'])}\n"
        f"Effective Date: {record['effective_date']}\n"
        f"Amendments on file: {len(record['amendments'])}"
    )


@mcp.tool()
def get_contract_effective_date(contract_id: str) -> str:
    """Return the effective date for a contract_id from the repository's
    own record (independent of week7's version-key based lookup).

    If the contract_id is not found, says so explicitly and lists the
    valid ids -- distinct from the case where the id is valid but a
    requested amendment doesn't exist yet (see get_amendment_chain).
    """
    record = CONTRACT_REPO.get(contract_id)
    if not record:
        return (
            f"No contract with id '{contract_id}' exists in the repository "
            f"(this is a repository lookup miss, not a system outage). "
            f"Known contract ids: {list(CONTRACT_REPO.keys())}"
        )
    return f"{contract_id} effective date: {record['effective_date']}"


@mcp.tool()
def get_amendment_chain(contract_id: str) -> str:
    """Return the ordered chain of amendments on file for a contract_id,
    each with its id, title, effective date, and a one-line summary of
    what it changed.

    If the contract_id is unknown, lists valid ids. If the contract_id is
    valid but has zero amendments on file, says so explicitly rather than
    returning an empty/ambiguous result -- so the model can distinguish
    "no amendments exist" from "lookup failed".
    """
    record = CONTRACT_REPO.get(contract_id)
    if not record:
        return (
            f"No contract with id '{contract_id}' exists in the repository. "
            f"Known contract ids: {list(CONTRACT_REPO.keys())}"
        )
    if not record["amendments"]:
        return f"{contract_id} has zero amendments on file (this is the original, unamended contract)."

    lines = [f"Amendment chain for {contract_id} ({len(record['amendments'])} amendment(s)), in order:"]
    for a in record["amendments"]:
        lines.append(f"  {a['id']} - {a['title']} (effective {a['effective_date']}): {a['summary']}")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
