"""Contract corpus for the agent vs workflow race.

Contains a master service agreement with two amendments, defined terms,
and schedules that create dependency chains between tools (clause → defined
term → schedule).
"""

# ---------------------------------------------------------------------------
# Master Agreement + Amendments
# ---------------------------------------------------------------------------

CONTRACTS = {
    "v1_master": {
        "name": "Master Service Agreement MSA-2026-014",
        "effective_date": "2026-01-01",
        "execution_date": "2025-12-15",
        "expiration_date": "2027-12-31",
        "parties": {
            "client": "Northstar Technologies Pvt. Ltd.",
            "provider": "Meridian Software Services Pvt. Ltd.",
        },
        "clauses": {
            "Section 4.1 - Payment Terms": (
                "The Client shall pay all undisputed invoices within thirty (30) "
                "calendar days from the invoice date. Payment shall be calculated "
                "on the basis of each Service Period as defined in Schedule A."
            ),
            "Section 4.2 - Late Payment Interest": (
                "Late payments may incur interest at one percent (1%) per month, "
                "calculated from the due date until the date of payment, subject "
                "to applicable law."
            ),
            "Section 6 - Confidentiality": (
                "Each party agrees to hold the other party's Confidential Information "
                "in strict confidence and not to disclose it to any third party. "
                "Confidentiality obligations survive termination for three (3) years."
            ),
            "Section 7.3 - Force Majeure": (
                "Neither party shall be liable for delay or failure caused by events "
                "beyond its reasonable control, including natural disasters, war, "
                "government action, or widespread network outages."
            ),
            "Section 8.2 - Termination for Convenience": (
                "Either party may terminate the Agreement for convenience by providing "
                "sixty (60) Business Days' written notice to the other party. "
                "Termination shall take effect at the end of the applicable Service "
                "Period following the notice period."
            ),
            "Section 11.4 - Intellectual Property": (
                "Project-specific deliverables created and paid for by the Customer "
                "shall be assigned to the Customer after full payment, unless the "
                "applicable SOW states otherwise."
            ),
        },
    },
    "v2_amendment_1": {
        "name": "Amendment No. 1 to MSA-2026-014",
        "effective_date": "2026-04-01",
        "execution_date": "2026-03-20",
        "expiration_date": "2027-12-31",
        "parties": {
            "client": "Northstar Technologies Pvt. Ltd.",
            "provider": "Meridian Software Services Pvt. Ltd.",
        },
        "clauses": {
            "Section 1 - Termination Notice Period": (
                "Section 8.2 of the Master Agreement is hereby amended. The "
                "termination notice period is reduced from sixty (60) Business "
                "Days to thirty (30) Business Days' written notice."
            ),
            "Section 3 - Payment Terms": (
                "Section 4.1 is amended to extend the payment window from thirty "
                "(30) to forty-five (45) calendar days for undisputed invoices. "
                "Payment continues to be calculated per Service Period."
            ),
        },
    },
    "v3_amendment_2": {
        "name": "Amendment No. 2 to MSA-2026-014",
        "effective_date": "2026-07-01",
        "execution_date": "2026-06-25",
        "expiration_date": "2027-12-31",
        "parties": {
            "client": "Northstar Technologies Pvt. Ltd.",
            "provider": "Meridian Software Services Pvt. Ltd.",
        },
        "clauses": {
            "Section 1 - Termination Notice Period": (
                "Section 8.2 of the Master Agreement, as previously amended by "
                "Amendment No. 1, is hereby further amended. The termination "
                "notice period is reduced to fifteen (15) Business Days' written "
                "notice. This supersedes all prior versions of Section 8.2."
            ),
            "Section 2 - Payment Terms Confirmation": (
                "The payment terms of forty-five (45) calendar days as established "
                "in Amendment No. 1 are confirmed and remain in effect."
            ),
        },
    },
}


# ---------------------------------------------------------------------------
# Defined Terms — some reference Schedules (creating the dependency chain)
# ---------------------------------------------------------------------------

DEFINED_TERMS = {
    "v1_master": {
        "Business Day": (
            "Any day other than a Saturday, Sunday, or public holiday observed "
            "in the jurisdiction specified in Section 15 (Governing Law). For "
            "the avoidance of doubt, the holiday calendar applicable to this "
            "Agreement is set forth in Schedule B (Holiday Calendar)."
        ),
        "Confidential Information": (
            "All non-public information, whether written, oral, or electronic, "
            "disclosed by either party that is marked as confidential or that a "
            "reasonable person would understand to be confidential given its "
            "nature and the circumstances of disclosure. Confidential Information "
            "includes trade secrets, business plans, customer lists, and "
            "technical data."
        ),
        "Service Period": (
            "The monthly billing cycle during which services are rendered, as "
            "defined in Schedule A (Service Level Agreement). Each Service "
            "Period begins on the first calendar day of the month and ends on "
            "the last calendar day of the month."
        ),
        "Material Breach": (
            "A failure by either party to perform a material obligation under "
            "this Agreement that remains uncured for thirty (30) days after "
            "written notice specifying the breach."
        ),
        "Effective Date": (
            "The date on which this Agreement becomes binding, as stated in the "
            "preamble. The Effective Date is 1 January 2026."
        ),
    },
    "v2_amendment_1": {
        "Business Day": (
            "Same definition as in the Master Agreement (Section 1.2). "
            "No changes to the Business Day definition in this amendment."
        ),
        "Service Period": (
            "Same definition as in the Master Agreement. "
            "No changes to the Service Period definition in this amendment."
        ),
    },
    "v3_amendment_2": {
        "Business Day": (
            "Same definition as in the Master Agreement (Section 1.2). "
            "No changes to the Business Day definition in this amendment."
        ),
        "Service Period": (
            "Same definition as in the Master Agreement. "
            "No changes to the Service Period definition in this amendment."
        ),
    },
}


# ---------------------------------------------------------------------------
# Schedules — referenced by defined terms
# ---------------------------------------------------------------------------

SCHEDULES = {
    "Schedule A": (
        "Service Level Agreement — The Provider shall maintain 99.5% uptime "
        "for all production systems. Each Service Period runs from the 1st to "
        "the last day of each calendar month. Scheduled maintenance windows are "
        "Sundays 02:00-06:00 IST. Penalties for downtime exceeding the SLA: "
        "5% credit per hour of unplanned downtime, capped at 30% of monthly fees."
    ),
    "Schedule B": (
        "Holiday Calendar — The following public holidays are observed and "
        "excluded from Business Day calculations: Republic Day (26 Jan), "
        "Independence Day (15 Aug), Gandhi Jayanti (2 Oct), Diwali (variable), "
        "Christmas Day (25 Dec), New Year's Day (1 Jan), and Tamil Nadu state "
        "holidays as published annually by the Government of Tamil Nadu."
    ),
}
