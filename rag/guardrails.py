import re

# Patterns that indicate prompt injection attempts
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|rules|prompts)",
    r"disregard\s+(all\s+)?(previous|above|prior)",
    r"you\s+are\s+now\s+a",
    r"act\s+as\s+(a\s+)?",
    r"pretend\s+(you\s+are|to\s+be)",
    r"forget\s+(all\s+)?(previous|your)\s+(instructions|rules)",
    r"override\s+(your\s+)?(instructions|rules|system)",
    r"new\s+instructions?\s*:",
    r"system\s*prompt\s*:",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"DAN\s+mode",
]

# Patterns that indicate the user is asking the model to draft new contract language
DRAFTING_PATTERNS = [
    r"\b(draft|write|compose|create|generate|prepare)\b.{0,30}\b(clause|provision|section|contract|agreement|amendment|addendum|term|language)\b",
    r"\b(suggest|propose|recommend)\b.{0,30}\b(new|revised|updated|alternative)\b.{0,30}\b(clause|wording|language|provision)\b",
    r"\bwrite\s+me\b",
    r"\bdraft\s+a\b",
]


def _matches_any(text, patterns):
    """Return True if text matches any of the compiled regex patterns."""
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def screen_query(query):
  
    if not query or not query.strip():
        return False, "empty"

    if _matches_any(query, INJECTION_PATTERNS):
        return False, "injection"

    if _matches_any(query, DRAFTING_PATTERNS):
        return False, "drafting"

    return True, None
