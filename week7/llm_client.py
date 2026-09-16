"""Shared Groq LLM client: chat-completion call with retry, and a converter
from the project's Gemini-style tool declarations (TOOL_DECLARATIONS in
week7/tools.py) to Groq/OpenAI-style tool schemas.

Replaces the earlier Gemini (google-genai) client used across week7/week8.
"""

import os
import re
import time

from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def get_client() -> Groq:
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def to_groq_tools(declarations: list[dict]) -> list[dict]:
    """Convert Gemini-style function declarations (type: OBJECT/STRING, ...)
    into Groq/OpenAI-style {"type": "function", "function": {...}} tools."""

    def _lower_types(node):
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                if k == "type" and isinstance(v, str):
                    out[k] = v.lower()
                else:
                    out[k] = _lower_types(v)
            return out
        if isinstance(node, list):
            return [_lower_types(v) for v in node]
        return node

    tools = []
    for decl in declarations:
        tools.append({
            "type": "function",
            "function": {
                "name": decl["name"],
                "description": decl["description"],
                "parameters": _lower_types(decl["parameters"]),
            },
        })
    return tools


def groq_call_with_retry(client, *, model, messages, tools=None,
                          max_retries=5, base_delay=5.0, log=None):
    """Call client.chat.completions.create with retry on rate-limit (429)
    and transient server errors (5xx)."""
    for attempt in range(1, max_retries + 1):
        try:
            kwargs = {"model": model, "messages": messages}
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            return client.chat.completions.create(**kwargs)

        except Exception as exc:
            err_msg = str(exc)
            is_retryable = (
                "429" in err_msg or "rate_limit" in err_msg.lower()
                or "500" in err_msg or "502" in err_msg
                or "503" in err_msg or "504" in err_msg
                or "overloaded" in err_msg.lower()
            )
            if not is_retryable or attempt == max_retries:
                raise

            match = re.search(r"try again in ([\d.]+)s", err_msg, re.IGNORECASE)
            if match:
                delay = float(match.group(1)) + 1.0
            else:
                delay = base_delay * (2 ** (attempt - 1))

            msg = f"[RETRY] Attempt {attempt}/{max_retries} hit retryable error. Waiting {delay:.1f}s ..."
            if log is not None:
                log.append(msg)
            else:
                print(f"  {msg}")
            time.sleep(delay)

    raise RuntimeError("groq_call_with_retry: exhausted retries")
