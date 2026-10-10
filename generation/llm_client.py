"""Groq LLM client: chat-completion call with retry."""

import os
import re
import time

from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def get_client() -> Groq:
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


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
