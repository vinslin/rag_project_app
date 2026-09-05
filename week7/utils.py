"""Shared utilities — retry wrapper for Gemini API calls with rate-limit handling."""

import re
import time


def gemini_call_with_retry(client, *, model, contents, config=None,
                           max_retries=5, base_delay=5.0, log=None):
    """Call client.models.generate_content with automatic retry on 429 errors.

    Parses the server-suggested retry delay from the error message when
    available and waits accordingly.  Falls back to exponential backoff.
    """
    for attempt in range(1, max_retries + 1):
        try:
            kwargs = {"model": model, "contents": contents}
            if config is not None:
                kwargs["config"] = config
            return client.models.generate_content(**kwargs)

        except Exception as exc:
            err_msg = str(exc)
            is_rate_limit = ("429" in err_msg
                             or "RESOURCE_EXHAUSTED" in err_msg
                             or "rate" in err_msg.lower())

            if not is_rate_limit or attempt == max_retries:
                raise  # re-raise non-rate-limit errors or final attempt

            # Try to parse the server-suggested retry delay
            match = re.search(r"retry\s*(?:in|Delay['\"]:\s*['\"])\s*([\d.]+)", err_msg, re.IGNORECASE)
            if match:
                delay = float(match.group(1)) + 1.0   # add 1 s buffer
            else:
                delay = base_delay * (2 ** (attempt - 1))  # exponential backoff

            if log is not None:
                log.append(f"[RETRY] Attempt {attempt}/{max_retries} hit rate limit. "
                           f"Waiting {delay:.1f}s ...")
            else:
                print(f"  [RETRY] Rate-limited, waiting {delay:.1f}s "
                      f"(attempt {attempt}/{max_retries}) ...")
            time.sleep(delay)

    # Should not reach here, but just in case
    raise RuntimeError("gemini_call_with_retry: exhausted retries")
