"""Langfuse observability — SDK v3 / langfuse ≥ 5.x.

Public API used by the rest of the codebase:
    from core.observability import observe, lf_client, LANGFUSE_ENABLED

Reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST from env.
Degrades silently when keys are absent — @observe becomes a transparent
pass-through and lf_client exposes no-op methods.
"""

from __future__ import annotations

import os
from functools import wraps
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv()

_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
_HOST       = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

LANGFUSE_ENABLED = bool(_PUBLIC_KEY and _SECRET_KEY)

if LANGFUSE_ENABLED:
    os.environ["LANGFUSE_PUBLIC_KEY"] = _PUBLIC_KEY
    os.environ["LANGFUSE_SECRET_KEY"] = _SECRET_KEY
    os.environ["LANGFUSE_HOST"]       = _HOST

    from langfuse import observe, get_client   # noqa: F401

    lf_client = get_client()

else:
    # ── No-op shims ───────────────────────────────────────────────────────

    def observe(*args, **kwargs):                               # type: ignore[misc]
        """No-op @observe when Langfuse is disabled."""
        def decorator(fn: Callable) -> Callable:
            @wraps(fn)
            def wrapper(*a, **kw):
                return fn(*a, **kw)
            return wrapper
        if len(args) == 1 and callable(args[0]):
            return decorator(args[0])
        return decorator

    class _NullClient:
        def update_current_span(self, **_: Any) -> None:
            pass
        def update_current_generation(self, **_: Any) -> None:
            pass
        def get_current_trace_id(self) -> str | None:
            return None
        def flush(self) -> None:
            pass

    lf_client = _NullClient()                                  # type: ignore[assignment]


def flush() -> None:
    """Flush pending events to Langfuse."""
    try:
        lf_client.flush()
    except Exception:
        pass
