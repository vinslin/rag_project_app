"""Centralised Python logging configuration.

Call ``configure_logging()`` once at startup (in app.py) before importing
any other project module.  Every module that calls
``logging.getLogger(__name__)`` automatically inherits three handlers:

  data/logs/app.jsonl   — INFO and above, rotating 10 MB × 5 backups (JSON)
  data/logs/error.jsonl — ERROR and above, rotating 5 MB × 3 backups (JSON)
  stderr                — WARNING and above, human-readable

Trace propagation
-----------------
Call ``set_trace_id(tid)`` at the start of each request to bind a trace_id
to the current execution context via contextvars.  A logging.Filter reads it
and injects ``trace_id`` into every LogRecord automatically — no caller needs
to pass it through logger calls.

    import uuid
    from core.logging_setup import set_trace_id

    trace_id = str(uuid.uuid4())
    set_trace_id(trace_id)          # all logger.* calls below carry this id
"""

from __future__ import annotations

import logging
import logging.handlers
from contextvars import ContextVar
from pathlib import Path

try:
    from pythonjsonlogger.jsonlogger import JsonFormatter as _JsonFormatter  # v2
except ImportError:
    from pythonjsonlogger import JsonFormatter as _JsonFormatter  # v4+

LOG_DIR    = Path(__file__).parent.parent / "data" / "logs"
_APP_LOG   = LOG_DIR / "app.jsonl"
_ERROR_LOG = LOG_DIR / "error.jsonl"

_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")


def set_trace_id(tid: str) -> None:
    """Bind a trace_id to the current execution context. Call once per request."""
    _trace_id.set(tid)


def get_trace_id() -> str:
    """Return the trace_id bound to the current context, or '-' if unset."""
    return _trace_id.get()


class _TraceFilter(logging.Filter):
    """Injects the current context trace_id into every LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id.get()
        return True


def configure_logging(level: int = logging.INFO) -> None:
    """Wire all log handlers to the root logger.

    Idempotent — safe to call multiple times (Streamlit reruns the script).
    """
    root = logging.getLogger()
    if root.handlers:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    json_fmt = _JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s %(trace_id)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        rename_fields={
            "asctime":   "timestamp",
            "levelname": "level",
            "name":      "logger",
        },
    )
    trace_filter = _TraceFilter()

    # app.jsonl — INFO+, 10 MB × 5 backups
    app_h = logging.handlers.RotatingFileHandler(
        _APP_LOG, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8",
    )
    app_h.setLevel(logging.INFO)
    app_h.setFormatter(json_fmt)
    app_h.addFilter(trace_filter)

    # error.jsonl — ERROR+, 5 MB × 3 backups
    err_h = logging.handlers.RotatingFileHandler(
        _ERROR_LOG, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    err_h.setLevel(logging.ERROR)
    err_h.setFormatter(json_fmt)
    err_h.addFilter(trace_filter)

    # stderr — WARNING+, human-readable with trace_id suffix
    console_h = logging.StreamHandler()
    console_h.setLevel(logging.WARNING)
    console_h.setFormatter(
        logging.Formatter(
            fmt="%(levelname)-8s %(name)s  %(message)s  [trace=%(trace_id)s]",
        )
    )
    console_h.addFilter(trace_filter)

    root.setLevel(level)
    root.addHandler(app_h)
    root.addHandler(err_h)
    root.addHandler(console_h)
