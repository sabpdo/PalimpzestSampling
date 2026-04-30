"""Optional tracing via the ``PALIMPZEST_TRACE`` environment variable.

Comma-separated tokens: ``llm``, ``sentinel``, or ``all``. The value ``1`` enables
both LLM and sentinel sampling traces.

Example::

    export PALIMPZEST_TRACE=llm,sentinel
"""

from __future__ import annotations

import os


def _trace_parts() -> set[str]:
    raw = os.environ.get("PALIMPZEST_TRACE", "").lower().strip()
    if not raw:
        return set()
    if raw == "1":
        return {"all"}
    return {p.strip() for p in raw.split(",") if p.strip()}


def palimpzest_trace_llm(verbose: bool = False) -> bool:
    """True if LLM completions / reasoning should be logged at INFO."""
    if verbose:
        return True
    parts = _trace_parts()
    return bool(parts.intersection({"llm", "all"}))


def palimpzest_trace_sentinel() -> bool:
    """True if sentinel operator sampling decisions should be logged at INFO."""
    parts = _trace_parts()
    return bool(parts.intersection({"sentinel", "all"}))
