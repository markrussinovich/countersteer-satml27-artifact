from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator, Mapping
from typing import Any


_REQUEST_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "ipi_aware_request_context",
    default=None,
)


def get_request_context() -> dict[str, Any] | None:
    value = _REQUEST_CONTEXT.get()
    if value is None:
        return None
    return dict(value)


@contextmanager
def active_request_context(context: Mapping[str, Any] | None) -> Iterator[None]:
    token = _REQUEST_CONTEXT.set(None if context is None else dict(context))
    try:
        yield
    finally:
        _REQUEST_CONTEXT.reset(token)
