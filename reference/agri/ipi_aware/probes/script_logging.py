"""Logging helpers for probe pipeline scripts."""
from __future__ import annotations

import sys
import time
from typing import Any, Callable, Mapping


def stderr_log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{remainder:04.1f}s"
    hours, remainder_minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(remainder_minutes)}m{remainder:04.1f}s"


def log_stage(
    log_fn: Callable[[str], Any] | None,
    stage_index: int,
    stage_total: int,
    title: str,
) -> None:
    if callable(log_fn):
        log_fn(f"Stage {stage_index}/{stage_total}: {title}")


def log_progress(
    log_fn: Callable[[str], Any] | None,
    label: str,
    completed: int,
    total: int,
    start_time: float,
    unit: str = "items",
    extra: Mapping[str, Any] | None = None,
) -> None:
    if not callable(log_fn):
        return
    elapsed = max(time.monotonic() - start_time, 1e-6)
    percentage = (completed / total * 100.0) if total else 100.0
    rate = (completed / elapsed) if completed else 0.0
    message = (
        f"{label}: {completed}/{total} "
        f"({percentage:.1f}%, {rate:.2f} {unit}/s, elapsed {format_elapsed(elapsed)})"
    )
    if extra:
        message += " " + " ".join(f"{k}={v}" for k, v in extra.items())
    log_fn(message)
