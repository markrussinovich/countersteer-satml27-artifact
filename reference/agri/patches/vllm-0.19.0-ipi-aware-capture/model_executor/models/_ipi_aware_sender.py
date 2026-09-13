"""IPI-Aware hidden-state push sender for serving mode.

Runs a background daemon thread that POSTs captured hidden states
to a consumer's callback URL. Registration is read from a JSON file
written by the /ipi-aware/register API endpoint.

Env vars:
    IPI_AWARE_HOOK_FILE  - path to registration JSON (default /tmp/ipi-aware_hook.json)

Registration JSON format:
    {"callback_url": "http://host:port", "layers": [16, 18], "position_offset": -1}
"""
from __future__ import annotations

import base64
import io
import json
import os
import queue
import threading
import urllib.request
import urllib.error
import numpy as np


class IPIAwareSender:
    _instance: IPIAwareSender | None = None

    def __init__(self):
        self._queue: queue.Queue = queue.Queue(maxsize=1024)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._registration: dict | None = None
        self._reg_mtime: float = 0.0
        self._reg_path: str = os.environ.get("IPI_AWARE_HOOK_FILE", "/tmp/ipi-aware_hook.json")
        self._consecutive_failures: int = 0
        self._max_failures: int = 5  # after this many, clear queue + invalidate
        self._failed_callback_url: str | None = None  # avoid re-reg with dead consumer

    def start(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._sender_loop, daemon=True)
            self._thread.start()

    def stop(self):
        with self._lock:
            if self._thread is not None:
                self._queue.put(None)  # sentinel
                self._thread.join(timeout=25)
                self._thread = None

    def load_registration(self):
        """Read registration file if it has changed since last read.

        Uses mtime to skip re-reads. Reads file first, then captures
        mtime after a successful parse to avoid stale-mtime races.
        """
        try:
            mtime = os.path.getmtime(self._reg_path)
        except OSError:
            self._registration = None
            self._reg_mtime = 0.0
            return
        if mtime == self._reg_mtime:
            return
        # File was rewritten — clear failure blacklist (consumer likely restarted)
        self._failed_callback_url = None
        try:
            with open(self._reg_path) as f:
                reg = json.load(f)
            # New registration — drain stale queue items and reset failures
            old_callback = (self._registration or {}).get("callback_url", "")
            new_callback = reg.get("callback_url", "")
            if new_callback != old_callback:
                self._drain_queue()
            self._registration = reg
            self._reg_mtime = mtime
            self._consecutive_failures = 0
        except (json.JSONDecodeError, OSError):
            pass  # keep existing registration

    def has_registration(self) -> bool:
        return self._registration is not None

    def get_position_offset(self) -> int | None:
        """Return the registered position offset, or None if not registered."""
        if self._registration is None:
            return None
        return self._registration.get("position_offset")

    def get_layers(self) -> list[int] | None:
        """Return the registered capture layers, or None if not registered."""
        if self._registration is None:
            return None
        return self._registration.get("layers")

    def enqueue(self, request_id: str, hs_tensor, layer_ids: list[int]):
        """Non-blocking enqueue. Clones tensor and defers serialization to sender."""
        try:
            # Clone to decouple from the GPU worker's tensor lifecycle.
            # Convert via float32 to handle bfloat16 (numpy has no bfloat16).
            arr = np.array(hs_tensor.float(), dtype=np.float16)
            item = (request_id, arr, layer_ids)
        except Exception:
            return  # never let this crash the forward pass
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            pass

    def _sender_loop(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            self._post(item)

    def _drain_queue(self):
        """Remove all pending items from the queue."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _post(self, item: tuple):
        """Serialize numpy array and POST to consumer. Runs on sender thread."""
        reg = self._registration
        if reg is None:
            return
        callback_url = reg.get("callback_url", "")
        if not callback_url:
            return
        request_id, arr, layer_ids = item
        try:
            buf = io.BytesIO()
            np.save(buf, arr)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            payload = json.dumps({
                "request_id": request_id,
                "layer_ids": layer_ids,
                "hidden_states_b64": b64,
            }).encode("utf-8")
        except Exception:
            return  # serialization failure — drop this item
        url = callback_url.rstrip("/") + "/ipi-aware/hidden-states"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(2):
            try:
                urllib.request.urlopen(req, timeout=10)
                self._consecutive_failures = 0
                return
            except Exception:
                if attempt == 1:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= self._max_failures:
                        failed_url = callback_url
                        self._drain_queue()
                        self._registration = None
                        self._reg_mtime = 0.0
                        self._failed_callback_url = failed_url
                    return


_sender: IPIAwareSender | None = None
_singleton_lock = threading.Lock()


def get_sender() -> IPIAwareSender:
    global _sender
    with _singleton_lock:
        if _sender is None:
            _sender = IPIAwareSender()
    return _sender


def start_sender():
    get_sender().start()


def stop_sender():
    global _sender
    with _singleton_lock:
        if _sender is not None:
            _sender.stop()
            _sender = None
