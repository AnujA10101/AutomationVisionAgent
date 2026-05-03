"""Shared stop flag for UI, agent loop, and executor."""

from __future__ import annotations

import threading


class AgentController:
    """Thread-safe emergency stop for multi-step agent runs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop_requested = False

    def request_stop(self) -> None:
        with self._lock:
            self._stop_requested = True

    def reset(self) -> None:
        with self._lock:
            self._stop_requested = False

    def should_stop(self) -> bool:
        with self._lock:
            return self._stop_requested
