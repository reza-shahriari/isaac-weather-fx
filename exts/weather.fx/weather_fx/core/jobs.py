"""Run slow work off the UI thread, keeping only the newest request.

A cloud field takes seconds to synthesise and a dome seconds to bake. Doing either on Kit's
update thread froze the app for as long as it took, once per slider tick. :class:`LatestJob`
runs one job at a time in a worker thread; submitting while one runs *replaces* the pending job
rather than queueing behind it, so dragging a slider costs one job for where it started and one
for where it stopped, not one per frame in between.

Results are collected by :meth:`poll` from the caller's own thread, which in Kit is the update
loop, because the USD stage may only be written from there. Pure Python, so it is unit-testable.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional, Tuple

log = logging.getLogger("weather_fx")

__all__ = ["LatestJob"]


class LatestJob:
    """One worker, one pending slot, newest wins."""

    def __init__(self, name: str = "weather_fx-job"):
        self._name = name
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._pending: Optional[Tuple[Any, Callable[[], Any]]] = None
        self._done: Optional[Tuple[Any, Any, Optional[BaseException]]] = None
        self._running_key: Any = None
        self._generation = 0

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._thread is not None or self._pending is not None

    @property
    def running_key(self) -> Any:
        with self._lock:
            return self._running_key

    def submit(self, key: Any, fn: Callable[[], Any]) -> None:
        """Run ``fn`` in the background. A job already pending (not yet started) is dropped."""
        with self._lock:
            self._pending = (key, fn)
            if self._thread is None:
                self._start_locked()

    def cancel(self) -> None:
        """Forget the pending job and any result not yet collected. A running job finishes but its
        result is discarded."""
        with self._lock:
            self._pending = None
            self._done = None
            self._generation += 1

    def poll(self) -> Optional[Tuple[Any, Any]]:
        """``(key, result)`` of a finished job, once, or ``None``. Re-raises a job's exception."""
        with self._lock:
            done, self._done = self._done, None
        if done is None:
            return None
        key, result, error = done
        if error is not None:
            raise error
        return key, result

    def wait(self, timeout: Optional[float] = None) -> Optional[Tuple[Any, Any]]:
        """Block until the worker is idle, then :meth:`poll`. For tests and deterministic runs."""
        while True:
            with self._lock:
                thread = self._thread
            if thread is None:
                break
            thread.join(timeout)
            if timeout is not None and thread.is_alive():
                return None
        return self.poll()

    # --- worker ------------------------------------------------------------------------

    def _start_locked(self) -> None:
        key, fn = self._pending  # type: ignore[misc]
        self._pending = None
        self._running_key = key
        generation = self._generation
        self._thread = threading.Thread(
            target=self._run, args=(key, fn, generation), name=self._name, daemon=True
        )
        self._thread.start()

    def _run(self, key: Any, fn: Callable[[], Any], generation: int) -> None:
        result, error = None, None
        try:
            result = fn()
        except BaseException as exc:  # noqa: BLE001 -- handed back to the caller by poll()
            error = exc
        with self._lock:
            if generation == self._generation:
                # A newer submit supersedes this result only if it has not started yet; keep the
                # finished one either way, because it is still newer than what is on screen.
                self._done = (key, result, error)
            self._thread = None
            self._running_key = None
            if self._pending is not None:
                self._start_locked()
