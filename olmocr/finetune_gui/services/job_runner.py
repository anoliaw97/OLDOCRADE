"""
Thread-based job runner with log streaming, progress tracking, and cancellation.

Usage pattern in Gradio:
    runner = JobRunner()

    def on_button_click(...):
        # This generator both starts the job and streams logs - works with Gradio streaming.
        yield from runner.run_and_stream(my_service_fn, arg1, arg2, kwarg=val)
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable, Generator, Optional


class JobRunner:
    """
    Manages a single background job with:
    - cancellation via threading.Event
    - line-by-line log streaming via a queue
    - (current, total) progress counters
    """

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()
        self._log_queue: queue.Queue[Optional[str]] = queue.Queue()
        self._running = False
        self.progress: tuple[int, int] = (0, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    def cancel(self) -> None:
        """Signal the running job to stop."""
        self._cancel_event.set()

    def run_and_stream(
        self,
        fn: Callable[..., None],
        *args: Any,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """
        Start *fn* in a daemon thread, then yield accumulated log text until
        the job finishes.  Designed to be used as a Gradio generator.

        *fn* receives two extra keyword arguments:
            cancel_event: threading.Event — set when user cancels
            log_fn:       Callable[[str], None] — call to emit a log line
            progress_fn:  Callable[[int, int], None] — call with (current, total)
        """
        if self._running:
            yield "⚠️  A job is already running. Cancel it first.\n"
            return

        # Reset state
        self._cancel_event.clear()
        self._log_queue = queue.Queue()
        self._running = True
        self.progress = (0, 1)

        def _wrapper() -> None:
            try:
                fn(
                    *args,
                    cancel_event=self._cancel_event,
                    log_fn=self._emit,
                    progress_fn=self._set_progress,
                    **kwargs,
                )
            except Exception as exc:
                self._emit(f"[ERROR] {exc}")
            finally:
                self._running = False
                self._log_queue.put(None)  # sentinel

        self._thread = threading.Thread(target=_wrapper, daemon=True)
        self._thread.start()

        # Stream log lines until the sentinel arrives
        accumulated = ""
        while True:
            try:
                msg = self._log_queue.get(timeout=0.25)
                if msg is None:  # job finished
                    break
                accumulated += msg + "\n"
                yield accumulated
            except queue.Empty:
                if not self._running:
                    break
                yield accumulated  # heartbeat so Gradio keeps the connection alive

        yield accumulated  # final state

    # ------------------------------------------------------------------
    # Internal helpers (called from worker thread)
    # ------------------------------------------------------------------

    def _emit(self, message: str) -> None:
        self._log_queue.put(message)

    def _set_progress(self, current: int, total: int) -> None:
        self.progress = (current, total)
        self._emit(f"[{current}/{total}]  …")
