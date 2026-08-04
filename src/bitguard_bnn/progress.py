"""Dependency-free terminal progress rendering for setup and training."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Mapping
from typing import TextIO


class TerminalProgress:
    """Render one in-place progress line without contaminating stdout JSON."""

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        width: int = 28,
        min_interval: float = 0.1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.stream = sys.stderr if stream is None else stream
        self.width = max(int(width), 10)
        self.min_interval = max(float(min_interval), 0.0)
        self.clock = clock
        try:
            self.enabled = bool(self.stream.isatty())
        except (AttributeError, OSError):
            self.enabled = False
        self._last_rendered_at = -float("inf")
        self._last_width = 0
        self._active = False

    def __call__(self, event: Mapping[str, object]) -> None:
        if not self.enabled:
            return
        status = str(event.get("status", "advanced"))
        completed = self._count(event.get("completed"))
        total = self._count(event.get("total"))
        if total <= 0:
            return
        completed = min(max(completed, 0), total)
        now = self.clock()
        final = status in {"completed", "failed"} and completed >= total
        if not final and now - self._last_rendered_at < self.min_interval:
            return
        self._last_rendered_at = now
        ratio = completed / total
        filled = min(self.width, int(ratio * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        percentage = int(ratio * 100)
        label = self._label(event)
        suffix = " reused" if status == "reused" else ""
        line = (
            f"[{bar}] {percentage:3d}% {label} "
            f"({completed:,}/{total:,}){suffix}"
        )
        padded = line.ljust(self._last_width)
        try:
            self.stream.write("\r" + padded)
            if final:
                self.stream.write("\n")
            self.stream.flush()
        except (BrokenPipeError, OSError, ValueError):
            self.enabled = False
            return
        self._last_width = len(line)
        self._active = not final

    def close(self) -> None:
        if not self.enabled or not self._active:
            return
        try:
            self.stream.write("\n")
            self.stream.flush()
        except (BrokenPipeError, OSError, ValueError):
            self.enabled = False
        self._active = False

    @staticmethod
    def _count(value: object) -> int:
        if isinstance(value, bool):
            return 0
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return 0

    @staticmethod
    def _label(event: Mapping[str, object]) -> str:
        if event.get("scope") == "training":
            dataset = str(event.get("dataset") or "training")
            role = str(event.get("role") or "main")
            epoch = TerminalProgress._count(event.get("epoch"))
            epochs = TerminalProgress._count(event.get("epochs"))
            return f"{dataset}/{role} epoch {epoch}/{epochs}"
        stage = str(event.get("stage") or "setup")
        return f"setup {stage}"
