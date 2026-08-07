"""Dependency-free terminal progress rendering for setup and training."""

from __future__ import annotations

import os
import re
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any, TextIO, cast


class TerminalProgress:
    """Render one in-place progress line without contaminating stdout JSON."""

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        width: int = 28,
        min_interval: float = 0.1,
        clock: Callable[[], float] = time.monotonic,
        max_line_width: int | None = None,
    ) -> None:
        self.stream = sys.stderr if stream is None else stream
        if max_line_width is None:
            try:
                max_line_width = os.get_terminal_size(self.stream.fileno()).columns
            except (AttributeError, OSError, ValueError):
                max_line_width = 120
        self.max_line_width = max(int(max_line_width), 40)
        available_bar_width = max(self.max_line_width - 70, 10)
        self.width = min(max(int(width), 10), available_bar_width)
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
        prefix = f"[{bar}] {percentage:3d}% "
        counter = f" ({completed:,}/{total:,}){suffix}"
        label = self._fit_label(
            event,
            label,
            max(self.max_line_width - len(prefix) - len(counter), 1),
        )
        line = f"{prefix}{label}{counter}"
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
            return int(cast(Any, value))
        except (TypeError, ValueError, OverflowError):
            return 0

    @staticmethod
    def _fit_label(
        event: Mapping[str, object],
        label: str,
        limit: int,
    ) -> str:
        if len(label) <= limit:
            return label
        if event.get("stage") == "inspect":
            dataset = str(event.get("dataset") or "inspect")
            phase = str(event.get("phase") or "working")
            path = str(event.get("relative_path") or "")
            filename = re.split(r"[\\/]", path)[-1] if path else ""
            file_index = TerminalProgress._count(event.get("file_index"))
            file_count = TerminalProgress._count(event.get("file_count"))
            if file_index <= 0 or file_count <= 0:
                file_index = TerminalProgress._count(event.get("dataset_index"))
                file_count = TerminalProgress._count(event.get("dataset_count"))
            details = []
            if file_index > 0 and file_count > 0:
                details.append(f"{file_index}/{file_count}")
            rows = TerminalProgress._count(event.get("rows"))
            if rows > 0:
                details.append(f"{rows:,} rows")

            prefix = f"{dataset} {phase}"
            suffix = " ".join(details)
            required = " ".join(item for item in (prefix, suffix) if item)
            if len(required) > limit:
                if suffix and len(suffix) < limit:
                    prefix_limit = max(limit - len(suffix) - 1, 1)
                    return (
                        TerminalProgress._middle_truncate(prefix, prefix_limit)
                        + " "
                        + suffix
                    )
                return TerminalProgress._middle_truncate(required, limit)

            filename_limit = limit - len(required) - (1 if required else 0)
            if filename and filename_limit > 0:
                compact_filename = TerminalProgress._tail_truncate(
                    filename,
                    filename_limit,
                )
                return " ".join(
                    item for item in (prefix, compact_filename, suffix) if item
                )
            return required
        return TerminalProgress._middle_truncate(label, limit)

    @staticmethod
    def _middle_truncate(value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        if limit <= 3:
            return value[:limit]
        remaining = limit - 3
        head = (remaining + 1) // 2
        tail = remaining - head
        return value[:head] + "..." + (value[-tail:] if tail else "")

    @staticmethod
    def _tail_truncate(value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        if limit <= 3:
            return value[-limit:]
        return "..." + value[-(limit - 3) :]

    @staticmethod
    def _label(event: Mapping[str, object]) -> str:
        if event.get("scope") == "training":
            dataset = str(event.get("dataset") or "training")
            role = str(event.get("role") or "main")
            epoch = TerminalProgress._count(event.get("epoch"))
            epochs = TerminalProgress._count(event.get("epochs"))
            return f"{dataset}/{role} epoch {epoch}/{epochs}"
        stage = str(event.get("stage") or "setup")
        inspect_dataset = event.get("dataset")
        phase = event.get("phase")
        if stage == "inspect" and inspect_dataset and phase:
            details = ["setup inspect", str(inspect_dataset), str(phase)]
            relative_path = event.get("relative_path")
            if relative_path:
                details.append(str(relative_path))
            file_index = TerminalProgress._count(event.get("file_index"))
            file_count = TerminalProgress._count(event.get("file_count"))
            if file_index <= 0 or file_count <= 0:
                file_index = TerminalProgress._count(event.get("dataset_index"))
                file_count = TerminalProgress._count(event.get("dataset_count"))
            if file_index > 0 and file_count > 0:
                details.append(f"{file_index}/{file_count}")
            rows = TerminalProgress._count(event.get("rows"))
            if rows > 0:
                details.append(f"{rows:,} rows")
            return " ".join(details)
        return f"setup {stage}"
