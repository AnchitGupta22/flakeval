"""The progress display shown while a study runs: one overall bar that counts model calls."""
from __future__ import annotations

import shutil
import sys
import threading
import time


def hms(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


class Progress:
    """One overall bar that counts model calls, with the current system shown next to it.

    In a terminal it redraws a single line. When the output goes to a file or pipe (for example `| tee run.log`)
    it prints one plain line every 10% instead, so logs do not fill up with redraw characters."""

    def __init__(self, total: int, stream=None):
        self.total, self.done = max(total, 1), 0
        self.stream = stream or sys.stdout
        self.tty = hasattr(self.stream, "isatty") and self.stream.isatty()
        self.start = time.monotonic()
        self.system, self.sys_index, self.sys_count, self.sys_calls, self.sys_done = "", 0, 0, 0, 0
        self._last_draw, self._next_mark, self._drawn = 0.0, 10, False
        self._lock = threading.RLock()

    def start_system(self, name: str, index: int, count: int, calls: int) -> None:
        with self._lock:
            self.system, self.sys_index, self.sys_count, self.sys_calls, self.sys_done = name, index, count, calls, 0
            self._update(force=True)

    def advance(self) -> None:
        with self._lock:
            self.done += 1
            self.sys_done += 1
            self._update(force=self.done >= self.total)

    def message(self, text: str) -> None:
        """Print a normal line without leaving a half-drawn bar behind."""
        with self._lock:
            if self.tty and self._drawn:
                self.stream.write("\r" + " " * (self._columns() - 1) + "\r")
            self.stream.write(text + "\n")
            self.stream.flush()
            if self.tty:
                self._draw()

    def close(self) -> None:
        with self._lock:
            if self.tty and self._drawn:
                self.stream.write("\n")
                self.stream.flush()
            self._drawn = False

    # -- internals
    def _columns(self) -> int:
        return shutil.get_terminal_size((80, 20)).columns

    def _eta(self) -> str:
        if self.done < 3:
            return "--"
        elapsed = time.monotonic() - self.start
        return "~" + hms(elapsed / self.done * (self.total - self.done))

    def _variants(self) -> list[str]:
        pct = 100 * self.done // self.total
        filled = 20 * self.done // self.total
        bar = "[" + "█" * filled + "░" * (20 - filled) + "]"
        elapsed, eta = hms(time.monotonic() - self.start), self._eta()
        where = f"{self.system} (system {self.sys_index}/{self.sys_count})" if self.system else ""
        return [
            f"{bar} {pct:3d}%  {self.done}/{self.total} calls | {where} {self.sys_done}/{self.sys_calls} | "
            f"elapsed {elapsed} | ETA {eta}",
            f"{bar} {pct:3d}%  {self.done}/{self.total} calls | {where} | ETA {eta}",
            f"{bar} {pct:3d}%  {self.done}/{self.total} | ETA {eta}",
            f"{pct}% ETA {eta}",
        ]

    def _draw(self) -> None:
        cols = self._columns()
        variants = self._variants()
        line = next((v for v in variants if len(v) < cols), variants[-1][: cols - 1])
        self.stream.write("\r" + line.ljust(cols - 1))
        self.stream.flush()
        self._drawn, self._last_draw = True, time.monotonic()

    def _update(self, force: bool = False) -> None:
        if self.tty:
            if force or time.monotonic() - self._last_draw >= 0.2:
                self._draw()
            return
        pct = 100 * self.done // self.total
        if pct >= self._next_mark:  # plain log line every 10%
            self._next_mark = (pct // 10 + 1) * 10
            where = f" | {self.system} (system {self.sys_index}/{self.sys_count})" if self.system else ""
            self.stream.write(f"progress: {pct}% ({self.done}/{self.total} calls){where} | "
                              f"elapsed {hms(time.monotonic() - self.start)} | ETA {self._eta()}\n")
            self.stream.flush()
