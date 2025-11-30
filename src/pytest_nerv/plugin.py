from __future__ import annotations

import os
import pytest
import select
import shutil
import sys
import termios
import threading
import tty
from enum import Enum
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from _pytest.config import create_terminal_writer
from _pytest.reports import TestReport


class TestStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    __test__ = False  # prevent pytest from collecting this Enum as a test


ANSI_RESET = "\x1b[0m"
STATUS_COLORS: Dict[TestStatus, str] = {
    TestStatus.PENDING: "\x1b[100m",  # bright black background
    TestStatus.RUNNING: "\x1b[44m",  # blue background (was yellow)
    TestStatus.PASSED: "\x1b[42m",  # green background
    TestStatus.FAILED: "\x1b[41m",  # red background
    TestStatus.SKIPPED: "\x1b[43m",  # yellow background (was blue)
}
STATUS_SYMBOLS: Dict[TestStatus, str] = {
    TestStatus.PENDING: "?",  # keep indicator for undispatched tests
    TestStatus.RUNNING: " ",  # no symbol while running
    TestStatus.PASSED: " ",  # no symbol for success
    TestStatus.FAILED: " ",  # no symbol for failure
    TestStatus.SKIPPED: "~",
}
COUNT_COLORS: Dict[TestStatus, str] = {
    TestStatus.PASSED: "\x1b[32m",  # green text
    TestStatus.FAILED: "\x1b[31m",  # red text
    TestStatus.RUNNING: "\x1b[34m",  # blue text
    TestStatus.SKIPPED: "\x1b[33m",  # yellow text
    TestStatus.PENDING: "\x1b[90m",  # bright black text
}


def _compute_pagination(num_tests: int, max_block_rows: int, width: int) -> Tuple[int, int, int, int]:
    block_width = max(3, len(str(max(1, num_tests))))
    block_slot_width = block_width + 1  # width plus a spacer
    blocks_per_row = max(1, width // block_slot_width)
    page_size = max(1, blocks_per_row * max_block_rows)
    total_pages = max(1, (num_tests + page_size - 1) // page_size)
    return block_width, blocks_per_row, page_size, total_pages


class TUIState:
    def __init__(self, log_height: int = 8, log_limit: int = 800, keep_full_log: bool = False) -> None:
        self.tests: List[str] = []
        self.statuses: Dict[str, TestStatus] = {}
        self.logs: List[str] = []
        self.full_logs: List[str] = []
        self.keep_full_log = keep_full_log
        self.show_all_logs = False
        self.active_test: Optional[str] = None
        self.log_height = max(1, log_height)
        self.log_limit = max(1, log_limit)
        self.page_index = 0
        self._lock = threading.Lock()

    def register_tests(self, nodeids: Iterable[str]) -> None:
        with self._lock:
            for nodeid in nodeids:
                if nodeid not in self.statuses:
                    self.tests.append(nodeid)
                    self.statuses[nodeid] = TestStatus.PENDING
            self.page_index = 0

    def mark_running(self, nodeid: str) -> None:
        with self._lock:
            if nodeid not in self.statuses:
                self.tests.append(nodeid)
            self.statuses[nodeid] = TestStatus.RUNNING
            self.active_test = nodeid

    def mark_outcome(self, nodeid: str, status: TestStatus) -> None:
        with self._lock:
            previous = self.statuses.get(nodeid, TestStatus.PENDING)
            if previous == TestStatus.FAILED:
                resolved = previous
            elif status == TestStatus.FAILED:
                resolved = status
            elif previous == TestStatus.SKIPPED and status == TestStatus.PASSED:
                resolved = previous
            else:
                resolved = status

            self.statuses[nodeid] = resolved
            if resolved in (TestStatus.PASSED, TestStatus.FAILED, TestStatus.SKIPPED):
                if self.active_test == nodeid:
                    self.active_test = None

    def append_log(self, message: str) -> None:
        text = "" if message is None else str(message)
        lines = text.splitlines() or [text]
        with self._lock:
            self.logs.extend(lines)
            if len(self.logs) > self.log_limit:
                self.logs = self.logs[-self.log_limit :]
            if self.keep_full_log:
                self.full_logs.extend(lines)

    def toggle_logs(self) -> None:
        with self._lock:
            self.show_all_logs = not self.show_all_logs

    def get_page(self) -> int:
        with self._lock:
            return self.page_index

    def set_page(self, index: int) -> None:
        with self._lock:
            self.page_index = max(0, index)

    def clamp_page(self, max_pages: int) -> int:
        with self._lock:
            if max_pages <= 0:
                self.page_index = 0
                return 0
            if self.page_index >= max_pages:
                self.page_index = max_pages - 1
            return self.page_index

    def next_page(self) -> None:
        with self._lock:
            self.page_index += 1

    def prev_page(self) -> None:
        with self._lock:
            self.page_index = max(0, self.page_index - 1)

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "tests": list(self.tests),
                "statuses": dict(self.statuses),
                "logs": list(self.logs),
                "show_all_logs": self.show_all_logs,
                "active_test": self.active_test,
                "page": self.page_index,
            }

    def visible_logs(self) -> List[str]:
        with self._lock:
            if self.show_all_logs:
                return list(self.logs)
            return list(self.logs[-self.log_height :])

    def counts(self) -> Dict[TestStatus, int]:
        with self._lock:
            counts: Dict[TestStatus, int] = {status: 0 for status in TestStatus}
            for status in self.statuses.values():
                counts[status] = counts.get(status, 0) + 1
            return counts


class LogCaptureStream:
    """Minimal stream that captures pytest terminal output into the log pane."""

    def __init__(self, push_log: Callable[[str], None]) -> None:
        self.push_log = push_log
        self._buffer = ""
        self.encoding = "utf-8"

    def write(self, data: str) -> int:
        if not data:
            return 0
        self._buffer += str(data)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self.push_log(line)
        return len(data)

    def flush(self) -> None:  # pragma: no cover - nothing to flush
        return

    def isatty(self) -> bool:  # pragma: no cover - required by TerminalWriter
        return False


class TerminalSilencer:
    def __init__(self, config: object, pluginmanager: object, push_log: Callable[[str], None]) -> None:
        self.config = config
        self.pluginmanager = pluginmanager
        self.push_log = push_log
        self.original_tw = None
        self.stream: Optional[LogCaptureStream] = None
        self.terminalreporter = None

    def start(self) -> None:
        if self.terminalreporter is None:
            self.terminalreporter = getattr(self.pluginmanager, "getplugin", lambda _: None)("terminalreporter")
        if not self.terminalreporter:
            return
        if self.stream is None:
            self.stream = LogCaptureStream(self.push_log)
        current_tw = getattr(self.terminalreporter, "_tw", None)
        if self.original_tw is None and current_tw is not None:
            self.original_tw = current_tw
        self.terminalreporter._tw = create_terminal_writer(self.config, file=self.stream)  # type: ignore[attr-defined]

    def stop(self) -> None:
        if self.terminalreporter and self.original_tw is not None:
            self.terminalreporter._tw = self.original_tw  # type: ignore[attr-defined]
            self.original_tw = None


class ToggleListener(threading.Thread):
    def __init__(self, callback: Callable[[bytes], None]) -> None:
        super().__init__(daemon=True)
        self.callback = callback
        self._stop_event = threading.Event()
        self._original_termios = None

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        if not sys.stdin.isatty():
            return
        fd = sys.stdin.fileno()
        try:
            self._original_termios = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception:
            return

        try:
            while not self._stop_event.is_set():
                readable, _, _ = select.select([fd], [], [], 0.1)
                if fd in readable:
                    key = os.read(fd, 1)
                    self.callback(key)
        finally:
            if self._original_termios is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, self._original_termios)


class TUIRenderer:
    def __init__(self, state: TUIState, max_block_rows: int = 5, stream: Optional[object] = None) -> None:
        self.state = state
        self.max_block_rows = max(1, max_block_rows)
        # Prefer writing directly to the controlling TTY to avoid pytest capture.
        self.stream = stream or self._open_tty() or getattr(sys, "__stdout__", sys.stdout)
        self._cursor_hidden = False
        self._last_frame: Optional[str] = None
        # Consider both captured and real stdout for TTY detection.
        self.enabled = any(
            checker()
            for checker in (
                getattr(self.stream, "isatty", lambda: False),
                getattr(getattr(sys, "stdout", None), "isatty", lambda: False),
                getattr(getattr(sys, "__stdout__", None), "isatty", lambda: False),
            )
        )

    def _open_tty(self) -> Optional[object]:
        try:
            return open("/dev/tty", "w", buffering=1)
        except OSError:
            return None

    def build_screen(self) -> str:
        view = self.state.snapshot()
        width = shutil.get_terminal_size((80, 24)).columns

        lines: List[str] = []
        lines.append("pytest-nerv — TUI reporter (Ctrl+O toggles logs)")
        lines.append(self._render_counts_line(view, width))
        ratio_line = self._render_ratio_line(view, width)
        if ratio_line:
            lines.append(ratio_line)
            lines.append("-" * width)  # divider between ratio line and grid
        grid_lines, page_info = self._render_grid(view, width)
        lines.extend(grid_lines)
        if page_info:
            lines.append(page_info)
        lines.append("")
        lines.append(self._render_legend(width))
        lines.append("")
        lines.append(self._render_log_header(view))
        lines.extend(self._render_logs(width))
        clear_prefix = f"{ANSI_RESET}\x1b[H\x1b[J"
        return clear_prefix + "\n".join(lines) + ANSI_RESET

    def render(self) -> None:
        if not self.enabled:
            return
        screen = self.build_screen()
        if screen == self._last_frame:
            return
        self._last_frame = screen
        self._ensure_cursor_hidden()
        self.stream.write(screen + "\n")
        self.stream.flush()

    def restore_cursor(self) -> None:
        if self.enabled and self._cursor_hidden:
            self.stream.write("\x1b[?25h")
            self.stream.flush()
            self._cursor_hidden = False

    def _ensure_cursor_hidden(self) -> None:
        if self.enabled and not self._cursor_hidden:
            self.stream.write("\x1b[?25l")
            self._cursor_hidden = True

    def _render_counts_line(self, view: Dict[str, object], width: int) -> str:
        statuses: Dict[str, TestStatus] = view["statuses"]  # type: ignore[assignment]
        counts: Dict[TestStatus, int] = {status: 0 for status in TestStatus}
        for status in statuses.values():
            counts[status] = counts.get(status, 0) + 1
        color_num = lambda count, status: f"{COUNT_COLORS.get(status, '')}{count}{ANSI_RESET if status in COUNT_COLORS else ''}"

        parts = [
            f"total {len(statuses)}",
            f"passed {color_num(counts[TestStatus.PASSED], TestStatus.PASSED)}",
            f"failed {color_num(counts[TestStatus.FAILED], TestStatus.FAILED)}",
            f"running {color_num(counts[TestStatus.RUNNING], TestStatus.RUNNING)}",
            f"skipped {color_num(counts[TestStatus.SKIPPED], TestStatus.SKIPPED)}",
        ]
        active = view.get("active_test")
        if active:
            parts.append(f"active {active}")
        return " | ".join(parts)

    def _render_ratio_line(self, view: Dict[str, object], width: int) -> Optional[str]:
        statuses: Dict[str, TestStatus] = view["statuses"]  # type: ignore[assignment]
        total = len(statuses)
        if total == 0:
            return None

        counts: Dict[TestStatus, int] = {status: 0 for status in TestStatus}
        for status in statuses.values():
            counts[status] += 1

        bar_width = max(10, min(50, width // 2))
        order = [
            TestStatus.PASSED,
            TestStatus.FAILED,
            TestStatus.SKIPPED,
            TestStatus.RUNNING,
            TestStatus.PENDING,
        ]
        raw_lengths = [counts[s] * bar_width / total for s in order]
        lengths = [int(x) for x in raw_lengths]
        remainder = bar_width - sum(lengths)
        if remainder != 0:
            fractions = [x - int(x) for x in raw_lengths]
            # Distribute remaining space based on largest fractional parts.
            for _ in range(abs(remainder)):
                if not fractions:
                    break
                idx = max(range(len(fractions)), key=lambda i: fractions[i])
                lengths[idx] += 1 if remainder > 0 else -1 if lengths[idx] > 0 else 0
                fractions[idx] = 0

        bar_parts: List[str] = []
        for status, length in zip(order, lengths):
            if length <= 0 or counts[status] == 0:
                continue
            bar_parts.append(f"{STATUS_COLORS[status]}{' ' * length}{ANSI_RESET}")
        bar = "".join(bar_parts) if bar_parts else " " * bar_width

        info_parts: List[str] = []
        for status in order:
            count = counts[status]
            if count == 0:
                continue
            pct = (count / total) * 100
            info_parts.append(f"{status.value} {count} ({pct:.1f}%)")

        return f"overall: {bar} {' | '.join(info_parts)}"

    def _render_grid(self, view: Dict[str, object], width: int) -> Tuple[List[str], Optional[str]]:
        tests: List[str] = view["tests"]  # type: ignore[assignment]
        statuses: Dict[str, TestStatus] = view["statuses"]  # type: ignore[assignment]
        if not tests:
            return ["(no collected tests)"], None

        block_width, blocks_per_row, page_size, total_pages = _compute_pagination(
            len(tests), self.max_block_rows, width
        )
        block_height = 4  # 3x3 for color/shape + 1 line for index

        def make_block(idx: int, status: TestStatus) -> List[str]:
            color = STATUS_COLORS.get(status, STATUS_COLORS[TestStatus.PENDING])
            symbol = STATUS_SYMBOLS.get(status, "?")
            top = f"{color}{' ' * block_width}{ANSI_RESET}"
            pad_left = (block_width - 1) // 2
            pad_right = block_width - pad_left - 1
            mid = f"{color}{' ' * pad_left}{symbol}{' ' * pad_right}{ANSI_RESET}"
            index_line = f"{idx:0{block_width}d}".ljust(block_width)
            return [top, mid, top, index_line]

        blocks: List[List[str]] = []
        for idx, nodeid in enumerate(tests, start=1):
            status = statuses.get(nodeid, TestStatus.PENDING)
            blocks.append(make_block(idx, status))

        page_index = self.state.clamp_page(total_pages)
        start_block = page_index * page_size
        visible_blocks = blocks[start_block : start_block + page_size]

        rows: List[str] = []
        for start in range(0, len(visible_blocks), blocks_per_row):
            group = visible_blocks[start : start + blocks_per_row]
            for line_idx in range(block_height):
                line = " ".join(block[line_idx] for block in group)
                rows.append(line)
            if start + blocks_per_row < len(visible_blocks):
                rows.append("")  # spacer between block rows

        page_line = None
        if total_pages > 1:
            page_line = f"page {page_index + 1}/{total_pages} (n/]:next p/[:prev)"

        return rows, page_line

    def _render_legend(self, width: int) -> str:
        parts = []
        for status in (
            TestStatus.PENDING,
            TestStatus.RUNNING,
            TestStatus.PASSED,
            TestStatus.SKIPPED,
            TestStatus.FAILED,
        ):
            color = STATUS_COLORS[status]
            parts.append(f"{color}  {ANSI_RESET} {status.value}")

        text = "  ".join(parts)
        return text if len(text) <= width else text[: width - 1]

    def _render_log_header(self, view: Dict[str, object]) -> str:
        show_all = bool(view.get("show_all_logs"))
        suffix = "full log" if show_all else f"last {self.state.log_height} lines"
        return f"Logs ({suffix})"

    def _render_logs(self, width: int) -> List[str]:
        logs = self.state.visible_logs()
        if not logs:
            return ["(waiting for pytest output)"]

        rendered: List[str] = []
        for line in logs:
            truncated = line if len(line) < width else line[: width - 1]
            rendered.append(truncated)
        return rendered


class NervPlugin:
    def __init__(self, config: object) -> None:
        self.config = config
        log_height = getattr(config.option, "nerv_log_height", 8)
        log_limit = getattr(config.option, "nerv_log_limit", 800)
        max_grid_rows = getattr(config.option, "nerv_grid_rows", 5)
        keep_full_log = bool(getattr(config.option, "nerv_log_file", None))
        self.state = TUIState(log_height=log_height, log_limit=log_limit, keep_full_log=keep_full_log)
        self.renderer = TUIRenderer(self.state, max_block_rows=max_grid_rows)
        self.silencer = TerminalSilencer(config, config.pluginmanager, self.state.append_log)
        self.toggle_listener = ToggleListener(self._on_key)
        self._started = False
        self._stopped = False
        self.log_file = getattr(config.option, "nerv_log_file", None)

    def pytest_configure(self, config: object) -> None:  # pragma: no cover - exercised in live pytest runs
        self._start()

    def pytest_plugin_registered(self, plugin: object, manager: object) -> None:
        if manager is getattr(self.config, "pluginmanager", None):
            self.silencer.start()

    def pytest_sessionstart(self, session: object) -> None:
        self.silencer.start()
        self._render()

    def pytest_collection_finish(self, session: object) -> None:
        nodeids = [item.nodeid for item in getattr(session, "items", [])]
        self.state.register_tests(nodeids)
        self._render()

    def pytest_runtest_logstart(self, nodeid: str, location: object) -> None:
        self.state.mark_running(nodeid)
        self._render()

    def pytest_runtest_logreport(self, report: TestReport) -> None:
        self._update_status_from_report(report)
        self._append_report_output(report)
        self._render()

    @pytest.hookimpl(optionalhook=True)
    def pytest_warning_captured(self, warning_message: object, when: str, item: object, location: object) -> None:
        self.state.append_log(f"[warning during {when}] {warning_message}")
        self._render()

    def pytest_internalerror(self, excrepr: object, excinfo: object) -> None:
        self.state.append_log("[internal error] pytest encountered an internal error.")
        self.state.append_log(str(excrepr))
        self._render()

    def pytest_sessionfinish(self, session: object, exitstatus: int) -> None:
        self.state.append_log(f"Exit status: {exitstatus}")
        self._render()

    def teardown(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self.toggle_listener.stop()
        if self.toggle_listener.is_alive():
            self.toggle_listener.join(timeout=0.2)
        self.silencer.stop()
        self._render()
        self.renderer.restore_cursor()
        summary = self._format_summary()
        out_stream = getattr(self.renderer, "stream", sys.__stdout__)
        try:
            out_stream.write("\n" + summary + "\n")
            out_stream.flush()
        except Exception:
            sys.stdout.write("\n" + summary + "\n")
            sys.stdout.flush()
        self._write_log_file()

    def _start(self) -> None:
        if self._started:
            return
        self.silencer.start()
        self.state.append_log("pytest-nerv enabled. Press Ctrl+O to expand/collapse logs.")
        if sys.stdin.isatty():
            self.toggle_listener.start()
        self._started = True

    def _render(self) -> None:
        width = shutil.get_terminal_size((80, 24)).columns
        auto_advance_page(self.state, self.renderer.max_block_rows, width)
        self.silencer.start()
        self.renderer.render()

    def _on_key(self, key: bytes) -> None:
        if key == b"\x0f":  # Ctrl+O toggles log pane
            self.state.toggle_logs()
        elif key in (b"n", b"]"):
            self.state.next_page()
        elif key in (b"p", b"["):
            self.state.prev_page()
        else:
            return
        self._render()

    def _update_status_from_report(self, report: TestReport) -> None:
        nodeid = report.nodeid
        if report.when == "setup":
            if report.failed:
                self.state.mark_outcome(nodeid, TestStatus.FAILED)
            elif report.skipped:
                self.state.mark_outcome(nodeid, TestStatus.SKIPPED)
            else:
                self.state.mark_running(nodeid)
            return

        if report.when == "call":
            if report.failed:
                self.state.mark_outcome(nodeid, TestStatus.FAILED)
            elif report.skipped:
                self.state.mark_outcome(nodeid, TestStatus.SKIPPED)
            else:
                self.state.mark_outcome(nodeid, TestStatus.PASSED)
            return

        if report.when == "teardown" and report.failed:
            self.state.mark_outcome(nodeid, TestStatus.FAILED)

    def _append_report_output(self, report: TestReport) -> None:
        details: List[str] = []
        header = f"[{report.when}] {report.nodeid} -> {report.outcome}"
        details.append(header)

        for label in ("capstdout", "capstderr", "caplogtext"):
            content = getattr(report, label, "")
            if content:
                prefix = label.replace("cap", "")
                for line in str(content).splitlines():
                    details.append(f"{prefix}: {line}")

        if report.failed and getattr(report, "longreprtext", None):
            details.extend(str(report.longreprtext).splitlines())

        for line in details:
            self.state.append_log(line)

    def _write_log_file(self) -> None:
        if not self.log_file:
            return
        try:
            lines = self.state.full_logs or self.state.logs
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception:
            # Fail silently; do not disrupt pytest flow.
            return

    def _format_summary(self) -> str:
        counts = self.state.counts()
        parts = [
            "pytest-nerv summary:",
            f"passed={counts[TestStatus.PASSED]}",
            f"failed={counts[TestStatus.FAILED]}",
            f"skipped={counts[TestStatus.SKIPPED]}",
            f"total={sum(counts.values())}",
        ]
        return " ".join(parts)


def should_enable(config: object) -> bool:
    no_nerv = getattr(config.option, "no_nerv", None)
    if no_nerv:
        return False
    nerv = getattr(config.option, "nerv", None)
    if nerv is True:
        return True
    if nerv is False:
        return False
    return sys.stdout.isatty()


def auto_advance_page(state: TUIState, max_block_rows: int, width: int) -> bool:
    """Advance to the next page when all tests on the current page are finished."""
    view = state.snapshot()
    tests: List[str] = view["tests"]  # type: ignore[assignment]
    statuses: Dict[str, TestStatus] = view["statuses"]  # type: ignore[assignment]
    if not tests:
        return False

    _, _, page_size, total_pages = _compute_pagination(len(tests), max_block_rows, width)
    page_index = state.clamp_page(total_pages)
    if page_index >= total_pages - 1:
        return False

    start = page_index * page_size
    end = min(len(tests), start + page_size)
    page_tests = tests[start:end]
    if not page_tests:
        return False

    unfinished = {TestStatus.PENDING, TestStatus.RUNNING}
    if all(statuses.get(nodeid, TestStatus.PENDING) not in unfinished for nodeid in page_tests):
        state.set_page(page_index + 1)
        return True

    return False
