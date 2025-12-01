import io
import os
import re
import shutil

from pytest_nerv.plugin import TestStatus, TUIRenderer, TUIState, auto_advance_page


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", text)


def test_state_transitions_and_counts() -> None:
    state = TUIState(log_height=2, log_limit=10)
    state.register_tests(["test_a.py::test_one", "test_b.py::test_two"])
    state.mark_running("test_a.py::test_one")
    state.mark_outcome("test_a.py::test_one", TestStatus.PASSED)
    state.mark_outcome("test_b.py::test_two", TestStatus.SKIPPED)

    counts = state.counts()
    assert counts[TestStatus.PASSED] == 1
    assert counts[TestStatus.SKIPPED] == 1
    snapshot = state.snapshot()
    assert snapshot["statuses"]["test_a.py::test_one"] == TestStatus.PASSED
    assert snapshot["statuses"]["test_b.py::test_two"] == TestStatus.SKIPPED


def test_log_window_toggle_and_limit() -> None:
    state = TUIState(log_height=2, log_limit=4)
    state.register_tests(["test_a.py::test_one"])

    for idx in range(5):
        state.append_log(f"line {idx}")

    # Only the last 4 lines are retained because of log_limit.
    assert state.logs == ["line 1", "line 2", "line 3", "line 4"]
    assert state.visible_logs() == ["line 3", "line 4"]

    state.toggle_logs()
    assert state.visible_logs() == ["line 1", "line 2", "line 3", "line 4"]


def test_renderer_builds_screen_with_ansi_blocks(monkeypatch) -> None:
    monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback=None: os.terminal_size((80, 24)))
    state = TUIState(log_height=2)
    state.register_tests(["test_a.py::test_one"])
    state.mark_outcome("test_a.py::test_one", TestStatus.PASSED)

    renderer = TUIRenderer(state, stream=io.StringIO())
    screen = renderer.build_screen()

    assert "\x1b[42m" in screen  # green background for passed blocks
    assert "passed \x1b[32m1\x1b[0m" in screen  # colored count only
    assert "overall:" in screen
    assert "passed 1 (100.0%)" in screen
    assert "Logs" in screen


def test_ratio_line_includes_pending(monkeypatch) -> None:
    monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback=None: os.terminal_size((80, 24)))
    state = TUIState(log_height=2)
    state.register_tests(["test_a.py::test_one", "test_b.py::test_two"])
    # Leave both pending to exercise pending ratio display.

    renderer = TUIRenderer(state, stream=io.StringIO())
    screen = renderer.build_screen()

    assert "overall:" in screen
    assert "pending 2 (100.0%)" in screen


def test_auto_advance_page_moves_when_page_complete() -> None:
    state = TUIState()
    tests = [f"test_{idx}" for idx in range(12)]
    state.register_tests(tests)
    width = 10  # small width to force a low page_size (10 tests per page)

    # Should not move while page tests are pending.
    moved = auto_advance_page(state, max_block_rows=5, width=width)
    assert moved is False
    assert state.get_page() == 0

    for nodeid in tests[:10]:
        state.mark_outcome(nodeid, TestStatus.PASSED)

    moved = auto_advance_page(state, max_block_rows=5, width=width)
    assert moved is True
    assert state.get_page() == 1

    # Last page done; no further advancement.
    for nodeid in tests[10:]:
        state.mark_outcome(nodeid, TestStatus.PASSED)

    moved = auto_advance_page(state, max_block_rows=5, width=width)
    assert moved is False
    assert state.get_page() == 1


def test_counts_line_truncates_active_node(monkeypatch) -> None:
    width = 80
    monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback=None: os.terminal_size((width, 24)))
    state = TUIState(log_height=2)
    nodeid = (
        "test/data/services/test_adjustfactor_service.py::"
        "AdjustfactorServiceTest::test_sync_incremental_empty_data"
    )
    state.register_tests([nodeid])
    state.mark_running(nodeid)

    renderer = TUIRenderer(state, stream=io.StringIO())
    counts_line = renderer._render_counts_line(state.snapshot(), width)

    assert len(strip_ansi(counts_line)) <= width
    assert "active" in counts_line
    assert "..." in counts_line


def test_ratio_line_truncates_when_info_is_wide(monkeypatch) -> None:
    width = 60
    monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback=None: os.terminal_size((width, 24)))
    state = TUIState(log_height=2)
    tests = [f"test_{idx}" for idx in range(20)]
    state.register_tests(tests)

    for nodeid in tests[:10]:
        state.mark_outcome(nodeid, TestStatus.PASSED)
    for nodeid in tests[10:15]:
        state.mark_outcome(nodeid, TestStatus.FAILED)
    for nodeid in tests[15:17]:
        state.mark_outcome(nodeid, TestStatus.SKIPPED)
    state.mark_running(tests[17])

    renderer = TUIRenderer(state, stream=io.StringIO())
    ratio_line = renderer._render_ratio_line(state.snapshot(), width)

    assert ratio_line is not None
    assert ratio_line.startswith("overall:")
    assert len(strip_ansi(ratio_line)) <= width
