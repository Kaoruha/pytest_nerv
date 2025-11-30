import io

from pytest_nerv.plugin import TestStatus, TUIRenderer, TUIState


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


def test_renderer_builds_screen_with_ansi_blocks() -> None:
    state = TUIState(log_height=2)
    state.register_tests(["test_a.py::test_one"])
    state.mark_outcome("test_a.py::test_one", TestStatus.PASSED)

    renderer = TUIRenderer(state, stream=io.StringIO())
    screen = renderer.build_screen()

    assert "\x1b[42m" in screen  # green background for passed blocks
    assert "Logs" in screen
