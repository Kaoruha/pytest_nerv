# pytest-nerv

Interactive TUI reporter for pytest. Replaces the default terminal output with a dual-pane view: a status grid on top and a live log window below. Designed for large suites with fast visual feedback, pagination, and quick log toggling.

## Quick start
- Install in editable mode (recommended for development):
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -e .  # or pip install -e .[dev] if you add dev extras
  ```
- Run pytest with the TUI:
  ```bash
  pytest --nerv
  ```
- Toggle log window: press `Ctrl+O`.
- Paginate the grid when there are many tests: `n` or `]` for next page, `p` or `[` for previous page.
  - The view auto-advances to the next page when all tests on the current page finish.

## Features
- Color-coded status blocks (pending, running, passed, skipped, failed) with per-test indices.
- Paginated grid for large suites; configurable rows per page.
- Live log pane showing recent pytest output with an expand/collapse toggle.
- Optional full log export to a file for detailed debugging.
- Works in TTY terminals; falls back gracefully when not attached to a TTY.

## Key options
- `--nerv` / `--no-nerv`: force enable/disable the reporter (auto-enables on TTY by default).
- `--nerv-log-height <int>`: visible log lines before toggling (default 8).
- `--nerv-log-limit <int>`: in-memory log buffer size (default 800).
- `--nerv-log-file <path>`: write full captured log to a file after the run.
- `--nerv-grid-rows <int>`: max block rows per page in the status grid (default 5).

## Development
- Run tests locally:
  ```bash
  pytest
  ```
- Code lives under `src/pytest_nerv/`; add tests under `tests/`.
- The plugin entrypoint is registered via `pyproject.toml` (`[project.entry-points.pytest11]`).
- Keep TUI rendering logic and pytest hook handling decoupled (see `plugin.py`).

## Troubleshooting
- If collection errors stop the run, open the log pane (`Ctrl+O`) or enable full log export (`--nerv-log-file /tmp/pytest-nerv.log`) to inspect stack traces.
- If the TUI does not appear, ensure you are in a real TTY and try `pytest --nerv`; avoid `-s` unless you intentionally want raw output.
