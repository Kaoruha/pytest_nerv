"""
pytest-nerv: an interactive TUI reporter for pytest.

The plugin replaces the default terminal output with a dual-pane layout that
shows test status blocks above and a rolling log view below. Press Ctrl+O
while the test run is active to expand or collapse the log pane.
"""

from __future__ import annotations

import sys
from typing import Any

from .plugin import NervPlugin, should_enable


def pytest_addoption(parser: Any) -> None:
    group = parser.getgroup("nerv")
    group.addoption(
        "--nerv",
        action="store_true",
        default=None,
        help="Enable pytest-nerv TUI reporter (default: auto when TTY).",
    )
    group.addoption(
        "--no-nerv",
        action="store_true",
        default=None,
        help="Disable pytest-nerv reporter even if installed.",
    )
    group.addoption(
        "--nerv-log-height",
        action="store",
        type=int,
        default=8,
        help="Visible log pane height before toggling to the full log buffer.",
    )
    group.addoption(
        "--nerv-log-limit",
        action="store",
        type=int,
        default=800,
        help="Maximum log lines retained in memory for the TUI.",
    )
    group.addoption(
        "--nerv-log-file",
        action="store",
        default=None,
        help="Path to write the full pytest-nerv log (all captured output).",
    )
    group.addoption(
        "--nerv-grid-rows",
        action="store",
        type=int,
        default=5,
        help="Maximum block rows shown per page in the status grid.",
    )


def pytest_configure(config: Any) -> None:
    if not should_enable(config):
        return

    plugin = NervPlugin(config)
    config.pluginmanager.register(plugin, "pytest-nerv")
    config._nerv_plugin = plugin  # type: ignore[attr-defined]


def pytest_unconfigure(config: Any) -> None:
    plugin = getattr(config, "_nerv_plugin", None)
    if plugin is not None:
        plugin.teardown()
        config.pluginmanager.unregister(plugin)
