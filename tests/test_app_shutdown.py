"""Shutdown-path tests for the GUI orchestrator.

Regression guard for Ctrl+C escaping Tk's mainloop as an uncaught traceback:
when that happened, `_quit` never ran, so the global keyboard hook and the
tray icon were left running until the process died.
"""
import sys
from unittest.mock import MagicMock

import pytest

_GUI_STACK = (
    "tkinter", "tkinter.constants", "tkinter.font", "tkinter.ttk",
    "tkinter.messagebox", "tkinter.filedialog",
    "customtkinter", "pystray", "keyboard", "sounddevice", "pyperclip",
)

# app.py imports the whole GUI stack at module level, but the shutdown path
# under test is pure Python and touches none of it. Prefer the real modules —
# only when the import genuinely fails (a headless box typically has no system
# tkinter) fall back to stubs, so a working install is never shadowed.
#
# The stubs are then removed again: leaving them in sys.modules would let a
# later test module import a mock instead of the real dependency and pass
# against it, which is how a suite starts lying about what it covers.
try:
    from app import GUIApp
except Exception:
    for _name in [n for n in sys.modules if n == "app" or n.startswith("ui.")]:
        del sys.modules[_name]
    _injected = [n for n in _GUI_STACK if n not in sys.modules]
    for _name in _injected:
        sys.modules[_name] = MagicMock()
    try:
        from app import GUIApp
    finally:
        for _name in _injected:
            sys.modules.pop(_name, None)


class _Stop:
    """Stand-in for engine.stop / tray.stop that records how often it ran."""

    def __init__(self, raises: Exception | None = None):
        self.calls = 0
        self._raises = raises

    def __call__(self):
        self.calls += 1
        if self._raises is not None:
            raise self._raises


class _Root:
    """Minimal Tk root: mainloop can be told to raise, destroy is recorded."""

    def __init__(self, mainloop_raises: Exception | None = None):
        self.destroyed = 0
        self._raises = mainloop_raises

    def mainloop(self):
        if self._raises is not None:
            raise self._raises

    def destroy(self):
        self.destroyed += 1


def _bare_app(engine_stop, tray_stop, root=None):
    """A GUIApp carrying only the attributes the shutdown path touches.

    Built with __new__ so the test needs no display, no model files and no
    tray thread — the real __init__ constructs all three.
    """
    app = GUIApp.__new__(GUIApp)
    app._torn_down = False
    app._engine = type("_Engine", (), {"stop": engine_stop})()
    app._tray = type("_Tray", (), {"stop": tray_stop})()
    app._root = root if root is not None else _Root()
    return app


class TestTeardown:
    def test_stops_engine_and_tray(self):
        engine, tray = _Stop(), _Stop()
        _bare_app(engine, tray)._teardown()
        assert (engine.calls, tray.calls) == (1, 1)

    def test_is_idempotent(self):
        """The tray Quit and the Ctrl+C path can both fire; the second is a no-op."""
        engine, tray = _Stop(), _Stop()
        app = _bare_app(engine, tray)
        app._teardown()
        app._teardown()
        assert (engine.calls, tray.calls) == (1, 1)

    def test_tray_still_stops_when_engine_stop_raises(self):
        """A failing engine.stop must not strand the tray icon."""
        engine, tray = _Stop(raises=RuntimeError("hook already gone")), _Stop()
        _bare_app(engine, tray)._teardown()
        assert tray.calls == 1


class TestMainloopInterrupt:
    def test_keyboard_interrupt_is_swallowed_and_tears_down(self):
        engine, tray = _Stop(), _Stop()
        root = _Root(mainloop_raises=KeyboardInterrupt())
        app = _bare_app(engine, tray, root)

        app._mainloop()  # must not propagate

        assert (engine.calls, tray.calls) == (1, 1)
        assert root.destroyed == 1

    def test_normal_exit_also_tears_down(self):
        engine, tray = _Stop(), _Stop()
        root = _Root()
        app = _bare_app(engine, tray, root)

        app._mainloop()

        assert (engine.calls, tray.calls) == (1, 1)
        assert root.destroyed == 1

    def test_real_errors_still_propagate(self):
        """Only KeyboardInterrupt is absorbed — a genuine crash must surface."""
        engine, tray = _Stop(), _Stop()
        root = _Root(mainloop_raises=RuntimeError("Tk exploded"))
        app = _bare_app(engine, tray, root)

        with pytest.raises(RuntimeError, match="Tk exploded"):
            app._mainloop()

        assert (engine.calls, tray.calls) == (1, 1)  # teardown still ran
