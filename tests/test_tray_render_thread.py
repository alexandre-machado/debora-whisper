"""The tray must never call pystray (Shell_NotifyIcon) from its caller.

app.py calls update_state/update_audio_level from Tk callbacks. When those
blocked in Shell_NotifyIcon, Ctrl+C landed inside the callback and Tk
swallowed the KeyboardInterrupt ("Exception in Tkinter callback"), so the
app did not close.
"""
import sys
import threading
import time
from unittest.mock import MagicMock, patch


def _tray_class():
    try:
        import pystray  # noqa: F401
    except ImportError:
        with patch.dict(sys.modules, {"pystray": MagicMock()}):
            sys.modules.pop("ui.tray", None)
            from ui.tray import TrayManager
        sys.modules.pop("ui.tray", None)
        return TrayManager
    from ui.tray import TrayManager
    return TrayManager


class _FakeIcon:
    """Records which thread touched each pystray attribute."""
    visible = True

    def __init__(self):
        object.__setattr__(self, "calls", [])

    def __setattr__(self, name, value):
        self.calls.append((name, threading.current_thread().name))
        object.__setattr__(self, name, value)

    def update_menu(self):
        self.calls.append(("update_menu", threading.current_thread().name))

    def stop(self):
        pass


def _tray():
    tray = _tray_class()(on_toggle=lambda: None, on_quit=lambda: None)
    tray._icon = _FakeIcon()
    return tray


def _run_renderer(tray):
    tray._running = True
    t = threading.Thread(target=tray._render_loop, name="tray-render", daemon=True)
    t.start()
    return t


def _settle(icon, timeout=2.0):
    """Wait until the renderer stops producing calls."""
    deadline = time.time() + timeout
    last = -1
    while time.time() < deadline:
        n = len(icon.calls)
        if n == last:
            return
        last = n
        time.sleep(0.25)


def test_updates_do_not_touch_pystray_on_caller_thread():
    tray = _tray()
    tray.update_state("recording", "NPU Dictation — Recording...")
    tray.update_audio_level(0.8)
    tray.update_state("ready", "NPU Dictation — Ready")
    assert tray._icon.calls == []


def test_renderer_applies_state_on_its_own_thread():
    tray = _tray()
    t = _run_renderer(tray)
    try:
        tray.update_state("ready", "NPU Dictation — Ready")
        _settle(tray._icon)
        names = {name for name, _ in tray._icon.calls}
        assert {"icon", "title", "menu", "update_menu"} <= names
        assert {thread for _, thread in tray._icon.calls} == {"tray-render"}
    finally:
        tray.stop()
        t.join(1)
    assert not t.is_alive()


def test_repeated_state_makes_no_windows_calls():
    # Continuous mode re-sends RECORDING with every draft.
    tray = _tray()
    t = _run_renderer(tray)
    try:
        tray.update_state("ready", "NPU Dictation — Ready")
        _settle(tray._icon)
        tray._icon.calls.clear()
        for _ in range(5):
            tray.update_state("ready", "NPU Dictation — Ready")
        _settle(tray._icon)
        assert tray._icon.calls == []
    finally:
        tray.stop()
        t.join(1)


def test_silent_recording_does_not_redraw():
    tray = _tray()
    t = _run_renderer(tray)
    try:
        tray.update_state("recording", "NPU Dictation — Recording...")
        tray.update_audio_level(0.0)
        _settle(tray._icon)
        tray._icon.calls.clear()
        time.sleep(0.5)  # five 100ms frames of silence
        assert tray._icon.calls == []
    finally:
        tray.stop()
        t.join(1)
