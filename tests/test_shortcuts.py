"""Start Menu shortcuts and the single-instance guard (Windows only)."""
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows shortcuts")

from npu_whisper import shortcuts  # noqa: E402


@pytest.fixture
def appdata(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData"))
    monkeypatch.setattr(shortcuts, "ICON_PATH", tmp_path / "npw.ico")
    return tmp_path


def _lnk(path):
    script = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:L);"
              "\"$($s.TargetPath)|$($s.Arguments)|$($s.IconLocation)\"")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                         env={**os.environ, "L": str(path)},
                         capture_output=True, text=True, check=True).stdout.strip()
    return out.split("|")


def test_install_creates_a_console_less_start_menu_shortcut(appdata):
    created = shortcuts.install()
    paths = shortcuts.shortcut_paths()
    assert created == [paths["start_menu"]] and paths["start_menu"].exists()
    assert not paths["startup"].exists()
    target, args, icon = _lnk(paths["start_menu"])
    assert target.lower() == str(shortcuts.gui_python()).lower()
    assert target.lower().endswith("pythonw.exe")
    assert args == "-m npu_whisper"
    assert icon.lower().startswith(str(shortcuts.ICON_PATH).lower())


def test_autostart_and_remove(appdata):
    shortcuts.install(autostart=True)
    paths = shortcuts.shortcut_paths()
    assert paths["startup"].exists()
    assert sorted(shortcuts.remove()) == sorted(paths.values())
    assert not any(p.exists() for p in paths.values())
    assert not shortcuts.ICON_PATH.exists()
    assert shortcuts.remove() == []


def test_paths_with_quotes_do_not_break_the_script(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path / "it's $(x) dir"))
    monkeypatch.setattr(shortcuts, "ICON_PATH", tmp_path / "npw.ico")
    shortcuts.install()
    assert shortcuts.shortcut_paths()["start_menu"].exists()


def test_second_tray_app_is_refused():
    import ctypes
    from npu_whisper import app
    handles = []
    try:
        assert app._claim_single_instance() is True
        handles.append(app._instance_mutex)
        assert app._claim_single_instance() is False
        handles.append(app._instance_mutex)
    finally:
        for h in handles:
            ctypes.windll.kernel32.CloseHandle(h)
