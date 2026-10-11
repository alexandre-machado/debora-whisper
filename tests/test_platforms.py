"""Process handling follows the operating system's provider."""
import subprocess
import sys
from unittest.mock import Mock

import pytest

from debora_whisper import platforms, processes
from debora_whisper.platforms.base import Platform
from debora_whisper.platforms.windows import WindowsPlatform


@pytest.mark.parametrize("returncode", [None, 0, 1])
def test_generic_kill_tree(returncode):
    process = Mock()
    process.poll.return_value = returncode
    Platform().kill_tree(process)
    if returncode is None:
        process.kill.assert_called_once_with()
    else:
        process.kill.assert_not_called()


@pytest.mark.parametrize("returncode", [None, 0, 1])
def test_windows_kill_tree(monkeypatch, returncode):
    # The provider module also imports on systems without Windows constants.
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    run = Mock()
    monkeypatch.setattr(subprocess, "run", run)
    process = Mock(pid=1234)
    process.poll.return_value = returncode
    WindowsPlatform().kill_tree(process)
    if returncode is None:
        run.assert_called_once_with(
            ["taskkill", "/T", "/F", "/PID", "1234"],
            capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        run.assert_not_called()
    process.kill.assert_not_called()


def test_no_window_comes_from_provider():
    assert Platform().no_window == 0
    assert processes.NO_WINDOW == platforms.current().no_window


def test_kill_tree_delegates_to_current_provider(monkeypatch):
    provider = Mock()
    current = Mock(return_value=provider)
    monkeypatch.setattr(platforms, "current", current)
    process = Mock()
    processes.kill_tree(process)
    current.assert_called_once_with()
    provider.kill_tree.assert_called_once_with(process)
    process.poll.assert_not_called()
    process.kill.assert_not_called()


def test_audio_fixture_keeps_real_provider():
    provider = platforms.current()
    assert type(provider) is (WindowsPlatform if sys.platform == "win32" else Platform)
    assert provider.default_output_name() is None
    assert provider.output_is_headphones("Headphones") is None
