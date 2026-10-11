"""OS resources follow the operating system's provider."""
import os
import shutil
import subprocess
import sys
from unittest.mock import Mock

import pytest

from debora_whisper import device_queries as queries, dictation_engine as de, platforms, processes
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


def test_windows_nvidia_smi_prefers_system32(monkeypatch, tmp_path):
    system = tmp_path / "Windows" / "System32" / "nvidia-smi.exe"
    system.parent.mkdir(parents=True)
    system.touch()
    monkeypatch.setenv("SystemRoot", str(system.parent.parent))
    which = Mock(return_value=str(tmp_path / "bin" / "nvidia-smi.exe"))
    monkeypatch.setattr(shutil, "which", which)
    assert WindowsPlatform().nvidia_smi() == str(system)
    which.assert_not_called()


@pytest.mark.parametrize("provider", [Platform, WindowsPlatform])
@pytest.mark.parametrize("found", [None, "nvidia-smi.exe", "bin/nvidia-smi.exe", "cwd", "absolute"])
def test_nvidia_smi_path(monkeypatch, tmp_path, provider, found):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SystemRoot", str(tmp_path / "no-windows"))
    expected = str(tmp_path / "bin" / "nvidia-smi.exe") if found == "absolute" else None
    if found == "cwd":
        found = str(tmp_path / "nvidia-smi.exe")
    elif found == "absolute":
        found = str(tmp_path / "bin" / "nvidia-smi.exe")
    which = Mock(return_value=found)
    monkeypatch.setattr(shutil, "which", which)
    assert provider().nvidia_smi() == expected
    which.assert_called_once_with("nvidia-smi")


@pytest.mark.parametrize("found", [None, "absolute-path"])
def test_device_queries_delegates(monkeypatch, found):
    provider = Mock()
    provider.nvidia_smi.return_value = found
    monkeypatch.setattr(platforms, "current", Mock(return_value=provider))
    assert queries.nvidia_smi() == found
    provider.nvidia_smi.assert_called_once_with()


@pytest.mark.parametrize("provider", [Platform, WindowsPlatform])
@pytest.mark.parametrize("path_env", [None, "existing-path"])
@pytest.mark.parametrize("has_add_dll_directory", [False, True])
def test_add_library_dir(monkeypatch, tmp_path, provider, path_env, has_add_dll_directory):
    if path_env is None:
        monkeypatch.delenv("PATH", raising=False)
    else:
        monkeypatch.setenv("PATH", path_env)
    add = Mock()
    if has_add_dll_directory:
        monkeypatch.setattr(os, "add_dll_directory", add, raising=False)
    else:
        monkeypatch.delattr(os, "add_dll_directory", raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "existing-libraries")
    provider().add_library_dir(tmp_path)
    expected = str(tmp_path) + os.pathsep + (path_env or "") if provider is WindowsPlatform else path_env
    assert os.environ.get("PATH") == expected
    assert os.environ["LD_LIBRARY_PATH"] == "existing-libraries"
    if provider is WindowsPlatform and has_add_dll_directory:
        add.assert_called_once_with(str(tmp_path))
        add.return_value.close.assert_not_called()
    else:
        add.assert_not_called()


@pytest.mark.parametrize("return_name", [False, True])
@pytest.mark.parametrize("result", ["success", "missing", "failure"])
def test_has_nvidia_gpu_absolute_path(monkeypatch, tmp_path, return_name, result):
    executable = str(tmp_path / "bin" / "nvidia-smi.exe")
    monkeypatch.setattr(queries, "nvidia_smi", lambda: None if result == "missing" else executable)
    check = Mock(return_value=b"NVIDIA RTX\n")
    if result == "failure":
        check.side_effect = OSError("driver unavailable")
    monkeypatch.setattr(subprocess, "check_output", check)
    expected = "NVIDIA RTX" if return_name else True
    if result != "success":
        expected = None if return_name else False
    assert de.has_nvidia_gpu(return_name) == expected
    if result == "missing":
        check.assert_not_called()
    else:
        args = [executable]
        if return_name:
            args += ["--query-gpu=name", "--format=csv,noheader"]
        check.assert_called_once_with(args, stderr=subprocess.DEVNULL,
                                      creationflags=processes.NO_WINDOW, timeout=None)


@pytest.mark.parametrize("result", ["success", "missing", "failure"])
def test_accelerator_memory_absolute_path(monkeypatch, tmp_path, result):
    executable = str(tmp_path / "bin" / "nvidia-smi.exe")
    monkeypatch.setattr(queries, "nvidia_smi", lambda: None if result == "missing" else executable)
    check = Mock(return_value=b"5324, 8192\n")
    if result == "failure":
        check.side_effect = OSError("driver unavailable")
    monkeypatch.setattr(subprocess, "check_output", check)
    app = de.DictationApp.__new__(de.DictationApp)
    app.config = {"device": "CUDA"}
    assert app._accelerator_memory() == (" VRAM 5.2/8.0G" if result == "success" else "")
    if result == "missing":
        check.assert_not_called()
    else:
        check.assert_called_once_with(
            [executable, "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
            creationflags=processes.NO_WINDOW)


@pytest.mark.parametrize("failure", [False, True])
def test_cuda_pipeline_adds_package_bins_through_provider(monkeypatch, tmp_path, failure):
    import site
    bins = [tmp_path / "nvidia" / package / "bin" for package in ("cublas", "cudnn")]
    for path in bins:
        path.mkdir(parents=True)
    (tmp_path / "nvidia" / "without-bin").mkdir()
    monkeypatch.setattr(site, "getsitepackages", lambda: [str(tmp_path)])
    add = Mock()
    if failure:
        add.side_effect = OSError("DLL directory unavailable")
    monkeypatch.setattr(platforms.current(), "add_library_dir", add)
    model = Mock()
    monkeypatch.setattr(de, "import_faster_whisper", lambda: model)
    monkeypatch.setattr(de, "has_nvidia_gpu", lambda **kwargs: "NVIDIA RTX")
    log = Mock()
    monkeypatch.setattr(de, "log", log)
    pipeline = de.FasterWhisperCUDA("turbo")
    if failure:
        assert add.call_count == 1
        log.assert_any_call("Warning: Failed to inject NVIDIA DLL paths: DLL directory unavailable")
    else:
        assert {call.args[0] for call in add.call_args_list} == set(bins)
        assert add.call_count == 2
    model.assert_called_once_with("turbo", device="cuda", compute_type="int8_float16")
    assert pipeline.pipeline is model.return_value
