"""Settings and managed TTS switching, without a display, servers or models."""
import threading
from unittest.mock import MagicMock

import pytest

from debora_whisper import dictation_engine as de
from debora_whisper import voice_chat as vc
from debora_whisper.ui.settings import (
    SettingsWindow, _DEFAULT_VOICE, _TTS_ENGINES, _TTS_ENGINE_KEYS, _make_wide_dropdown,
)


@pytest.mark.parametrize("backend,label", [
    ("chatterbox", "Chatterbox (clones the voice)"),
    ("qwen3", "Qwen3-TTS (faster, clones the voice)"),
    ("kokoro", "Kokoro (fastest, pt-BR voice pf_dora)"),
])
@pytest.mark.parametrize("voice", [_DEFAULT_VOICE, "isabel"])
def test_settings_save_engine_and_voice(backend, label, voice, monkeypatch):
    assert _TTS_ENGINES[backend] == label
    assert _TTS_ENGINE_KEYS[label] == backend
    window = SettingsWindow.__new__(SettingsWindow)
    window._config = dict(de.DEFAULT_CONFIG)
    for attr, value in {
        "_model_radio_var": window._config["model_size"],
        "_hotkey_var": window._config["hotkey"],
        "_beep_var": True, "_enter_var": False, "_inline_drafts_var": False,
        "_voice_chat_var": True, "_backend_var": "Local (Qwen)",
        "_harness_new_session_var": True, "_harness_session_name_var": "Debora",
        "_voice_var": voice, "_balloon_var": True, "_font_size_var": "16",
        "_tts_engine_var": label,
    }.items():
        setattr(window, attr, MagicMock(get=MagicMock(return_value=value)))
    window._harness_cwd = None
    window._get_selected_lang_code = lambda: "pt"
    window._on_apply = MagicMock()
    monkeypatch.setattr(de, "detect_devices", lambda: ["CPU"])
    monkeypatch.setattr(de, "select_device", lambda *args: "CPU")
    window._apply()
    saved = window._on_apply.call_args.args[0]
    assert saved["tts_backend"] == backend
    assert saved["tts_voice"] == (None if voice == _DEFAULT_VOICE else voice)


def test_kokoro_disables_voice_and_other_engines_restore_it():
    window = SettingsWindow.__new__(SettingsWindow)
    window._voice_dropdown = MagicMock()
    for backend in ("kokoro", "qwen3", "kokoro", "chatterbox"):
        window._on_tts_engine_change(_TTS_ENGINES[backend])
        window._voice_dropdown.configure.assert_called_with(
            state="disabled" if backend == "kokoro" else "normal")


def test_disabled_wide_dropdown_does_not_open():
    menu = MagicMock()
    menu.cget.return_value = "disabled"
    _make_wide_dropdown(menu, ["voice"], None)
    menu._open_dropdown_menu()
    menu.winfo_rootx.assert_not_called()


@pytest.fixture
def managed_tts(monkeypatch):
    monkeypatch.setattr(vc, "_tts_process", None)
    monkeypatch.setattr(vc, "_tts_server_key", None)
    monkeypatch.setattr(vc, "_tts_stopping", threading.Event())
    monkeypatch.setattr(vc.shutil, "which", lambda name: "uv.exe")
    monkeypatch.setattr(vc, "tts_server_up", MagicMock(return_value=False))
    monkeypatch.setattr(vc.atexit, "register", MagicMock())
    kill = MagicMock()
    monkeypatch.setattr(vc, "kill_tree", kill)
    processes = [MagicMock(), MagicMock()]
    for process in processes:
        process.poll.return_value = None
    popen = MagicMock(side_effect=processes)
    monkeypatch.setattr(vc.subprocess, "Popen", popen)
    return popen, kill, processes


@pytest.mark.parametrize("backend", ["qwen3", "kokoro"])
def test_backend_change_stops_old_server_before_starting_new(managed_tts, backend):
    popen, kill, processes = managed_tts
    calls = MagicMock()
    calls.attach_mock(popen, "start")
    calls.attach_mock(kill, "stop")
    vc.ensure_tts_server(de.DEFAULT_CONFIG)
    vc.ensure_tts_server({**de.DEFAULT_CONFIG, "tts_backend": backend})
    kill.assert_called_once_with(processes[0])
    processes[0].wait.assert_called_once_with(5)
    assert [call[0] for call in calls.mock_calls] == ["start", "stop", "start"]
    assert str(vc.TTS_SERVER_SCRIPTS[backend]) in popen.call_args.args[0]
    assert vc._tts_process is processes[1]
    assert not vc._tts_stopping.is_set()


@pytest.mark.parametrize("backend", ["chatterbox", "qwen3", "kokoro"])
def test_same_backend_and_command_reuse_server(managed_tts, backend):
    popen, kill, _ = managed_tts
    config = {**de.DEFAULT_CONFIG, "tts_backend": backend}
    vc.ensure_tts_server(config)
    vc.ensure_tts_server({**config, "show_balloon": False})
    popen.assert_called_once()
    kill.assert_not_called()


def test_qwen_voice_change_restarts_startup_clone(managed_tts):
    popen, kill, processes = managed_tts
    config = {**de.DEFAULT_CONFIG, "tts_backend": "qwen3"}
    vc.ensure_tts_server(config)
    vc.ensure_tts_server({**config, "tts_voice": "isabel"})
    kill.assert_called_once_with(processes[0])
    assert popen.call_count == 2
    command = popen.call_args.args[0]
    assert command[command.index("--voice") + 1].endswith("isabel.wav")


def test_kokoro_ignores_voice_change(managed_tts):
    popen, kill, _ = managed_tts
    config = {**de.DEFAULT_CONFIG, "tts_backend": "kokoro"}
    vc.ensure_tts_server(config)
    vc.ensure_tts_server({**config, "tts_voice": "isabel"})
    popen.assert_called_once()
    kill.assert_not_called()


def test_external_server_is_kept_and_backend_change_logged(managed_tts):
    popen, kill, _ = managed_tts
    vc.tts_server_up.return_value = True
    logs = []
    vc.ensure_tts_server(de.DEFAULT_CONFIG, logs.append)
    vc.ensure_tts_server({**de.DEFAULT_CONFIG, "tts_backend": "kokoro"}, logs.append)
    popen.assert_not_called()
    kill.assert_not_called()
    assert vc._tts_process is None
    assert any("kokoro" in line and "external server" in line for line in logs)


def test_engine_loader_restarts_tts_off_the_calling_thread(managed_tts, monkeypatch):
    popen, kill, _ = managed_tts
    vc.ensure_tts_server(de.DEFAULT_CONFIG)
    threads = []
    kill.side_effect = lambda process: threads.append(threading.get_ident())
    monkeypatch.setattr(de, "ensure_tts_server", lambda config, *args: vc.ensure_tts_server(config))
    engine = de.DictationApp.__new__(de.DictationApp)
    engine.config = {**de.DEFAULT_CONFIG, "tts_backend": "kokoro",
                     "voice_chat_backend": "claude", "voice_chat": False}
    engine._voice_chat_loading = threading.Lock()
    engine.voice_chat = MagicMock()
    done = threading.Event()
    # Signal after the entire loader has returned, before fixtures are restored.
    load = engine._load_voice_chat

    def load_and_signal():
        try:
            load()
        finally:
            done.set()

    engine._load_voice_chat = load_and_signal
    engine._warm_up_voice_chat()
    assert done.wait(5)
    assert len(threads) == 1 and threads[0] != threading.get_ident()
    assert popen.call_count == 2
    engine.voice_chat.warm_tts.assert_called_once()
