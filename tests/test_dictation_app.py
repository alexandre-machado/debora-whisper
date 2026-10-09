"""Tests for DictationApp transcription threading."""
import threading
import time
from unittest.mock import MagicMock, patch
import pytest

from debora_whisper.dictation_engine import AppState, DictationApp, DEFAULT_CONFIG


class TestTranscriptionThreading:
    """Verify transcription runs in background thread, not hotkey thread."""

    def test_toggle_recording_stop_does_not_block(self):
        """Stopping recording must return quickly (transcription in background)."""
        config = {**DEFAULT_CONFIG, "beep_on_start": False}
        app = DictationApp(config)

        # Mock the whisper model with a slow transcribe
        app.whisper = MagicMock()
        app.whisper.transcribe = MagicMock(side_effect=lambda *a, **kw: (time.sleep(0.5) or "hello"))
        app._model_ready.set()  # Mark model as loaded

        # Mock recorder to return some audio
        import numpy as np
        app.recorder = MagicMock()
        app.recorder.stop = MagicMock(return_value=np.zeros(16000, dtype=np.float32))

        # Simulate: start recording, then stop
        app.is_recording = True

        finished = threading.Event()
        app.add_callback(lambda s, d: finished.set() if s.value == 'ready' else None)
        with patch('keyboard.is_pressed', return_value=False), patch('debora_whisper.dictation_engine.type_text'):
            start = time.time()
            app.toggle_recording()
            elapsed = time.time() - start
            assert finished.wait(2), 'Transcription worker did not finish'

        # toggle_recording should return quickly (< 200ms), not wait for transcription
        assert elapsed < 0.3, f"toggle_recording blocked for {elapsed:.2f}s — transcription must run in background"

    @pytest.mark.parametrize('shutdown', [False, True])
    def test_inflight_transcription_is_serialized_and_respects_shutdown(self, shutdown):
        import numpy as np
        app = DictationApp({**DEFAULT_CONFIG, 'beep_on_start': False})
        app.recorder = MagicMock()
        app.recorder.stop.return_value = np.zeros(16000, dtype=np.float32)
        app.is_recording = True
        app._model_ready.set()
        entered = threading.Event()
        release = threading.Event()

        def transcribe(*args, **kwargs):
            entered.set()
            assert release.wait(2), 'Test did not release transcription'
            return 'completed speech'

        app.whisper = MagicMock()
        app.whisper.transcribe.side_effect = transcribe
        states = []
        app.add_callback(lambda state, data: states.append(state))
        with patch('debora_whisper.dictation_engine.type_text') as paste, patch('keyboard.unhook_all'):
            worker = threading.Thread(target=app._finish_recording)
            worker.start()
            try:
                assert entered.wait(2)
                # Neither another stop nor another hotkey may start parallel work.
                app._finish_recording()
                app.toggle_recording()
                app.recorder.stop.assert_called_once()
                app.recorder.start.assert_not_called()
                if shutdown:
                    app.stop()
            finally:
                release.set()
                worker.join(2)
            assert not worker.is_alive()
            assert not app._transcribing
            if shutdown:
                paste.assert_not_called()
                assert not app.history
                assert states == [AppState.PROCESSING]
            else:
                paste.assert_called_once_with('completed speech', auto_enter=False)
                assert states == [AppState.PROCESSING, AppState.READY]


def test_voice_chat_delivers_both_sides_to_the_single_overlay():
    import numpy as np
    from debora_whisper.app import GUIApp
    from debora_whisper.ui.overlay import OverlayWindow

    gui = GUIApp.__new__(GUIApp)
    gui._config = {**DEFAULT_CONFIG, "beep_on_start": False}
    gui._root = MagicMock()
    gui._tray = MagicMock()
    gui._stop_audio_polling = MagicMock()
    gui._settings_status = MagicMock()
    gui._settings_set_apply = MagicMock()
    with patch.object(OverlayWindow, "_build"), \
            patch.object(OverlayWindow, "_get_scale", return_value=1.0):
        gui._overlay = OverlayWindow(gui._root)
    gui._overlay._update_layout = MagicMock()
    engine = gui._engine = DictationApp(gui._config)
    engine.recorder = MagicMock()
    engine.add_callback(gui._update_ui)
    during_speech = []

    def respond(text, on_reply):
        assert gui._overlay._conversation_lines()[0][1] == "Você: abre o log"
        on_reply("O log mostra.")
        on_reply("O log mostra. O TTS demorou.")
        during_speech.extend(gui._overlay._conversation_lines())
        assert gui._overlay._balloon_id is None
        return "O log mostra. O TTS demorou."

    with patch.object(engine.voice_chat, "respond", side_effect=respond):
        engine._voice_chat_turn("abre o log", np.zeros(160), True)
    assert [line[1] for line in during_speech] == [
        "Você: abre o log", "Débora: O log mostra. O TTS demorou."]
    assert gui._overlay._conversation_lines() == during_speech
    assert gui._overlay._balloon_id is not None
    gui._update_ui(AppState.READY, {"notice": "Voice chat ready"})
    assert gui._overlay._conversation_lines() == during_speech
    gui._tray.update_state.assert_called_with("ready", "Débora Whisper — Voice chat ready")


@pytest.mark.parametrize("width", [None, 200, 360, 420.5])
def test_balloon_width_config_round_trip(tmp_path, monkeypatch, width):
    from debora_whisper import dictation_engine as de
    from debora_whisper.app import GUIApp

    monkeypatch.setattr(de, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(de, "CONFIG_FILE", tmp_path / "config.json")
    gui = GUIApp.__new__(GUIApp)
    gui._config = DEFAULT_CONFIG.copy()
    gui._on_width_changed(width)
    config = de.load_config()
    de.validate_config(config)
    assert config["balloon_width"] == width
    gui._on_pos_changed(200, 1042)
    assert de.load_config()["balloon_width"] == width


@pytest.mark.parametrize("width", [True, False, 0, -1, 199, "360", float("inf"), float("nan")])
def test_invalid_balloon_width_is_rejected(width):
    from debora_whisper.dictation_engine import validate_config
    with pytest.raises(ValueError, match="balloon_width"):
        validate_config({**DEFAULT_CONFIG, "balloon_width": width})
