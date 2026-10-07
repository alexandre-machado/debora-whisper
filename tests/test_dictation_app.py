"""Tests for DictationApp transcription threading."""
import threading
import time
from unittest.mock import MagicMock, patch
import pytest

from npu_whisper.dictation_engine import AppState, DictationApp, DEFAULT_CONFIG


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
        with patch('keyboard.is_pressed', return_value=False), patch('npu_whisper.dictation_engine.type_text'):
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
        with patch('npu_whisper.dictation_engine.type_text') as paste, patch('keyboard.unhook_all'):
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
