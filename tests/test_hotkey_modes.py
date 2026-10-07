"""Hotkey gestures: hold = push-to-talk, tap = continuous listening."""
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from npu_whisper.dictation_engine import (AppState, AudioRecorder, DEFAULT_CONFIG,
                                          DictationApp, validate_config)


def _wait(cond, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not cond() and time.monotonic() < deadline:
        time.sleep(0.005)
    return cond()


def _app(**config):
    app = DictationApp({**DEFAULT_CONFIG, "beep_on_start": False, **config})
    app.whisper = MagicMock()
    app.whisper.transcribe = MagicMock(return_value="hello world")
    app._model_ready.set()
    app.recorder = MagicMock()
    app.recorder.stop.return_value = np.zeros(16000, dtype=np.float32)
    app.HOLD_SECONDS = 0.05
    states = []
    app.add_callback(lambda s, d: states.append(s))
    return app, states


def _press(app, hold_for=0.0):
    """One hotkey press, released after hold_for seconds."""
    released_at = time.time() + hold_for
    with patch("keyboard.is_pressed", side_effect=lambda _k: time.time() < released_at), \
            patch("npu_whisper.dictation_engine.type_text"):
        app.toggle_recording()
        assert _wait(lambda: not app._hotkey_held)
        # Let a hold's transcription finish under the patches.
        assert _wait(lambda: not app._transcribing)


def test_tap_starts_continuous_listening():
    app, states = _app()
    _press(app)
    assert _wait(lambda: app.continuous_active)
    app.recorder.start.assert_called_once()
    app.recorder.begin_continuous.assert_called_once()
    app.recorder.stop.assert_not_called()  # push-to-talk audio is not transcribed
    app.whisper.transcribe.assert_not_called()
    assert _wait(lambda: states[-1] == AppState.RECORDING)
    assert app.is_recording


def test_second_tap_stops_continuous_listening():
    app, states = _app()
    _press(app)
    assert _wait(lambda: app.continuous_active)
    _press(app)
    assert _wait(lambda: not app.continuous_active)
    app.recorder.end_continuous.assert_called_once()
    assert _wait(lambda: states[-1] == AppState.READY)
    assert not app.is_recording
    app.recorder.start.assert_called_once()  # the stopping tap starts nothing


def test_tap_stops_continuous_even_while_a_segment_transcribes():
    app, _ = _app(continuous_listening=True)
    app.is_recording = True
    app._transcribing = True
    with patch("keyboard.is_pressed", return_value=False):
        app.toggle_recording()
        assert _wait(lambda: not app.continuous_active)
    app.recorder.end_continuous.assert_called_once()


def test_hold_is_push_to_talk():
    app, states = _app()
    _press(app, hold_for=0.15)
    assert not app.continuous_active
    app.recorder.begin_continuous.assert_not_called()
    app.recorder.stop.assert_called_once()
    app.whisper.transcribe.assert_called_once()
    assert AppState.PROCESSING in states and states[-1] == AppState.READY


def test_toggle_tap_action_keeps_the_recording_open():
    app, states = _app(tap_action="toggle")
    _press(app)
    assert app.is_recording and not app.continuous_active
    app.recorder.begin_continuous.assert_not_called()
    _press(app)  # the next tap stops and transcribes
    app.whisper.transcribe.assert_called_once()
    assert states[-1] == AppState.READY


def test_invalid_tap_action_is_rejected():
    with pytest.raises(ValueError):
        validate_config({**DEFAULT_CONFIG, "tap_action": "double"})


def test_recorder_switches_between_push_to_talk_and_vad():
    rec = AudioRecorder(sample_rate=16000, config={})
    rec._ensure_vad_thread = MagicMock()
    rec._write_pos = 32000
    rec.recording = True
    rec._frames = [np.zeros((10, 1), dtype=np.float32)]

    rec.begin_continuous(rewind_seconds=0.5)
    assert rec.continuous and not rec.paused and not rec.recording
    assert rec._frames == []
    assert rec._read_pos == 32000 - 8000  # VAD re-reads the tap
    rec._ensure_vad_thread.assert_called_once()

    rec.end_continuous()
    assert rec.paused and not rec.continuous
