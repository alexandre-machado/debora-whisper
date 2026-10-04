"""Regression tests for the fail-closed accelerator recovery policy (issue #11).

A user hit OpenVINO GPU CL_OUT_OF_RESOURCES after 16.3 s of audio. OpenVINO
warns that later OpenCL calls may hang, yet the GUI reloaded the same GPU.
These tests drive the real engine (WhisperNPU / DictationApp) into the real
GUI consumer (GUIApp._on_state_change / _update_ui). Only the hardware
boundaries are faked: the openvino_genai pipeline, the microphone recorder,
the Tk root/tray/overlay widgets and the modal dialog.
"""
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import dictation_engine as de
from dictation_engine import AppState, DictationApp, DEFAULT_CONFIG
from tests.test_app_shutdown import GUIApp

CL_ERROR = (
    "Exception from src/inference/src/cpp/infer_request.cpp:223:\n"
    "[GPU] clWaitForEvents, error code: -5 CL_OUT_OF_RESOURCES. "
    "The OpenCL context may be in an unrecoverable state; "
    "subsequent OpenCL calls may hang."
)


class FakeGenAI:
    """Stand-in for the openvino_genai module (the GPU/NPU boundary)."""

    def __init__(self, generate_error=None, load_errors=None):
        self.generate_error = generate_error
        self.load_errors = dict(load_errors or {})
        self.loads = []
        self.generate_calls = 0
        fake = self

        class WhisperPipeline:
            def __init__(self, path, device, **kwargs):
                fake.loads.append(device)
                if device in fake.load_errors:
                    raise RuntimeError(fake.load_errors[device])

            def get_generation_config(self):
                return SimpleNamespace()

            def generate(self, audio, config):
                fake.generate_calls += 1
                if fake.generate_error:
                    raise RuntimeError(fake.generate_error)
                return "hello"

        self.module = types.ModuleType("openvino_genai")
        self.module.WhisperPipeline = WhisperPipeline


@pytest.fixture
def genai(monkeypatch, tmp_path):
    fake = FakeGenAI(generate_error=CL_ERROR)
    monkeypatch.setitem(sys.modules, "openvino_genai", fake.module)
    monkeypatch.setattr(de, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(de, "setup_model", lambda config: tmp_path / "whisper-base")
    return fake


@pytest.fixture(autouse=True)
def isolate_desktop():
    with patch("keyboard.is_pressed", return_value=False), \
            patch("keyboard.unhook_all"), patch("dictation_engine.type_text") as paste:
        yield paste


class FakeRecorder:
    """Microphone boundary: returns 16.3 s of audio like the user's report."""
    _recording_generation = 0
    telemetry = {"live_frames": 16000}

    def __init__(self):
        self.starts = 0

    def warmup(self):
        pass

    def wait_ready(self, timeout=3.0):
        pass

    def start(self):
        self.starts += 1

    def stop(self):
        return np.zeros(int(16000 * 16.3), dtype=np.float32)

    def close(self):
        pass


class Root:
    def after(self, delay, fn, *args):
        fn(*args)
        return "after-id"

    def after_cancel(self, _id):
        pass


def _engine(config=None):
    app = DictationApp({**DEFAULT_CONFIG, "device": "GPU", "beep_on_start": False,
                        **(config or {})})
    app.recorder = FakeRecorder()
    return app


def _gui(engine):
    gui = GUIApp.__new__(GUIApp)
    gui._config = engine.config
    gui._root = Root()
    gui._engine = engine
    gui._tray = MagicMock()
    gui._overlay = MagicMock()
    gui._settings_win = None
    gui._audio_poll_id = None
    gui._alert_error = MagicMock()
    engine.add_callback(gui._on_state_change)
    return gui


def _transcribe_once(app):
    app.is_recording = True
    app._finish_recording()


def test_gpu_failure_during_transcription_fails_closed(genai, isolate_desktop):
    app = _engine()
    app.ensure_model()
    app._model_ready.set()
    gui = _gui(app)
    states = []
    app.add_callback(lambda state, data: states.append((state, data)))

    with patch.object(app, "fallback_device", wraps=app.fallback_device) as fallback, \
            patch.object(app, "_load_model_background") as reload:
        _transcribe_once(app)

        # The GUI did not reload anything, on GPU or elsewhere.
        fallback.assert_not_called()
        reload.assert_not_called()
    assert genai.loads == ["GPU"]
    assert genai.generate_calls == 1

    state, data = states[-1]
    assert state == AppState.ERROR
    assert data["restart_required"] is True
    assert data["device_failure"] == "GPU"
    assert "CL_OUT_OF_RESOURCES" in data["cause"]
    assert "restart" in data["error"].lower()

    # Original OpenVINO error is the preserved cause.
    latched = de.device_failure()
    assert isinstance(latched["exception"], de.DeviceFailureError)
    assert CL_ERROR in str(latched["exception"].__cause__)

    # Clear, actionable GUI instruction shown exactly once.
    gui._alert_error.assert_called_once()
    title, message = gui._alert_error.call_args.args
    assert "restart" in title.lower()
    assert "Quit and restart" in message and "CL_OUT_OF_RESOURCES" in message
    tooltip = gui._tray.update_state.call_args.args[1]
    assert "GPU failed" in tooltip and "restart" in tooltip

    # Further hotkeys neither record nor infer, and do not re-alert.
    app.toggle_recording()
    _transcribe_once(app)
    for t in threading.enumerate():
        if t is not threading.current_thread() and t.daemon:
            t.join(1)
    assert app.recorder.starts == 0
    assert genai.generate_calls == 1
    isolate_desktop.assert_not_called()
    gui._alert_error.assert_called_once()


def test_gpu_failure_during_warmup_blocks_rebuilt_engines(genai):
    app = _engine()
    gui = _gui(app)
    states = []
    app.add_callback(lambda state, data: states.append((state, data)))

    with patch.object(app._stopping, "wait", return_value=False):
        app._load_model_background()

    assert states[-1][0] == AppState.ERROR
    assert states[-1][1]["restart_required"] is True
    assert app._load_error and "restart" in app._load_error.lower()
    gui._alert_error.assert_called_once()

    # Neither an explicit fallback nor a brand-new engine (what Settings
    # creates) may load a model in this process again, even on CPU.
    app.fallback_device("CPU")
    fresh = _engine({"device": "CPU"})
    fresh_states = []
    fresh.add_callback(lambda state, data: fresh_states.append((state, data)))
    fresh._load_model_background()
    assert fresh_states[-1][1]["restart_required"] is True
    with pytest.raises(de.RestartRequiredError):
        fresh.ensure_model()
    assert genai.loads == ["GPU"]
    assert genai.generate_calls == 1


def test_settings_apply_after_gpu_failure_saves_without_reload(genai):
    app = _engine()
    app.ensure_model()
    app._model_ready.set()
    gui = _gui(app)
    gui._settings_status = MagicMock()
    _transcribe_once(app)

    factory = MagicMock()
    with patch.dict(GUIApp._on_settings_apply.__globals__, {
        "DictationApp": factory, "save_config": MagicMock(),
    }) as _:
        gui._on_settings_apply({**gui._config, "device": "CPU"})
    factory.assert_not_called()
    assert gui._config["device"] == "CPU"  # persisted for the next start
    assert "Restart" in gui._settings_status.call_args.args[0]
    assert genai.loads == ["GPU"]


def test_gpu_load_failure_does_not_fall_back_to_cpu(genai, tmp_path):
    genai.load_errors = {"GPU": CL_ERROR}
    with pytest.raises(de.DeviceFailureError) as info:
        de.WhisperNPU(tmp_path, device="GPU")
    assert info.value.device == "GPU"
    assert CL_ERROR in str(info.value.__cause__)
    assert genai.loads == ["GPU"]


def test_benign_load_failure_keeps_cpu_fallback(genai, tmp_path):
    genai.load_errors = {"GPU": "[GPU] unsupported operation Foo"}
    model = de.WhisperNPU(tmp_path, device="GPU")
    assert model.device == "CPU"
    assert genai.loads == ["GPU", "CPU"]
    assert de.device_failure() is None


def test_npu_device_lost_still_falls_back_to_gpu(genai):
    genai.generate_error = "[NPU] ZE_RESULT_ERROR_DEVICE_LOST"
    app = _engine({"device": "NPU"})
    app.ensure_model()
    app._model_ready.set()
    gui = _gui(app)
    with patch.object(app, "_load_model_background") as reload:
        _transcribe_once(app)
        for t in threading.enumerate():
            if t is not threading.current_thread() and t.daemon:
                t.join(1)
    reload.assert_called_once()
    assert app.config["device"] == "GPU"
    assert de.device_failure() is None
    gui._alert_error.assert_not_called()


def test_classification_uses_backend_devices_not_config():
    lost = RuntimeError("DEVICE_LOST")
    # Configured NPU, but Parakeet's decoder ran on GPU: fail closed.
    parakeet = de.ParakeetNPU.__new__(de.ParakeetNPU)
    parakeet.device, parakeet.dec_device = "NPU", "GPU"
    assert de.classify_device_failure(lost, parakeet.active_devices()) == "UNKNOWN"
    parakeet.dec_device = "CPU"
    assert de.classify_device_failure(lost, parakeet.active_devices()) == "NPU"
    # Whisper that fell back to CPU at load is not blamed on a GPU/NPU name.
    assert de.classify_device_failure(RuntimeError(CL_ERROR), {"CPU"}) == "GPU"
    assert de.classify_device_failure(RuntimeError("bad input shape"), {"GPU"}) is None
    # The cause chain is searched, not only the outermost message.
    try:
        try:
            raise RuntimeError(CL_ERROR)
        except RuntimeError as inner:
            raise ValueError("wrapped") from inner
    except ValueError as outer:
        assert de.classify_device_failure(outer, {"NPU"}) == "GPU"


def test_parakeet_inference_failure_is_attributed_to_gpu_decoder():
    parakeet = de.ParakeetNPU.__new__(de.ParakeetNPU)
    parakeet.device, parakeet.dec_device = "NPU", "GPU"
    parakeet._preprocess = lambda audio: (np.zeros((1, 128, 100), np.float32), None)
    parakeet.enc_compiled = {b: (lambda inputs: {
        "outputs": np.zeros((1, 1024, 10), np.float32),
        "encoded_lengths": np.array([10]),
    }) for b in de.ParakeetNPU.MEL_BUCKETS}

    def decoder(inputs):
        raise RuntimeError("[GPU] DEVICE_LOST while executing decoder")

    parakeet.dec_compiled = decoder
    parakeet.vocab = {}
    with pytest.raises(de.DeviceFailureError) as info:
        parakeet.transcribe(np.zeros(16000, np.float32))
    assert info.value.device == "GPU"
