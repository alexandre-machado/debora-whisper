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
        self.on_generate = None  # test hook to pause inside inference
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
                if fake.on_generate:
                    fake.on_generate()
                if fake.generate_error:
                    raise RuntimeError(fake.generate_error)
                return "hello"

        self.module = types.ModuleType("openvino_genai")
        self.module.WhisperPipeline = WhisperPipeline


@pytest.fixture
def genai(monkeypatch, tmp_path):
    fake = FakeGenAI(generate_error=CL_ERROR)
    monkeypatch.setitem(sys.modules, "openvino_genai", fake.module)
    fake_ov = types.ModuleType("openvino")
    class Core:
        def get_property(self, device, prop):
            return f"Fake {device}"
    fake_ov.Core = Core
    monkeypatch.setitem(sys.modules, "openvino", fake_ov)

    fake_fw = types.ModuleType("faster_whisper")
    class WhisperModel:
        def __init__(self, model_size_or_path, device, compute_type):
            pass
        def transcribe(self, audio, **kwargs):
            return [SimpleNamespace(text="hello")], {}
    fake_fw.WhisperModel = WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

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


def test_gpu_failure_during_transcription_falls_back(genai, isolate_desktop):
    app = _engine()
    app.ensure_model()
    app._model_ready.set()
    gui = _gui(app)
    states = []
    app.add_callback(lambda state, data: states.append((state, data)))

    with patch.object(app, "fallback_device", wraps=app.fallback_device) as fallback, \
            patch.object(app, "_load_model_background") as reload:
        _transcribe_once(app)

        # The GUI caught the device lost and swapped to NPU
        fallback.assert_called_once_with("NPU")
        reload.assert_called_once()
    assert genai.loads == ["GPU"]
    assert genai.generate_calls == 1

    # App no longer latches GPU failures
    assert de.device_failure() is None


def test_gpu_failure_during_warmup_falls_back_to_npu(genai):
    app = _engine()
    gui = _gui(app)
    states = []
    app.add_callback(lambda state, data: states.append((state, data)))

    with patch.object(app._stopping, "wait", return_value=False), \
            patch.object(app, "fallback_device") as fallback:
        app._load_model_background()

    # The app should fall back to NPU rather than failing closed
    fallback.assert_called_once_with("NPU")
    assert not any(d.get("restart_required") for _, d in states if isinstance(d, dict))
    assert de.device_failure() is None


def test_settings_apply_after_gpu_failure_reloads_normally(genai):
    app = _engine()
    app.ensure_model()
    app._model_ready.set()
    gui = _gui(app)
    gui._settings_status = MagicMock()
    
    with patch.object(app, "_load_model_background"):
        _transcribe_once(app)
        _join_daemons()

    factory = MagicMock()
    with patch.dict(GUIApp._on_settings_apply.__globals__, {
        "DictationApp": factory, "save_config": MagicMock(),
    }) as _:
        gui._on_settings_apply({**gui._config, "device": "CPU"})
        
    # It should successfully instantiate a new engine and reload
    factory.assert_called_once()
    assert gui._config["device"] == "CPU"


def _install_fake_parakeet_runtime(monkeypatch, tmp_path, compile_model):
    """OpenVINO/onnxruntime boundary for a real ParakeetNPU._load_pipeline."""
    class Model:
        def reshape(self, shapes):
            pass

    class Core:
        def read_model(self, path):
            return Model()

        def compile_model(self, model, device, cfg=None):
            return compile_model(device)

    ov = types.ModuleType("openvino")
    ov.Core = Core
    ort = types.ModuleType("onnxruntime")
    ort.InferenceSession = lambda *args, **kwargs: object()
    monkeypatch.setitem(sys.modules, "openvino", ov)
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    (tmp_path / "nemo128.onnx").write_bytes(b"")
    monkeypatch.setattr(de.ParakeetNPU, "_load_vocab", lambda self: None)
    monkeypatch.setattr(de, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(de, "setup_model", lambda config: tmp_path)
    monkeypatch.delenv("PARAKEET_LATENCY_OVERRIDE", raising=False)





def test_parakeet_benign_gpu_fallback_failure_still_reaches_cpu(monkeypatch, tmp_path):
    compiles = []

    def compile_model(device):
        compiles.append(device)
        if device in ("NPU", "GPU"):
            raise RuntimeError(f"[{device}] unsupported layer Foo")
        raise RuntimeError("stop here: CPU reached")

    _install_fake_parakeet_runtime(monkeypatch, tmp_path, compile_model)
    with pytest.raises(RuntimeError, match="CPU reached"):
        de.ParakeetNPU(tmp_path, device="NPU")
    assert compiles == ["NPU", "GPU", "CPU"]
    assert de.device_failure() is None


def _join_daemons():
    for t in threading.enumerate():
        if t is not threading.current_thread() and t.daemon:
            t.join(5)


def test_stop_waits_for_paste_in_progress(genai, isolate_desktop):
    """stop() during the final paste returns only after it finished, so
    nothing can be pasted after stop() has returned."""
    genai.generate_error = None
    app = _engine()
    app.ensure_model()
    app._model_ready.set()
    events, entered, release = [], threading.Event(), threading.Event()

    def paste(text, auto_enter=False):
        events.append("paste_start")
        entered.set()
        assert release.wait(5)
        events.append("paste_end")

    isolate_desktop.side_effect = paste
    worker = threading.Thread(target=_transcribe_once, args=(app,), daemon=True)
    worker.start()
    assert entered.wait(5)

    def stopper():
        app.stop()
        events.append("stop_returned")

    stop_thread = threading.Thread(target=stopper, daemon=True)
    stop_thread.start()
    stop_thread.join(0.3)
    assert stop_thread.is_alive(), "stop() returned while a paste was in progress"
    release.set()
    stop_thread.join(5)
    worker.join(5)
    assert events == ["paste_start", "paste_end", "stop_returned"]
    assert len(app.history) == 1


def test_stop_at_output_boundary_discards_text(genai, isolate_desktop):
    """A stop that completes after inference but right before the final
    check: the text is neither pasted nor stored."""
    genai.generate_error = None
    app = _engine()
    app.ensure_model()
    app._model_ready.set()
    states = []
    app.add_callback(lambda state, data: states.append(state))
    real_lock = app._output_lock

    class StopFirst:
        def __enter__(self):
            app._output_lock = real_lock
            app.stop()  # completes fully before the final check runs
            return real_lock.__enter__()

        def __exit__(self, *exc):
            return real_lock.__exit__(*exc)

    app._output_lock = StopFirst()
    _transcribe_once(app)
    assert genai.generate_calls == 1
    isolate_desktop.assert_not_called()
    assert app.history == []
    assert AppState.READY not in states


def test_settings_rebuild_refused_while_old_inference_runs(genai, isolate_desktop):
    """Old engine is mid-inference on the GPU when Settings asks for a rebuild:
    no config change, no new engine, no stop; after it finishes, Apply works."""
    genai.generate_error = None
    app = _engine()
    app.ensure_model()
    app._model_ready.set()
    gui = _gui(app)
    gui._settings_status = MagicMock()
    gui._settings_set_apply = MagicMock()
    inside, release = threading.Event(), threading.Event()

    def pause():
        inside.set()
        assert release.wait(5)

    genai.on_generate = pause
    worker = threading.Thread(target=_transcribe_once, args=(app,), daemon=True)
    worker.start()
    assert inside.wait(5)

    before = dict(gui._config)
    factory, save = MagicMock(), MagicMock()
    with patch.dict(GUIApp._on_settings_apply.__globals__, {
        "DictationApp": factory, "save_config": save,
    }):
        gui._on_settings_apply({**gui._config, "device": "CPU"})
        factory.assert_not_called()
        save.assert_not_called()
        gui._tray.update_info.assert_not_called()
        assert gui._config == before  # shared with the running engine
        assert not app._stopping.is_set()
        message = gui._settings_status.call_args.args[0]
        assert "transcription in progress" in message and "Apply again" in message

        release.set()
        worker.join(5)
        assert not worker.is_alive()
        isolate_desktop.assert_called_once()
        assert len(app.history) == 1

        gui._on_settings_apply({**gui._config, "device": "CPU"})
        factory.assert_called_once_with(gui._config)
        assert app._stopping.is_set()
    assert gui._config["device"] == "CPU"
    assert genai.loads == ["GPU"]


def test_stop_if_idle_refuses_recording_and_loading():
    app = _engine()
    app.is_recording = True
    assert app.stop_if_idle() == "recording"
    assert not app._stopping.is_set()
    app.is_recording = False

    entered, gate = threading.Event(), threading.Event()

    def slow_load():
        entered.set()
        assert gate.wait(5)

    with patch.object(app, "_load_model_background_inner", side_effect=slow_load):
        app._start_loader()
        assert entered.wait(5)
        assert app.stop_if_idle() == "model loading"
        assert not app._stopping.is_set()
        gate.set()
        _join_daemons()
    assert app.stop_if_idle() is None
    assert app._stopping.is_set()


def test_npu_loss_fallback_loader_stays_busy_after_first_loader_returns(
        genai, monkeypatch, tmp_path):
    """The NPU-loss ERROR callback synchronously starts the GPU retry fallback
    loader before the first loader has returned. Once the first loader is
    done, the second (held pending) must still report 'model loading' and
    refuse a Settings rebuild."""
    genai.generate_error = None
    genai.load_errors = {"NPU": "[NPU] ZE_RESULT_ERROR_DEVICE_LOST"}
    entered, gate = threading.Event(), threading.Event()

    def setup_model(config):
        if not getattr(setup_model, 'first_done', False):
            setup_model.first_done = True
        else:
            # Second load (the fallback to GPU)
            entered.set()
            assert gate.wait(5)
        return tmp_path / "whisper-base"
    setup_model.first_done = False

    monkeypatch.setattr(de, "setup_model", setup_model)
    app = _engine({"device": "NPU"})
    gui = _gui(app)  # synchronous Root: ERROR -> _update_ui -> fallback_device
    states = []
    app.add_callback(lambda state, data: states.append((state, data)))

    with patch.object(app._stopping, "wait", return_value=False), \
         patch.object(gui, "_schedule_npu_recovery"):
        app._load_model_background()  # first loader, returns here
        assert entered.wait(5), "GPU fallback loader never started"
        assert any(d.get("device_failure") == "NPU" for _, d in states)
        assert app.busy_reason() == "model loading"
        assert app.stop_if_idle() == "model loading"
        assert not app._stopping.is_set()
        gate.set()
        _join_daemons()

    # The NPU failed first time, then the second load was instantly attempted on GPU or CUDA
    if app.config["device"] == "CUDA":
        assert genai.loads == ["NPU"]
    else:
        assert genai.loads == ["NPU", "GPU"]
    assert states[-1][0] == AppState.READY
    assert app.busy_reason() is None
    assert de.device_failure() is None
    gui._alert_error.assert_not_called()


def test_loader_thread_start_failure_releases_reservation():
    app = _engine()
    with patch.object(threading.Thread, "start",
                      side_effect=RuntimeError("can't start new thread")):
        with pytest.raises(RuntimeError, match="start new thread"):
            app._start_loader()
    assert app.busy_reason() is None
    assert app.stop_if_idle() is None


def test_ready_callback_may_stop_engine_without_deadlock(genai):
    """READY used to be emitted under _audio_lifecycle_lock; a callback that
    stops the engine (or waits on a thread that does) deadlocked."""
    genai.generate_error = None
    app = _engine()
    stopped = []

    def on_state(state, data):
        if state == AppState.READY:
            app.stop()
            stopped.append(True)

    app.add_callback(on_state)
    with patch.object(app._stopping, "wait", return_value=False):
        loader = threading.Thread(target=app._load_model_background, daemon=True)
        loader.start()
        loader.join(5)
    assert not loader.is_alive(), "READY callback deadlocked against the loader"
    assert stopped == [True]
    assert app._stopping.is_set()
    assert app.busy_reason() is None





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


def test_npu_device_lost_falls_back_to_gpu_and_schedules_npu_recovery(genai):
    genai.generate_error = "[NPU] ZE_RESULT_ERROR_DEVICE_LOST"
    app = _engine({"device": "NPU"})
    app.ensure_model()
    app._model_ready.set()
    gui = _gui(app)
    
    with patch.object(app, "_load_model_background") as reload, \
         patch.object(gui, "_schedule_npu_recovery") as recovery:
        _transcribe_once(app)
        _join_daemons()

    reload.assert_called_once()
    assert app.config["device"] in ("GPU", "CUDA")
    recovery.assert_called_once()
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
