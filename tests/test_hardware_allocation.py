"""Actual placements reach the main console, including worker identities."""
import os
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from debora_whisper import hardware_allocation as hw
from debora_whisper import dictation_engine as de


class Core:
    available_devices = ["CPU", "GPU.0", "GPU.1", "NPU"]

    def get_property(self, device, prop):
        if prop == "DEVICE_ID":
            return "0"
        return {"CPU": "Intel CPU", "GPU.0": "Intel Arc (iGPU)",
                "GPU.1": "NVIDIA RTX (dGPU)", "NPU": "Intel AI Boost"}[device]


def test_summary_devices_pids_residency_and_mode_changes(monkeypatch):
    monkeypatch.setattr(hw, "cpu_target", lambda: "CPU Intel CPU")
    config, logs = {"voice_chat": False}, []
    reporter = hw.AllocationReporter()
    reporter.configure(config, logs.append)
    reporter.update("STT", "turbo (faster-whisper) -> CUDA:0 NVIDIA RTX  pid 101")
    assert "Hardware allocation (mode=dictation):" in logs[-1]
    assert "LLM  inactive in dictation" in logs[-1]
    assert "DRAFT STT  shares STT model and pid" in logs[-1]
    reporter.update("TTS", "Chatterbox -> CUDA:0 NVIDIA RTX  server pid 202")
    assert "inactive in dictation; resident: Chatterbox" in logs[-1]
    count = len(logs)
    reporter.emit()
    reporter.update("TTS", "Chatterbox -> CUDA:0 NVIDIA RTX  server pid 202")
    assert len(logs) == count
    config.update(voice_chat=True, voice_chat_backend="claude")
    reporter.emit()
    assert len(logs) == count + 1
    assert "mode=chat_claude" in logs[-1]
    assert "external Claude Code" in logs[-1]
    assert "server pid 202" in logs[-1]


def test_openvino_alias_and_loaded_graph_disambiguate_gpus():
    assert hw.ov_target(Core(), "GPU") == "OV:GPU.0 Intel Arc (iGPU)"
    graph = SimpleNamespace(get_property=lambda prop: ["GPU.1"])
    assert hw.ov_target(Core(), "GPU", graph) == "OV:GPU.1 NVIDIA RTX (dGPU)"


def test_missing_identity_is_explicit_not_guessed():
    core = SimpleNamespace(get_property=Mock(side_effect=RuntimeError("unsupported")))
    assert "unavailable" in hw.ov_target(core, "GPU")
    assert "GPU.0" not in hw.ov_target(core, "GPU")


def test_load_time_fallback_uses_actual_whisper_device(monkeypatch, tmp_path):
    calls = []

    def pipeline(path, device, **kwargs):
        calls.append(device)
        if device == "NPU":
            raise RuntimeError("unsupported operation")
        return object()

    monkeypatch.setitem(sys.modules, "openvino_genai", SimpleNamespace(WhisperPipeline=pipeline))
    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=Core))
    monkeypatch.setattr(de, "CACHE_DIR", tmp_path)
    model = de.WhisperNPU(tmp_path, "NPU")
    detail = hw.model_description(model, "turbo", "NPU")
    assert calls == ["NPU", "CPU"]
    assert "-> OV:CPU Intel CPU" in detail
    assert "planned NPU, fell back to OV:CPU Intel CPU: NPU: unsupported operation" in detail
    assert f"pid {os.getpid()}" in detail


def test_runtime_device_loss_then_reload_keeps_original_reason(monkeypatch):
    config = {**de.DEFAULT_CONFIG, "device": "NPU"}
    logs = []
    hw.allocation.configure(config, logs.append)
    app = de.DictationApp.__new__(de.DictationApp)
    app.config = config
    app.whisper = SimpleNamespace(device="NPU")
    app._model_lock = threading.RLock()
    app._model_ready = threading.Event()
    app._quarantined_models = []
    app._lost_device = None
    app._load_error = None
    monkeypatch.setattr(app, "_start_loader", lambda: None)
    monkeypatch.setattr(de, "setup_model", lambda config: Path("model"))
    monkeypatch.setattr(de, "create_model", lambda *a, **k: SimpleNamespace(
        device="GPU", hardware_target="OV:GPU.0 Intel Arc (iGPU)",
        hardware_backend="openvino-genai", hardware_fallbacks=[]))
    app._error_payload(de.DeviceFailureError("NPU", RuntimeError("ZE_RESULT_ERROR_DEVICE_LOST")))
    assert "device lost/quarantined" in logs[-1]
    app.fallback_device("GPU")
    app.ensure_model()
    assert "planned NPU, fell back to OV:GPU.0" in logs[-1]
    assert "ZE_RESULT_ERROR_DEVICE_LOST" in logs[-1]
    count = len(logs)
    app.ensure_model()
    assert len(logs) == count


def test_optional_device_identity_failure_does_not_reload_whisper(monkeypatch, tmp_path):
    pipeline = Mock(return_value=object())
    monkeypatch.setitem(sys.modules, "openvino_genai", SimpleNamespace(WhisperPipeline=pipeline))
    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=Mock(side_effect=RuntimeError("query failed"))))
    monkeypatch.setattr(de, "CACHE_DIR", tmp_path)
    model = de.WhisperNPU(tmp_path, "GPU")
    assert model.device == "GPU"
    assert "identity unreported" in model.hardware_target
    pipeline.assert_called_once()


@pytest.mark.parametrize("cuda_available,identity_fails", [(False, False), (True, False), (True, True)])
def test_tts_loader_reports_its_own_device_and_fallback(monkeypatch, cuda_available, identity_fails):
    from debora_whisper import tts_server
    from types import ModuleType
    model = SimpleNamespace(conds=None)
    torch = ModuleType("torch")
    torch._dynamo = ModuleType("torch._dynamo")
    torch.cuda = SimpleNamespace(is_available=lambda: cuda_available, current_device=lambda: 1,
        get_device_name=Mock(side_effect=RuntimeError("query failed")) if identity_fails else lambda i: "RTX worker GPU")
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch._dynamo", torch._dynamo)
    monkeypatch.setitem(sys.modules, "chatterbox", ModuleType("chatterbox"))
    monkeypatch.setitem(sys.modules, "chatterbox.mtl_tts",
                        SimpleNamespace(ChatterboxMultilingualTTS=SimpleNamespace(from_pretrained=factory)))
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setattr(tts_server, "_stub_sklearn", lambda: None)
    tts_server.load_model("cuda", None)
    assert model._debora_allocation["pid"] == os.getpid()
    if not cuda_available:
        assert model._debora_device == "CPU"
        assert model._debora_allocation["device_id"] == "CPU"
        assert "CUDA unavailable" in model._debora_allocation["fallback_reason"]
    elif identity_fails:
        assert model._debora_device == "CUDA"
        assert "unreported" in model._debora_allocation["device_id"]
    else:
        assert model._debora_allocation["device_id"] == "CUDA:1"
        assert model._debora_allocation["device_name"] == "RTX worker GPU"
    factory.assert_called_once_with(device="cuda" if cuda_available else "cpu")


def test_parakeet_reports_mixed_encoder_buckets_and_decoder_fallback(monkeypatch, tmp_path):
    import numpy as np

    class Graph:
        def __init__(self, device):
            self.device = device

        def get_property(self, prop):
            return ["GPU.1" if self.device == "GPU" else self.device]

        def __call__(self, inputs):
            return {"outputs": np.zeros((1, 1024, 3))}

    class Runtime(Core):
        encoder_loads = 0

        def read_model(self, path):
            return SimpleNamespace(decoder="decoder" in path, reshape=lambda shape: None)

        def compile_model(self, model, device, config=None):
            if model.decoder and device == "GPU":
                raise RuntimeError("decoder unsupported")
            if not model.decoder and device == "NPU":
                self.encoder_loads += 1
                if self.encoder_loads > 1:
                    raise RuntimeError("bucket unsupported")
            return Graph(device)

    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=Runtime))
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=lambda *a, **k: object()))
    monkeypatch.setattr(de.ParakeetNPU, "_load_vocab", lambda self: None)
    monkeypatch.setattr(de.ParakeetNPU, "_validate_vocab_against_decoder", lambda self: None)
    monkeypatch.setattr(de.ParakeetNPU, "MEL_BUCKETS", (10, 20))
    monkeypatch.setattr(de, "CACHE_DIR", tmp_path / "cache")
    (tmp_path / "nemo128.onnx").touch()
    model = de.ParakeetNPU(tmp_path, "NPU")
    detail = hw.model_description(model, "parakeet", "NPU")
    assert "bucket 10=OV:NPU Intel AI Boost" in detail
    assert "bucket 20=OV:GPU.1 NVIDIA RTX (dGPU)" in detail
    assert "decoder OV:CPU Intel CPU" in detail
    assert "mel preprocessor CPU" in detail
    assert "bucket unsupported" in detail and "decoder unsupported" in detail


def test_tts_health_uses_server_pid_not_launcher_and_is_deduplicated(monkeypatch):
    from debora_whisper import tts_server, voice_chat
    model = SimpleNamespace(sr=24000, conds=None, _debora_device="CPU",
        _debora_allocation={"pid": 4321, "device_id": "CPU", "device_name": "Intel CPU",
                            "requested_device": "CUDA", "fallback_reason": "CUDA unavailable in TTS worker"})
    server = ThreadingHTTPServer(("127.0.0.1", 0), tts_server.make_handler(model, "pt"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = {"voice_chat": True, "tts_url": f"http://127.0.0.1:{server.server_port}"}
    logs = []
    hw.allocation.configure(config, logs.append)
    monkeypatch.setattr(voice_chat, "_tts_process", SimpleNamespace(pid=9999))
    try:
        assert voice_chat.tts_server_up(config)
        assert "ChatterboxMultilingualTTS" in logs[-1]
        assert "server pid 4321" in logs[-1]
        assert "9999" not in logs[-1]
        assert "planned CUDA, fell back to CPU: CUDA unavailable in TTS worker" in logs[-1]
        assert voice_chat.tts_server_up(config)
        assert len(logs) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)


def test_external_tts_never_borrows_local_hardware():
    logs = []
    config = {"voice_chat": True, "tts_url": "https://voice.example"}
    hw.allocation.configure(config, logs.append)
    hw.report_tts(config, None)
    assert "external endpoint=https://voice.example; device/model/pid unreported" in logs[-1]
    assert "CUDA" not in logs[-1]


def test_claude_protocol_reports_actual_external_model_and_harness_pid():
    import io
    import queue
    from debora_whisper import harness
    logs = []
    hw.allocation.configure({"voice_chat": True, "voice_chat_backend": "claude"}, logs.append)
    session = harness.HarnessSession.__new__(harness.HarnessSession)
    session.process = SimpleNamespace(pid=7654, stdout=io.StringIO(
        '{"type":"system","subtype":"init","model":"claude-test"}\n'))
    session._events = queue.Queue()
    session.error = None
    session.log = logs.append
    session._read()
    assert "external Claude Code; model=claude-test" in logs[0]
    assert "harness pid 7654 (CPU orchestration)" in logs[0]
    assert "endpoint=Claude Code managed" in logs[0]
    assert "harness exited" in logs[-1]


def test_summary_console_and_app_log_match(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(de, "LOG_DIR", tmp_path)
    monkeypatch.setattr(de, "LOG_FILE", tmp_path / "app.log")
    block = hw.format_summary({}, {"STT": "turbo -> CUDA:0 NVIDIA RTX  pid 1234"})
    de.log(block)
    console = capsys.readouterr().out
    assert block in console
    assert console == de.LOG_FILE.read_text(encoding="utf-8")


def test_closed_console_does_not_drop_app_log(tmp_path, monkeypatch):
    monkeypatch.setattr(de, "LOG_DIR", tmp_path)
    monkeypatch.setattr(de, "LOG_FILE", tmp_path / "app.log")
    monkeypatch.setattr("builtins.print", Mock(side_effect=OSError("closed")))
    de.log("Hardware allocation (mode=dictation):")
    assert "Hardware allocation" in de.LOG_FILE.read_text(encoding="utf-8")


def test_npu_probe_forwards_temporary_worker_allocation(monkeypatch):
    import json
    from debora_whisper import npu_probe
    logs = []
    detail = "turbo (openvino-genai) -> OV:NPU Intel AI Boost  pid 8765"
    process = SimpleNamespace(returncode=0, pid=8765,
        communicate=lambda **kw: (("@allocation " + json.dumps(detail) + "\n").encode(), None))
    monkeypatch.setattr(npu_probe.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(de, "log", logs.append)
    npu_probe.probe_npu({}, "model", command=["fake"])
    assert "mode=npu_probe" in logs[-1]
    assert detail in logs[-1]
    assert "temporary worker now exited" in logs[-1]
