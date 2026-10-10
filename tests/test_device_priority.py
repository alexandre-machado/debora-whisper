"""config["device_priority"] picks the startup device and the fallback."""
import pytest

from debora_whisper import dictation_engine as de

ALL = {"CUDA", "NPU", "GPU", "CPU"}


def _cfg(model="turbo", priority=("CUDA", "NPU", "GPU", "CPU")):
    return {**de.DEFAULT_CONFIG, "model_size": model, "device_priority": list(priority)}


def test_default_priority_puts_rtx_first():
    assert de.DEFAULT_CONFIG["device_priority"][:2] == ["CUDA", "NPU"]


def test_first_present_device_wins():
    assert de.select_device(_cfg(), ALL) == "CUDA"
    assert de.select_device(_cfg(), {"NPU", "GPU", "CPU"}) == "NPU"


def test_failed_device_falls_back_to_next_in_priority():
    assert de.select_device(_cfg(), ALL, exclude={"CUDA"}) == "NPU"
    assert de.select_device(_cfg(), ALL, exclude={"NPU"}) == "CUDA"


def test_user_order_is_respected():
    assert de.select_device(_cfg(priority=["NPU", "CUDA"]), ALL) == "NPU"


def test_parakeet_skips_cuda():
    assert de.select_device(_cfg(model="parakeet"), ALL) == "NPU"


def test_nothing_usable_returns_none():
    assert de.select_device(_cfg(priority=["CUDA"]), {"CPU"}) is None


def test_apply_device_priority_sets_device(monkeypatch):
    monkeypatch.setattr(de, "detect_devices", lambda: {"NPU", "CPU"})
    config = _cfg()
    de.apply_device_priority(config)
    assert config["device"] == "NPU"


@pytest.mark.parametrize("priority", [[], ["RTX"], "CUDA"])
def test_invalid_priority_is_rejected(priority):
    config = {**_cfg(), "device_priority": priority}
    with pytest.raises(ValueError):
        de.validate_config(config)


@pytest.mark.parametrize("has_backend", [True, False])
def test_cuda_requires_the_cuda_extra(monkeypatch, has_backend):
    # A plain install on an NVIDIA box has no faster-whisper: CUDA must not be
    # offered, or the default priority would pick it and fail to load.
    import importlib.util
    import sys
    import types
    monkeypatch.setattr(de, "has_nvidia_gpu", lambda return_name=False: True)
    monkeypatch.setitem(sys.modules, "openvino", types.SimpleNamespace(
        Core=lambda: types.SimpleNamespace(available_devices=["CPU", "NPU"])))
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: (
        (object() if has_backend else None) if name == "faster_whisper"
        else real_find_spec(name, *a)))
    monkeypatch.setattr(de, "import_faster_whisper", lambda: object)
    devices = de.detect_devices()
    assert ("CUDA" in devices) is has_backend
    assert de.select_device(_cfg(), devices) == ("CUDA" if has_backend else "NPU")


def _nvidia_with_faster_whisper(monkeypatch):
    import importlib.util
    import sys
    import types
    monkeypatch.setattr(de, "has_nvidia_gpu", lambda return_name=False: True)
    monkeypatch.setitem(sys.modules, "openvino", types.SimpleNamespace(
        Core=lambda: types.SimpleNamespace(available_devices=["CPU", "NPU"])))
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: (
        object() if name == "faster_whisper" else real_find_spec(name, *a)))


def test_cuda_backend_that_cannot_load_falls_back(monkeypatch):
    # Installed but blocked (e.g. Windows Application Control on a DLL):
    # start on the next device instead of failing to load CUDA.
    _nvidia_with_faster_whisper(monkeypatch)

    def blocked():
        raise RuntimeError("faster-whisper cannot be loaded: DLL load failed")
    monkeypatch.setattr(de, "import_faster_whisper", blocked)
    devices = de.detect_devices()
    assert "CUDA" not in devices
    assert de.select_device(_cfg(), devices) == "NPU"


def test_faster_whisper_loads_without_pyav(monkeypatch):
    # PyAV only decodes audio files; a blocked PyAV must not disable CUDA.
    import sys
    import types
    monkeypatch.setitem(sys.modules, "av", None)  # `import av` raises ImportError
    model = object()
    monkeypatch.setitem(sys.modules, "faster_whisper",
                        types.SimpleNamespace(WhisperModel=model))
    assert de.import_faster_whisper() is model
    assert isinstance(sys.modules["av"], types.ModuleType)


def test_faster_whisper_load_error_names_the_cause(monkeypatch):
    import sys
    import types
    monkeypatch.setitem(sys.modules, "av", types.ModuleType("av"))
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    with pytest.raises(RuntimeError, match="cannot be loaded"):
        de.import_faster_whisper()
