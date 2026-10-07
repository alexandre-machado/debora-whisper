"""config["device_priority"] picks the startup device and the fallback."""
import pytest

from npu_whisper import dictation_engine as de

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
    devices = de.detect_devices()
    assert ("CUDA" in devices) is has_backend
    assert de.select_device(_cfg(), devices) == ("CUDA" if has_backend else "NPU")
