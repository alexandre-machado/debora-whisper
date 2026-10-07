"""config["device_priority"] picks the startup device and the fallback."""
import pytest

import dictation_engine as de

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
