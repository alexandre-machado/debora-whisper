"""Tests for config validation."""
import pytest

from debora_whisper.dictation_engine import validate_config, DEFAULT_CONFIG


class TestConfigValidation:
    def test_valid_config_passes(self):
        config = DEFAULT_CONFIG.copy()
        validate_config(config)  # should not raise

    def test_invalid_device_raises(self):
        config = {**DEFAULT_CONFIG, "device": "TPU"}
        with pytest.raises(ValueError, match="device"):
            validate_config(config)

    def test_invalid_model_size_raises(self):
        config = {**DEFAULT_CONFIG, "model_size": "huge"}
        with pytest.raises(ValueError, match="model_size"):
            validate_config(config)

    def test_invalid_sample_rate_raises(self):
        config = {**DEFAULT_CONFIG, "sample_rate": "banana"}
        with pytest.raises(ValueError, match="sample_rate"):
            validate_config(config)

    def test_invalid_max_record_seconds_raises(self):
        config = {**DEFAULT_CONFIG, "max_record_seconds": -5}
        with pytest.raises(ValueError, match="max_record_seconds"):
            validate_config(config)

    @pytest.mark.parametrize("key", ["vad_end_silence_seconds", "vad_incomplete_silence_seconds"])
    @pytest.mark.parametrize("value", [None, True, "3", 0, -1, float("nan"), float("inf")])
    def test_invalid_vad_silence_raises(self, key, value):
        with pytest.raises(ValueError, match=key):
            validate_config({**DEFAULT_CONFIG, key: value})

    def test_existing_config_without_vad_keys_uses_defaults(self):
        config = {key: value for key, value in DEFAULT_CONFIG.items() if not key.startswith("vad_")}
        validate_config(config)
