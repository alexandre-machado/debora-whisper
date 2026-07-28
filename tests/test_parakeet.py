"""Tests for Parakeet TDT integration."""

import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path
import numpy as np
import pytest

from dictation_engine import (
    MODEL_REGISTRY, DictationApp, DEFAULT_CONFIG,
    create_model, ParakeetNPU, WhisperNPU,
    LANGUAGES, get_models_for_language, is_model_downloaded, MODEL_DIR,
)


class TestParakeetRegistry:
    """Verify parakeet is correctly registered."""

    def test_parakeet_in_registry(self):
        assert "parakeet" in MODEL_REGISTRY

    def test_parakeet_has_required_fields(self):
        info = MODEL_REGISTRY["parakeet"]
        assert info["backend"] == "parakeet"
        assert info["preferred_device"] == "NPU"
        assert "ov_repo" in info
        assert "local_dir" in info
        assert "description" in info

    def test_all_models_have_backend_field(self):
        for name, info in MODEL_REGISTRY.items():
            assert "backend" in info, f"{name} missing backend field"
            assert info["backend"] in ("whisper", "parakeet"), f"{name} has invalid backend"

    def test_all_models_have_local_dir_field(self):
        for name, info in MODEL_REGISTRY.items():
            assert "local_dir" in info, f"{name} missing local_dir field"


class TestCreateModel:
    """Verify factory function dispatches correctly."""

    @patch("dictation_engine.WhisperNPU")
    def test_creates_whisper_for_whisper_backend(self, mock_cls):
        mock_cls.return_value = MagicMock()
        result = create_model(Path("/fake"), device="NPU", backend="whisper")
        mock_cls.assert_called_once_with(Path("/fake"), device="NPU")

    @patch("dictation_engine.ParakeetNPU")
    def test_creates_parakeet_for_parakeet_backend(self, mock_cls):
        mock_cls.return_value = MagicMock()
        result = create_model(Path("/fake"), device="NPU", backend="parakeet")
        mock_cls.assert_called_once_with(Path("/fake"), device="NPU")


class TestEnsureModelDispatch:
    """Verify ensure_model uses factory pattern."""

    @patch("dictation_engine.create_model")
    @patch("dictation_engine.setup_model")
    def test_ensure_model_parakeet(self, mock_setup, mock_create):
        mock_setup.return_value = Path("/fake/parakeet")
        mock_create.return_value = MagicMock()

        config = {**DEFAULT_CONFIG, "model_size": "parakeet", "device": "NPU"}
        app = DictationApp(config)
        app.ensure_model()

        mock_create.assert_called_once_with(
            Path("/fake/parakeet"), device="NPU", backend="parakeet",
        )

    @patch("dictation_engine.create_model")
    @patch("dictation_engine.setup_model")
    def test_ensure_model_whisper(self, mock_setup, mock_create):
        mock_setup.return_value = Path("/fake/whisper")
        mock_create.return_value = MagicMock()

        config = {**DEFAULT_CONFIG, "model_size": "base", "device": "NPU"}
        app = DictationApp(config)
        app.ensure_model()

        mock_create.assert_called_once_with(
            Path("/fake/whisper"), device="NPU", backend="whisper",
        )


class TestParakeetConfig:
    """Verify parakeet works with config validation."""

    def test_parakeet_config_validates(self):
        from dictation_engine import validate_config
        config = {**DEFAULT_CONFIG, "model_size": "parakeet", "device": "NPU"}
        # Should not raise
        validate_config(config)

    def test_invalid_model_still_rejected(self):
        from dictation_engine import validate_config
        config = {**DEFAULT_CONFIG, "model_size": "nonexistent"}
        with pytest.raises(ValueError, match="model_size"):
            validate_config(config)


class TestLanguageFiltering:
    """Verify language-based model filtering."""

    def test_all_models_have_languages_field(self):
        for name, info in MODEL_REGISTRY.items():
            assert "languages" in info, f"{name} missing languages field"

    def test_english_returns_all_models(self):
        models = get_models_for_language("en")
        assert len(models) == len(MODEL_REGISTRY)
        assert "parakeet" in models

    def test_non_english_excludes_parakeet(self):
        for lang in ("ru", "es", "fr", "de", "ja", "zh"):
            models = get_models_for_language(lang)
            assert "parakeet" not in models, f"parakeet should not be in {lang} models"
            assert "base" in models
            assert "small" in models

    def test_whisper_models_support_all_languages(self):
        for name, info in MODEL_REGISTRY.items():
            if info["backend"] == "whisper":
                assert info["languages"] == "all", f"{name} should support all languages"

    def test_parakeet_english_only(self):
        info = MODEL_REGISTRY["parakeet"]
        assert info["languages"] == ["en"]

    def test_languages_dict_has_entries(self):
        assert len(LANGUAGES) >= 10
        assert "en" in LANGUAGES
        assert LANGUAGES["en"] == "English"


class TestModelDownloadStatus:
    """Verify download status detection."""

    def test_nonexistent_model_not_downloaded(self, tmp_path, monkeypatch):
        monkeypatch.setattr("dictation_engine.MODEL_DIR", tmp_path)
        assert is_model_downloaded("base") is False

    def test_empty_dir_not_downloaded(self, tmp_path, monkeypatch):
        monkeypatch.setattr("dictation_engine.MODEL_DIR", tmp_path)
        (tmp_path / "whisper-base-openvino").mkdir()
        assert is_model_downloaded("base") is False

    def test_dir_with_xml_is_downloaded(self, tmp_path, monkeypatch):
        monkeypatch.setattr("dictation_engine.MODEL_DIR", tmp_path)
        model_dir = tmp_path / "whisper-base-openvino"
        model_dir.mkdir()
        (model_dir / "model.xml").write_text("")
        assert is_model_downloaded("base") is True

    def test_dir_with_onnx_is_downloaded(self, tmp_path, monkeypatch):
        monkeypatch.setattr("dictation_engine.MODEL_DIR", tmp_path)
        model_dir = tmp_path / "parakeet-tdt-openvino"
        model_dir.mkdir()
        (model_dir / "encoder-model.onnx").write_text("")
        assert is_model_downloaded("parakeet") is True


class TestShapeBucketing:
    """Regression tests for issue #3 shape-bucket selection (no NPU needed —
    select_bucket() and the padding logic are pure/deterministic)."""

    def test_bucket_set_is_ascending_and_nonempty(self):
        buckets = ParakeetNPU.MEL_BUCKETS
        assert len(buckets) >= 1
        assert list(buckets) == sorted(buckets)
        assert len(set(buckets)) == len(buckets)

    def test_smallest_audio_gets_smallest_bucket(self):
        bucket, truncated = ParakeetNPU.select_bucket(1)
        assert bucket == ParakeetNPU.MEL_BUCKETS[0]
        assert truncated is False

    def test_exact_boundary_picks_that_bucket_not_the_next(self):
        for bucket in ParakeetNPU.MEL_BUCKETS:
            chosen, truncated = ParakeetNPU.select_bucket(bucket)
            assert chosen == bucket, f"exact-length input {bucket} should pick bucket {bucket}, got {chosen}"
            assert truncated is False

    def test_one_frame_over_boundary_picks_next_bucket(self):
        buckets = ParakeetNPU.MEL_BUCKETS
        for smaller, larger in zip(buckets, buckets[1:]):
            chosen, truncated = ParakeetNPU.select_bucket(smaller + 1)
            assert chosen == larger, f"{smaller + 1} frames should round up to {larger}, got {chosen}"
            assert truncated is False

    def test_over_length_audio_falls_back_to_largest_bucket_and_flags_truncation(self):
        largest = ParakeetNPU.MEL_BUCKETS[-1]
        chosen, truncated = ParakeetNPU.select_bucket(largest + 1)
        assert chosen == largest
        assert truncated is True

        chosen, truncated = ParakeetNPU.select_bucket(largest * 10)
        assert chosen == largest
        assert truncated is True

    def test_zero_frames_picks_smallest_bucket(self):
        chosen, truncated = ParakeetNPU.select_bucket(0)
        assert chosen == ParakeetNPU.MEL_BUCKETS[0]
        assert truncated is False


class TestMelPaddingToBucket:
    """Verify mel features end up padded to exactly the selected bucket's
    frame count, via ParakeetNPU.pad_to_bucket() — the actual method
    transcribe() and the benchmark harness call, not a reimplementation.
    """

    def test_short_audio_padded_to_exact_bucket_size(self):
        actual_frames = 50
        mel = np.ones((1, 128, actual_frames), dtype=np.float32)
        bucket, truncated = ParakeetNPU.select_bucket(actual_frames)
        assert truncated is False
        padded = ParakeetNPU.pad_to_bucket(mel, bucket)
        assert padded.shape == (1, 128, bucket)
        # original content preserved
        assert np.array_equal(padded[:, :, :actual_frames], mel)
        # padding is zeros
        assert np.all(padded[:, :, actual_frames:] == 0)

    def test_exact_bucket_length_unchanged(self):
        bucket = ParakeetNPU.MEL_BUCKETS[1]
        mel = np.ones((1, 128, bucket), dtype=np.float32)
        chosen, truncated = ParakeetNPU.select_bucket(bucket)
        assert chosen == bucket
        assert truncated is False
        padded = ParakeetNPU.pad_to_bucket(mel, chosen)
        assert padded.shape == (1, 128, bucket)
        assert np.array_equal(padded, mel)

    def test_over_length_audio_truncated_to_largest_bucket_not_dropped_silently(self):
        largest = ParakeetNPU.MEL_BUCKETS[-1]
        actual_frames = largest + 500
        mel = np.arange(actual_frames, dtype=np.float32).reshape(1, 1, -1)
        mel = np.broadcast_to(mel, (1, 128, actual_frames)).copy()
        bucket, truncated = ParakeetNPU.select_bucket(actual_frames)
        assert bucket == largest
        assert truncated is True  # caller (transcribe()) must log this, not swallow it
        result = ParakeetNPU.pad_to_bucket(mel, bucket)
        assert result.shape == (1, 128, largest)
        # the leading `largest` frames of real speech are kept, not zeroed/dropped
        assert np.array_equal(result, mel[:, :, :largest])
