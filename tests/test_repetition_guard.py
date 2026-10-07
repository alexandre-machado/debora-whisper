"""Whisper repetition loops ("Oi Oi Oi ..." filling all 448 tokens)."""
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from debora_whisper import dictation_engine as de

LOOP = "Oi, tudo bem? " + "Oi " * 200
SPEECH = ("Hoje eu preciso revisar o pull request do empacotamento, conferir se o "
          "pipeline passou no Windows e no Ubuntu e depois responder o time.")


@pytest.mark.parametrize("text, looping", [
    (LOOP, True),
    ("a b c " * 40, True),
    (SPEECH, False),
    ("Oi Oi Oi", False),  # too short to judge
    ("", False),
    # Dictated digits compress like a loop but are speech.
    ("o número da conta é " + "zero " * 16 + "um dois três", False),
    ("meu telefone é nove nove nove nove nove nove meia meia meia meia "
     "meia meia um um um um um um dois dois dois dois", False),
    ("0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1", False),
    ("Oi " * 40 + "zero zero zero", True),
])
def test_is_repetition_loop(text, looping):
    assert de.is_repetition_loop(text) is looping


@pytest.mark.parametrize("text, expected", [
    ("Oi Oi Oi Oi Oi", "Oi"),
    ("Oi, tudo bem? Oi Oi Oi Oi", "Oi, tudo bem? Oi"),
    ("eu acho que eu acho que eu acho que sim", "eu acho que sim"),
    ("não, não, tudo certo", "não, não, tudo certo"),  # two repeats stay
    (SPEECH, SPEECH),
    ("", ""),
    ("conta zero zero zero zero um", "conta zero zero zero zero um"),
    ("Oi Oi Oi Oi conta 0 0 0 0 1", "Oi conta 0 0 0 0 1"),
])
def test_collapse_repetitions(text, expected):
    assert de.collapse_repetitions(text) == expected


class _Config(SimpleNamespace):
    pass


def _whisper_npu(outputs):
    """A WhisperNPU whose pipeline returns `outputs` in turn, recording the
    repetition_penalty each generate() saw."""
    npu = de.WhisperNPU.__new__(de.WhisperNPU)
    npu.device = "NPU"
    npu.pipeline = MagicMock()
    npu.pipeline.get_generation_config.return_value = _Config(repetition_penalty=1.0)
    seen = []

    def generate(_audio, config):
        seen.append(config.repetition_penalty)
        return outputs[len(seen) - 1]
    npu.pipeline.generate.side_effect = generate
    return npu, seen


AUDIO = np.zeros(16000, dtype=np.float32)


def test_npu_normal_output_decodes_once():
    npu, seen = _whisper_npu([SPEECH])
    assert npu.transcribe(AUDIO, language="pt") == SPEECH
    assert seen == [1.0]


def test_npu_loop_is_decoded_again_with_repeats_penalized():
    npu, seen = _whisper_npu([LOOP, SPEECH])
    assert npu.transcribe(AUDIO, language="pt") == SPEECH
    assert seen == [1.0, de.RETRY_REPETITION_PENALTY]


def test_npu_loop_that_survives_the_retry_is_collapsed():
    npu, seen = _whisper_npu([LOOP, LOOP])
    assert npu.transcribe(AUDIO, language="pt") == "Oi, tudo bem? Oi"
    assert len(seen) == 2


def _cuda(outputs):
    """A FasterWhisperCUDA whose pipeline returns `outputs` in turn,
    recording the keyword arguments of each call."""
    cuda = de.FasterWhisperCUDA.__new__(de.FasterWhisperCUDA)
    calls = []

    def transcribe(_audio, language=None, condition_on_previous_text=True,
                   without_timestamps=False, temperature=0.0, hotwords=None,
                   repetition_penalty=1.0):
        calls.append({"hotwords": hotwords, "repetition_penalty": repetition_penalty})
        return iter([SimpleNamespace(text=outputs[len(calls) - 1])]), None
    cuda.pipeline = SimpleNamespace(transcribe=transcribe)
    return cuda, calls


def test_cuda_loop_is_decoded_again_with_repeats_penalized():
    cuda, calls = _cuda([LOOP, SPEECH])
    with patch.object(de, "ensure_devices_usable"):
        assert cuda.transcribe(AUDIO, language="pt") == SPEECH
    assert [c["repetition_penalty"] for c in calls] == [1.0, de.RETRY_REPETITION_PENALTY]


def test_cuda_retry_keeps_the_hotwords():
    cuda, calls = _cuda([LOOP, SPEECH])
    with patch.object(de, "ensure_devices_usable"):
        cuda.transcribe(AUDIO, language="pt", hotwords="Débora")
    assert [c["hotwords"] for c in calls] == ["Débora", "Débora"]


def test_save_wav_round_trips(tmp_path):
    audio = np.array([0.0, 0.5, -0.5, 1.5], dtype=np.float32)
    path = tmp_path / "sub" / "x.wav"
    de.save_wav(path, audio, 16000)
    with wave.open(str(path)) as f:
        assert (f.getnchannels(), f.getsampwidth(), f.getframerate()) == (1, 2, 16000)
        pcm = np.frombuffer(f.readframes(4), dtype="<i2")
    assert pcm.tolist() == [0, 16383, -16383, 32767]


def _app(**config):
    app = de.DictationApp({**de.DEFAULT_CONFIG, "beep_on_start": False, **config})
    app.whisper = MagicMock()
    app.whisper.transcribe.return_value = "hello world"
    app._model_ready.set()
    app.recorder = MagicMock()
    app.recorder.stop.return_value = np.full(16000, 0.25, dtype=np.float32)
    app.is_recording = True
    return app


@pytest.mark.parametrize("enabled", [True, False])
def test_last_recording_is_saved_only_when_enabled(enabled):
    app = _app(save_last_recording=enabled)
    with patch.object(de, "type_text"), patch.object(de, "ensure_devices_usable"):
        app._finish_recording()
    assert de.LAST_RECORDING.exists() is enabled


def test_turning_the_option_off_deletes_the_last_recording():
    de.LAST_RECORDING.write_bytes(b"RIFF")
    _app(save_last_recording=True)
    assert de.LAST_RECORDING.exists()
    _app(save_last_recording=False)
    assert not de.LAST_RECORDING.exists()
