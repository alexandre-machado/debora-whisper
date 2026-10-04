"""
NPU Dictation Engine - Local voice-to-text using OpenVINO on Intel NPU
Supports Whisper (via openvino_genai) and Parakeet TDT (via OpenVINO + onnxruntime).

Usage: python dictation_engine.py [--setup] [--device NPU|GPU|CPU] [--model base|small|medium|parakeet]
"""

import sys
import os
import time
import json
import re
import argparse
import threading
from enum import Enum
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG_DIR = Path.home() / ".npu-dictation"
CONFIG_FILE = CONFIG_DIR / "config.json"
MODEL_DIR = CONFIG_DIR / "models"
LOG_FILE = CONFIG_DIR / "dictation.log"
CACHE_DIR = CONFIG_DIR / "ov-cache"

DEFAULT_CONFIG = {
    "device": "NPU",           # NPU, GPU, CPU
    "model_size": "base",      # tiny, base, small, medium (large not supported on NPU)
    "language": "en",          # Language code or "auto"
    "hotkey": "ctrl+space",    # Global hotkey to toggle recording
    "auto_enter": False,       # Press Enter after pasting (useful for Claude Code)
    "beep_on_start": True,     # Audio feedback when recording starts/stops
    "max_record_seconds": 60,  # Max recording length
    "sample_rate": 16000,      # Whisper expects 16kHz
    "show_balloon": True,      # Show text balloon under notch after transcription
}

# Supported languages (Whisper's top languages + display names)
LANGUAGES = {
    "en": "English",
    "ru": "Russian",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "ar": "Arabic",
    "uk": "Ukrainian",
}

# The 25 languages nvidia/parakeet-tdt-0.6b-v3 was trained on (automatic
# language ID, no language token needed at inference). Source:
# https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 model card.
PARAKEET_UPSTREAM_LANGUAGES = {
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu",
    "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru",
    "uk",
}

# Model registry: pre-exported models from HuggingFace
MODEL_REGISTRY = {
    "tiny": {
        "repo": "openai/whisper-tiny",
        "ov_repo": "OpenVINO/whisper-tiny-int8-ov",
        "description": "39M params. Lowest resource usage for quick commands.",
        "preferred_device": "NPU",
        "backend": "whisper",
        "local_dir": "whisper-tiny-openvino",
        "languages": "all",
    },
    "base": {
        "repo": "openai/whisper-base",
        "ov_repo": "OpenVINO/whisper-base-int8-ov",
        "description": "74M params, 5.0% WER. Fast, good for short commands.",
        "preferred_device": "NPU",
        "backend": "whisper",
        "local_dir": "whisper-base-openvino",
        "languages": "all",
    },
    "small": {
        "repo": "openai/whisper-small",
        "ov_repo": "OpenVINO/whisper-small-int8-ov",
        "description": "244M params, 3.4% WER. Balanced speed/accuracy.",
        "preferred_device": "NPU",
        "backend": "whisper",
        "local_dir": "whisper-small-openvino",
        "languages": "all",
    },
    "medium": {
        "repo": "openai/whisper-medium",
        "ov_repo": "OpenVINO/whisper-medium-int8-ov",
        "description": "769M params, 2.9% WER. High accuracy, slower.",
        "preferred_device": "GPU",
        "backend": "whisper",
        "local_dir": "whisper-medium-openvino",
        "languages": "all",
    },
    "turbo": {
        "repo": "openai/whisper-large-v3-turbo",
        "ov_repo": "FluidInference/whisper-large-v3-turbo-int4-ov-npu",
        "description": "809M params, 2.3% WER. Best multilingual quality.",
        "preferred_device": "GPU",
        "backend": "whisper",
        "local_dir": "whisper-turbo-openvino",
        "languages": "all",
    },
    "parakeet": {
        "repo": "nvidia/parakeet-tdt-0.6b-v3",
        "ov_repo": "goodsmileduck/parakeet-tdt-0.6b-v3-onnx",
        "description": (
            "600M params, 3.7% WER (LibriSpeech test-clean, publisher-reported). "
            "Best accuracy, hybrid NPU+CPU. Multilingual with automatic language ID."
        ),
        "preferred_device": "NPU",
        "backend": "parakeet",
        "local_dir": "parakeet-tdt-openvino",
        # Intersection of the upstream checkpoint's 25 supported languages with
        # the languages this app exposes in the UI (LANGUAGES below). Do not
        # hand-edit this list; it is derived so it can never drift ahead of
        # what the model picker can actually offer.
        "languages": sorted(PARAKEET_UPSTREAM_LANGUAGES & LANGUAGES.keys()),
    },
}


def get_models_for_language(lang: str) -> dict:
    """Return subset of MODEL_REGISTRY compatible with the given language."""
    return {k: v for k, v in MODEL_REGISTRY.items()
            if v["languages"] == "all" or lang in v["languages"]}


def is_model_downloaded(model_key: str) -> bool:
    """Check if model files exist locally."""
    info = MODEL_REGISTRY[model_key]
    path = MODEL_DIR / info["local_dir"]
    return path.exists() and (
        any(path.glob("*.xml")) or any(path.glob("*.onnx")))


# ---------------------------------------------------------------------------
# Application state machine
# ---------------------------------------------------------------------------
class AppState(Enum):
    LOADING = "loading"        # Model loading in progress
    READY = "ready"            # Idle, waiting for hotkey
    RECORDING = "recording"    # Microphone active
    PROCESSING = "processing"  # Transcribing audio
    ERROR = "error"            # Device lost or load failed


def log(msg: str):
    """Simple logging to file and console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Config management
# ---------------------------------------------------------------------------
def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            saved = json.load(f)
        config = {**DEFAULT_CONFIG, **saved}
    else:
        config = DEFAULT_CONFIG.copy()
    return config


def save_config(config: dict):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    log(f"Config saved to {CONFIG_FILE}")


def validate_config(config: dict):
    """Validate config values. Raises ValueError on invalid values."""
    valid_devices = {"NPU", "GPU", "CPU"}
    if config.get("device") not in valid_devices:
        raise ValueError(f"device must be one of {valid_devices}, got '{config.get('device')}'")

    valid_models = set(MODEL_REGISTRY.keys())
    if config.get("model_size") not in valid_models:
        raise ValueError(f"model_size must be one of {valid_models}, got '{config.get('model_size')}'")

    sr = config.get("sample_rate")
    if not isinstance(sr, (int, float)) or sr <= 0:
        raise ValueError(f"sample_rate must be a positive number, got '{sr}'")

    max_rec = config.get("max_record_seconds")
    if max_rec is not None and (not isinstance(max_rec, (int, float)) or max_rec <= 0):
        raise ValueError(f"max_record_seconds must be a positive number or null, got '{max_rec}'")


# ---------------------------------------------------------------------------
# Model setup (export Whisper to OpenVINO IR format)
# ---------------------------------------------------------------------------
def setup_model(config: dict, progress_callback=None):
    """Download pre-exported model from HuggingFace.

    Supports both Whisper (.xml OpenVINO IR) and Parakeet (.onnx) models.

    Args:
        config: Application config dict.
        progress_callback: Optional callable(downloaded_bytes, total_bytes) for
            download progress reporting.
    """
    model_size = config["model_size"]
    model_info = MODEL_REGISTRY[model_size]
    model_path = MODEL_DIR / model_info["local_dir"]

    # Check if model already exists (Whisper uses .xml, Parakeet uses .onnx)
    has_xml = model_path.exists() and any(model_path.glob("*.xml"))
    has_onnx = model_path.exists() and any(model_path.glob("*.onnx"))
    if has_xml or has_onnx:
        log(f"Model already available at {model_path}")
        return model_path

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    ov_repo = model_info["ov_repo"]
    log(f"Downloading pre-exported model from {ov_repo}...")
    try:
        from huggingface_hub import snapshot_download

        # Build tqdm_class wrapper for progress reporting.
        # snapshot_download creates one _ProgressBar per file, so we
        # accumulate bytes across all instances for overall progress.
        tqdm_kwargs = {}
        if progress_callback:
            _cumulative = [0, 0]  # [downloaded_bytes, total_bytes]

            class _ProgressBar:
                """Minimal tqdm-compatible wrapper that forwards to progress_callback."""
                _lock = None
                def __init__(self, *args, **kwargs):
                    self.total = kwargs.get("total", 0)
                    self.n = 0
                    _cumulative[1] += self.total
                @classmethod
                def get_lock(cls):
                    import threading
                    if cls._lock is None:
                        cls._lock = threading.Lock()
                    return cls._lock
                @classmethod
                def set_lock(cls, lock):
                    cls._lock = lock
                def update(self, n=1):
                    self.n += n
                    _cumulative[0] += n
                    if _cumulative[1] > 0:
                        progress_callback(_cumulative[0], _cumulative[1])
                def close(self):
                    pass
                def set_description(self, *a, **kw):
                    pass
                def set_postfix(self, *a, **kw):
                    pass
                def refresh(self, *a, **kw):
                    pass
                def __enter__(self):
                    return self
                def __exit__(self, *a):
                    pass

            tqdm_kwargs["tqdm_class"] = _ProgressBar

        snapshot_download(ov_repo, local_dir=str(model_path), **tqdm_kwargs)

        # Verify download
        has_xml = any(model_path.glob("*.xml"))
        has_onnx = any(model_path.glob("*.onnx"))
        if has_xml or has_onnx:
            log(f"Model downloaded to {model_path}")
            return model_path
    except Exception as e:
        log(f"Download failed: {e}")
        sys.exit(1)

    log(f"No model files found at {model_path}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Whisper pipeline using OpenVINO GenAI
# ---------------------------------------------------------------------------
class WhisperNPU:
    """Whisper speech-to-text using OpenVINO on NPU/GPU/CPU."""

    def __init__(self, model_path: Path, device: str = "NPU"):
        self.model_path = model_path
        self.device = device
        self.pipeline = None
        self._load_pipeline()

    def _load_pipeline(self):
        """Load the OpenVINO Whisper pipeline."""
        # Warn about large models on NPU — they may trigger driver instability
        model_name = self.model_path.name.lower()
        if self.device == "NPU" and ("turbo" in model_name or "large" in model_name or "medium" in model_name):
            log(f"NOTE: Large models on NPU may cause driver instability (DEVICE_LOST).")
            log(f"  If inference fails, try rebooting to reset the NPU, or use --device GPU")

        log(f"Loading Whisper pipeline on {self.device}...")
        start = time.time()

        try:
            import openvino_genai as ov_genai
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self.pipeline = ov_genai.WhisperPipeline(
                str(self.model_path), self.device,
                CACHE_DIR=str(CACHE_DIR)
            )
            log(f"Loaded via openvino_genai in {time.time() - start:.1f}s")
        except Exception as e:
            log(f"Failed to load model: {e}")
            log("Falling back to CPU...")
            if self.device != "CPU":
                self.device = "CPU"
                self._load_pipeline()
            else:
                raise

    def transcribe(self, audio_data, sample_rate: int = 16000, language: str = "en") -> str:
        """Transcribe audio numpy array to text."""
        import numpy as np
        start = time.time()

        # Ensure float32 normalized to [-1, 1]
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        config = self.pipeline.get_generation_config()
        config.max_new_tokens = 448
        if language and language != "auto":
            config.language = f"<|{language}|>"
        config.task = "transcribe"
        config.return_timestamps = False

        try:
            result = self.pipeline.generate(audio_data, config)
        except RuntimeError as e:
            if "DEVICE_LOST" in str(e) or "device hung" in str(e):
                log(f"NPU DEVICE_LOST — the NPU driver crashed or is in a bad state.")
                log(f"  A reboot will reset the NPU. Or use --device GPU to bypass it.")
                raise RuntimeError(
                    f"NPU device lost. Reboot to reset the NPU, "
                    f"or use --device GPU."
                ) from e
            raise
        text = str(result).strip()

        elapsed = time.time() - start
        audio_duration = len(audio_data) / sample_rate
        rtf = elapsed / audio_duration if audio_duration > 0 else 0
        log(f"Transcribed {audio_duration:.1f}s audio in {elapsed:.1f}s (RTF: {rtf:.2f}) on {self.device}")

        return text


# ---------------------------------------------------------------------------
# Parakeet TDT pipeline using OpenVINO (encoder on NPU, decoder on GPU)
# ---------------------------------------------------------------------------
class ParakeetNPU:
    """Parakeet TDT 0.6B speech-to-text using OpenVINO on NPU+GPU.

    Architecture:
        nemo128.onnx (onnxruntime CPU) → mel spectrogram
        encoder-model.onnx (OpenVINO NPU) → encoder features
        decoder_joint-model.onnx (OpenVINO GPU) → TDT greedy decode
    """

    # TDT constants. These are the expected values for the multilingual
    # nvidia/parakeet-tdt-0.6b-v3 checkpoint's 8192-entry vocab (indices
    # 0-8191) plus one blank token at index 8192. They are treated as
    # defaults only: _load_vocab() derives the real values from the loaded
    # vocab.txt and overrides these instance attributes if the checkpoint's
    # vocab size ever differs (e.g. a future re-export).
    BLANK_IDX = 8192
    VOCAB_SIZE = 8193  # 0-8192 are vocab tokens, 8193+ are duration tokens
    MAX_TOKENS_PER_STEP = 10

    # NeMo's mel preprocessor (nemo128.onnx) runs at a 10ms hop, i.e. 100
    # mel frames per second of audio.
    MEL_FRAME_RATE = 100

    # --------------------------------------------------------------------
    # MEL_BUCKETS — shape-bucketing for the NPU encoder (issue #3).
    #
    # NPU only supports static shapes (OpenVINO NPU docs: "Currently, only
    # models with static shapes are supported on NPU"), so the encoder must
    # be compiled ahead of time for a fixed audio_signal length. Before this
    # change, every utterance ran through one shape sized for the worst
    # case (1600 frames / ~16s), so a typical few-second dictation paid the
    # full 16s graph cost.
    #
    # ⚠️ CALIBRATED BY REASONING, NOT MEASUREMENT. No before/after NPU
    # numbers exist yet for this repo (see benchmarks/README.md and issue
    # #3). These sizes are a plausible starting spread for a
    # press-hotkey-to-talk dictation workflow (short commands through
    # multi-sentence dictation, default max_record_seconds=60 in
    # DEFAULT_CONFIG), not a tuned result. Expect to revisit this tuple
    # once a user runs benchmarks/bench_parakeet.py on real NPU hardware.
    # Edit freely — it's a flat tuple of frame counts, ascending, in frames
    # (frames = seconds * MEL_FRAME_RATE). The last entry is also the
    # fallback/max: longer audio is truncated to it (see transcribe()).
    MEL_BUCKETS = (
        200,   # ~2s  — short commands ("open terminal", "next line")
        500,   # ~5s  — typical single-sentence dictation
        900,   # ~9s  — longer dictated sentence / short paragraph
        1600,  # ~16s — original static shape; kept as the ceiling/fallback
    )

    ENC_DIM = 1024
    LSTM_DIM = 640
    DECODE_SPACE = re.compile(r"\A\s|\s\B|(\s)\b")

    def __init__(self, model_path: Path, device: str = "NPU", latency_override: bool = None):
        self.model_path = model_path
        self.device = device
        # Measurable, opt-in lever from issue #3: sets
        # ov::intel_npu::compilation_mode_params with
        # performance-hint-override="latency" (NPU's default for that
        # sub-property is "efficiency"). Effect is UNMEASURED — do not
        # infer anything from this flag existing. Off by default; toggle
        # with the PARAKEET_LATENCY_OVERRIDE=1 env var (kept out of
        # create_model()/DEFAULT_CONFIG so the existing factory call
        # signature and its tests stay untouched — this is a measurement
        # knob, not a shipped feature).
        if latency_override is None:
            latency_override = os.environ.get("PARAKEET_LATENCY_OVERRIDE", "") in ("1", "true", "yes")
        self.latency_override = latency_override
        self.vocab: dict[int, str] = {}
        self.preproc = None  # onnxruntime session
        self.enc_compiled: dict[int, "object"] = {}  # bucket frame count -> compiled encoder
        self.dec_compiled = None  # OpenVINO compiled decoder
        self.enc_time_dim: dict[int, int] = {}  # bucket frame count -> encoder output time dim
        self._load_pipeline()

    @classmethod
    def select_bucket(cls, actual_frames: int) -> tuple[int, bool]:
        """Pick the smallest bucket that fits `actual_frames`.

        Returns (bucket_frames, was_truncated). was_truncated is True when
        actual_frames exceeds every bucket, in which case the caller must
        truncate to the largest bucket (see transcribe()) — this never
        raises and never silently drops the fact that truncation happened.
        """
        for bucket in cls.MEL_BUCKETS:
            if actual_frames <= bucket:
                return bucket, False
        return cls.MEL_BUCKETS[-1], True

    @staticmethod
    def pad_to_bucket(mel, bucket: int):
        """Zero-pad or truncate `mel` (shape [1, mel_bins, frames]) to
        exactly `bucket` frames.

        Padding is appended after the real content (zeros at the end, not
        interleaved). Truncation keeps the leading `bucket` frames and drops
        the trailing ones — callers that truncate must surface that fact to
        the user themselves (see transcribe()'s truncation warning); this
        method just does the array op.

        Single source of truth for this logic — transcribe() and the
        benchmark harness (benchmarks/bench_parakeet.py) both call this
        instead of reimplementing it, so tests that exercise this method
        exercise the real production code path.
        """
        import numpy as np
        actual_frames = mel.shape[2]
        if actual_frames < bucket:
            padded = np.zeros((1, mel.shape[1], bucket), dtype=mel.dtype)
            padded[:, :, :actual_frames] = mel
            return padded
        elif actual_frames > bucket:
            return mel[:, :, :bucket]
        return mel

    def _load_vocab(self):
        """Load vocab.txt from model directory."""
        vocab_path = self.model_path / "vocab.txt"
        if not vocab_path.exists():
            raise FileNotFoundError(f"vocab.txt not found at {vocab_path}")
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip("\n").split(" ")
                if len(parts) == 2:
                    token, idx = parts[0], int(parts[1])
                    self.vocab[idx] = token.replace("\u2581", " ")
        log(f"Parakeet vocab loaded: {len(self.vocab)} tokens")

        # An empty or unparseable vocab.txt must fail loudly rather than
        # silently falling back to the hardcoded class defaults: those
        # defaults would then be paired with an empty self.vocab dict,
        # letting the pipeline load "successfully" while every decoded
        # token renders as "?" (via self.vocab.get(t, "?")) instead of
        # surfacing the broken download/export immediately.
        if not self.vocab:
            raise ValueError(
                f"Parakeet vocab.txt at {vocab_path} produced zero valid entries "
                f"(expected '<token> <index>' pairs per line, e.g. '▁the 42'). "
                f"The file is empty, truncated, or in an unexpected format. "
                f"Re-run setup: python dictation_engine.py --model parakeet --setup"
            )

        # Derive BLANK_IDX / VOCAB_SIZE from the loaded vocab instead of
        # trusting the hardcoded class defaults blindly. The joint network
        # emits logits laid out as [vocab tokens..., blank, duration bins...],
        # so blank sits immediately after the highest real vocab index.
        max_idx = max(self.vocab)
        derived_blank = max_idx + 1
        derived_vocab_size = max_idx + 2
        if derived_blank != self.BLANK_IDX or derived_vocab_size != self.VOCAB_SIZE:
            log(
                f"WARNING: TDT constants derived from vocab.txt (BLANK_IDX="
                f"{derived_blank}, VOCAB_SIZE={derived_vocab_size}) differ from "
                f"hardcoded defaults (BLANK_IDX={self.BLANK_IDX}, "
                f"VOCAB_SIZE={self.VOCAB_SIZE}); using derived values."
            )
        self.BLANK_IDX = derived_blank
        self.VOCAB_SIZE = derived_vocab_size

    def _validate_vocab_against_decoder(self):
        """Cross-check the vocab-derived VOCAB_SIZE against the decoder's
        actual compiled output width.

        _load_vocab() derives BLANK_IDX/VOCAB_SIZE purely from vocab.txt, with
        no guarantee it agrees with decoder_joint-model.onnx's real logit
        layout ([vocab tokens..., blank, duration bins...]). If VOCAB_SIZE is
        too large, `duration_logits = output[self.VOCAB_SIZE:]` in
        _tdt_greedy_decode becomes an empty array and `.argmax()` raises an
        unhandled ValueError mid-transcription. If it's too small (but still
        within bounds), decoding would silently misread real vocab logits as
        duration logits, producing garbled output with no error at all.
        Catching the "too large" / "no room left" case here, right after the
        decoder is compiled, turns both failure modes into one loud, actionable
        error at load time instead of a crash or silent corruption at
        inference time.
        """
        try:
            decoder_output_width = self.dec_compiled.output("outputs").get_partial_shape()[-1].get_length()
        except Exception as e:
            raise RuntimeError(
                f"Could not determine decoder_joint-model.onnx's output width to "
                f"validate the vocab.txt-derived VOCAB_SIZE ({self.VOCAB_SIZE}): {e}. "
                f"Re-run setup: python dictation_engine.py --model parakeet --setup"
            ) from e

        # At least one duration logit must remain after the vocab+blank
        # slice, or _tdt_greedy_decode's duration_logits.argmax() crashes.
        if self.VOCAB_SIZE >= decoder_output_width:
            raise RuntimeError(
                f"Parakeet vocab.txt is inconsistent with decoder_joint-model.onnx: "
                f"the vocab-derived VOCAB_SIZE ({self.VOCAB_SIZE}, BLANK_IDX="
                f"{self.BLANK_IDX}) leaves no room for duration logits in the "
                f"decoder's output width of {decoder_output_width}. This usually "
                f"means vocab.txt is truncated, stale, or paired with a decoder "
                f"model from a different export. Refusing to start transcription "
                f"with mismatched constants. Re-run setup: "
                f"python dictation_engine.py --model parakeet --setup"
            )
        log(
            f"  Vocab constants validated against decoder output width "
            f"({self.VOCAB_SIZE} vocab+blank, "
            f"{decoder_output_width - self.VOCAB_SIZE} duration bins)"
        )

    def _load_pipeline(self):
        """Load preprocessor, encoder, and decoder."""
        log(f"Loading Parakeet TDT pipeline (encoder on {self.device}, decoder on GPU)...")
        start = time.time()

        # 1. Load vocabulary
        self._load_vocab()

        # 2. Load mel preprocessor (onnxruntime)
        preproc_path = self.model_path / "nemo128.onnx"
        if not preproc_path.exists():
            raise FileNotFoundError(
                f"nemo128.onnx not found at {preproc_path}. "
                f"Re-run setup: python dictation_engine.py --model parakeet --setup"
            )
        try:
            import onnxruntime as ort
            self.preproc = ort.InferenceSession(
                str(preproc_path), providers=["CPUExecutionProvider"]
            )
        except ImportError:
            raise ImportError(
                "onnxruntime is required for Parakeet models. "
                "Install with: pip install onnxruntime"
            )
        log(f"  Preprocessor loaded from {preproc_path}")

        # 3. Load encoder on NPU/GPU via OpenVINO — one compiled graph per
        # bucket in MEL_BUCKETS (NPU requires static shapes, so each bucket
        # is its own compile). All compiles share CACHE_DIR, so only the
        # first run per bucket pays the compile cost; see benchmarks/ for
        # first-run cost measurement.
        try:
            import openvino as ov
            import numpy as np
            core = ov.Core()

            encoder_path = self.model_path / "encoder-model.onnx"

            compile_config = {"CACHE_DIR": str(CACHE_DIR)}
            if self.latency_override:
                # UNMEASURED lever (issue #3): NPU's default for this
                # sub-property is "efficiency". Opt-in only; do not assume
                # this helps or hurts until the harness reports a number.
                compile_config["NPU_COMPILATION_MODE_PARAMS"] = "performance-hint-override=latency"

            CACHE_DIR.mkdir(parents=True, exist_ok=True)

            def _fallback_compile(bucket, model_to_compile, cfg, first_exc):
                """NPU/GPU -> CPU device-fallback chain, shared by every
                compile call site so the deep CPU fallback always gets
                CACHE_DIR (uncached CPU compiles are otherwise paid on
                every single startup that hits this path)."""
                if self.device == "CPU":
                    raise first_exc
                fallback = "GPU" if self.device == "NPU" else "CPU"
                log(f"  Encoder bucket {bucket} failed on {self.device}: {first_exc}")
                log(f"  Falling back to {fallback}...")
                try:
                    c = core.compile_model(model_to_compile, fallback, cfg)
                    self.device = fallback
                    log(f"  Encoder bucket {bucket} compiled on {fallback}")
                    return c
                except Exception:
                    log(f"  Falling back to CPU...")
                    c = core.compile_model(model_to_compile, "CPU", {"CACHE_DIR": str(CACHE_DIR)})
                    self.device = "CPU"
                    return c

            for bucket in self.MEL_BUCKETS:
                encoder_model = core.read_model(str(encoder_path))
                encoder_model.reshape({
                    "audio_signal": [1, 128, bucket],
                    "length": [1],
                })
                try:
                    compiled = core.compile_model(encoder_model, self.device, compile_config)
                    log(f"  Encoder bucket {bucket} frames compiled on {self.device}")
                except Exception as e:
                    if "NPU_COMPILATION_MODE_PARAMS" in compile_config and (
                        "NPU_COMPILATION_MODE_PARAMS" in str(e) or "compilation_mode_params" in str(e).lower()
                    ):
                        # This OpenVINO/driver version rejects the latency
                        # override property; drop it and retry rather than
                        # crashing the whole pipeline load.
                        log(f"  latency_override property rejected ({e}); disabling it")
                        del compile_config["NPU_COMPILATION_MODE_PARAMS"]
                        self.latency_override = False
                        try:
                            compiled = core.compile_model(encoder_model, self.device, compile_config)
                            log(f"  Encoder bucket {bucket} frames compiled on {self.device} (no latency_override)")
                        except Exception as e2:
                            # The retry can fail too (e.g. a driver edge
                            # case unrelated to the rejected property) — run
                            # it through the same device-fallback chain a
                            # plain compile failure gets, instead of letting
                            # it escape and abort pipeline load entirely.
                            compiled = _fallback_compile(bucket, encoder_model, compile_config, e2)
                    else:
                        compiled = _fallback_compile(bucket, encoder_model, compile_config, e)
                self.enc_compiled[bucket] = compiled

                # Determine encoder output time dimension via dummy inference
                dummy_mel = np.zeros((1, 128, bucket), dtype=np.float32)
                dummy_len = np.array([bucket], dtype=np.int64)
                dummy_out = compiled({"audio_signal": dummy_mel, "length": dummy_len})
                self.enc_time_dim[bucket] = dummy_out["outputs"].shape[2]
                log(f"  Encoder bucket {bucket} output: [1, {self.ENC_DIM}, {self.enc_time_dim[bucket]}]")

            # 4. Load decoder on GPU (1.8x faster than CPU for sequential loop)
            decoder_path = self.model_path / "decoder_joint-model.onnx"
            decoder_model = core.read_model(str(decoder_path))
            decoder_model.reshape({
                "encoder_outputs": [1, self.ENC_DIM, 1],
                "targets": [1, 1],
                "target_length": [1],
                "input_states_1": [2, 1, self.LSTM_DIM],
                "input_states_2": [2, 1, self.LSTM_DIM],
            })
            try:
                self.dec_compiled = core.compile_model(
                    decoder_model, "GPU", {"CACHE_DIR": str(CACHE_DIR)}
                )
                log(f"  Decoder compiled on GPU")
            except Exception as e:
                log(f"  Decoder failed on GPU: {e}, falling back to CPU")
                self.dec_compiled = core.compile_model(decoder_model, "CPU")
                log(f"  Decoder compiled on CPU")

            # 5. Validate the vocab-derived constants against the decoder's
            # real compiled output width. vocab.txt and decoder_joint-model.onnx
            # are separate files shipped side by side; a partial download, a
            # stale cached vocab.txt from a prior model version, or a tampered
            # HF repo could leave them disagreeing. Fail loudly here rather
            # than letting a bad VOCAB_SIZE crash (or silently corrupt) the
            # first transcription in _tdt_greedy_decode.
            self._validate_vocab_against_decoder()

        except Exception as e:
            log(f"Failed to load Parakeet pipeline: {e}")
            raise

        log(f"Parakeet pipeline loaded in {time.time() - start:.1f}s")

    def _preprocess(self, audio_data) -> tuple:
        """Convert raw audio to mel features using nemo128.onnx."""
        import numpy as np
        waveform = audio_data.reshape(1, -1).astype(np.float32)
        waveform_lens = np.array([waveform.shape[1]], dtype=np.int64)
        features, features_lens = self.preproc.run(
            ["features", "features_lens"],
            {"waveforms": waveform, "waveforms_lens": waveform_lens},
        )
        return features, features_lens

    def _tdt_greedy_decode(self, enc_out, enc_len: int) -> str:
        """TDT greedy decoding loop."""
        import numpy as np

        states_1 = np.zeros((2, 1, self.LSTM_DIM), dtype=np.float32)
        states_2 = np.zeros((2, 1, self.LSTM_DIM), dtype=np.float32)
        tokens = []
        t = 0
        emitted_at_frame = 0

        while t < enc_len:
            frame = enc_out[:, :, t:t+1].astype(np.float32)
            prev_token = tokens[-1] if tokens else self.BLANK_IDX
            target = np.array([[prev_token]], dtype=np.int32)
            target_len = np.array([1], dtype=np.int32)

            result = self.dec_compiled({
                "encoder_outputs": frame,
                "targets": target,
                "target_length": target_len,
                "input_states_1": states_1,
                "input_states_2": states_2,
            })

            output = result["outputs"].squeeze()
            new_states_1 = result["output_states_1"]
            new_states_2 = result["output_states_2"]

            vocab_logits = output[:self.VOCAB_SIZE]
            duration_logits = output[self.VOCAB_SIZE:]
            token_id = int(vocab_logits.argmax())
            duration = int(duration_logits.argmax())

            if token_id != self.BLANK_IDX:
                states_1 = new_states_1
                states_2 = new_states_2
                tokens.append(token_id)
                emitted_at_frame += 1

            if duration > 0:
                t += duration
                emitted_at_frame = 0
            elif token_id == self.BLANK_IDX or emitted_at_frame >= self.MAX_TOKENS_PER_STEP:
                t += 1
                emitted_at_frame = 0

        text = "".join(self.vocab.get(t, "?") for t in tokens)
        text = self.DECODE_SPACE.sub(lambda x: " " if x.group(1) else "", text)
        return text.strip()

    def transcribe(self, audio_data, sample_rate: int = 16000, language: str = "en") -> str:
        """Transcribe audio numpy array to text.

        Same interface as WhisperNPU.transcribe() for drop-in compatibility.
        Note: Parakeet TDT performs automatic language identification and
        does not take a language token at inference time (see the upstream
        model card). The `language` parameter is accepted for interface
        compatibility with WhisperNPU.transcribe() but is intentionally
        unused here -- it is not silently dropped support, the model simply
        has no language-conditioning input to plumb it into. Supported
        languages are declared in MODEL_REGISTRY["parakeet"]["languages"].
        """
        import numpy as np
        start = time.time()

        # Ensure float32 normalized to [-1, 1]
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # 1. Mel spectrogram
        mel, mel_lens = self._preprocess(audio_data)
        actual_frames = mel.shape[2]

        # Pick the smallest bucket that fits, and pad/truncate the mel
        # features to exactly that bucket's frame count. Over-length audio
        # (beyond the largest bucket) is truncated — same ceiling as the
        # pre-bucketing static shape had — but, unlike before, it's now
        # logged instead of silent, so dropped speech is visible to the user.
        bucket, was_truncated = self.select_bucket(actual_frames)
        if was_truncated:
            log(f"  WARNING: {actual_frames / self.MEL_FRAME_RATE:.1f}s audio exceeds the "
                f"largest bucket ({bucket / self.MEL_FRAME_RATE:.0f}s); truncating — "
                f"trailing speech will be dropped from the transcript.")

        mel = self.pad_to_bucket(mel, bucket)
        if was_truncated:
            # Truncation drops real frames, so the encoder's `length` input
            # must shrink to match (see pad_to_bucket()'s docstring).
            actual_frames = bucket
        # else: actual_frames stays the true (shorter) frame count so the
        # encoder does not attend over the zero-padding pad_to_bucket() added.

        # 2. Encoder (NPU/GPU) — dispatch to the compiled graph for this bucket
        enc_result = self.enc_compiled[bucket]({
            "audio_signal": mel,
            "length": np.array([actual_frames], dtype=np.int64),
        })
        enc_out = enc_result["outputs"]
        enc_len = int(enc_result["encoded_lengths"][0])

        # 3. TDT Decoder (CPU)
        text = self._tdt_greedy_decode(enc_out, enc_len)

        elapsed = time.time() - start
        audio_duration = len(audio_data) / sample_rate
        rtf = elapsed / audio_duration if audio_duration > 0 else 0
        log(f"Transcribed {audio_duration:.1f}s audio in {elapsed:.1f}s "
            f"(RTF: {rtf:.2f}) on {self.device}+CPU [Parakeet, bucket={bucket}f]")

        return text


def create_model(model_path: Path, device: str, backend: str):
    """Factory function to create the right model class based on backend.

    Args:
        model_path: Path to model directory.
        device: Device string (NPU, GPU, CPU).
        backend: "whisper" or "parakeet" from MODEL_REGISTRY.

    Returns:
        WhisperNPU or ParakeetNPU instance.
    """
    if backend == "parakeet":
        return ParakeetNPU(model_path, device=device)
    return WhisperNPU(model_path, device=device)


# ---------------------------------------------------------------------------
# Audio recording
# ---------------------------------------------------------------------------
class AudioRecorder:
    """Record audio from microphone using sounddevice."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1,
                 max_record_seconds: float = None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_record_seconds = max_record_seconds
        self.recording = False
        self._frames = []
        self._stream = None
        self._lock = threading.Lock()
        self._timer = None

    def start(self):
        """Start recording."""
        import sounddevice as sd

        with self._lock:
            self._frames = []
            self.recording = True

        def callback(indata, frames, time_info, status):
            with self._lock:
                if self.recording:
                    self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=1024,
            callback=callback,
        )
        self._stream.start()
        log("Recording started...")

        # Enforce max recording duration
        if self.max_record_seconds:
            self._timer = threading.Timer(self.max_record_seconds, self._timeout_stop)
            self._timer.daemon = True
            self._timer.start()

    def _timeout_stop(self):
        """Called when max_record_seconds is reached."""
        if self.recording:
            log(f"Max recording time ({self.max_record_seconds}s) reached, stopping.")
            self.stop()

    def stop(self):
        """Stop recording and return audio as numpy array."""
        import numpy as np

        with self._lock:
            self.recording = False
            frames = list(self._frames)
            self._frames = []

        if self._timer:
            self._timer.cancel()
            self._timer = None

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not frames:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(frames, axis=0).flatten()
        duration = len(audio) / self.sample_rate
        log(f"Recording stopped. Duration: {duration:.1f}s")
        return audio

    @property
    def audio_level(self) -> float:
        """Return current RMS audio level normalized to 0.0-1.0."""
        import numpy as np
        with self._lock:
            if not self._frames or not self.recording:
                return 0.0
            last_frame = self._frames[-1].copy()
        rms = float(np.sqrt(np.mean(last_frame ** 2)))
        # Scale aggressively — typical speech RMS is 0.01-0.05
        return min(rms * 25.0, 1.0)


# ---------------------------------------------------------------------------
# Text output (type into active window)
# ---------------------------------------------------------------------------
def type_text(text: str, auto_enter: bool = False):
    """Type text into the currently active window using keyboard simulation."""
    if not text:
        return

    try:
        # Try pyperclip + keyboard for reliable pasting
        import pyperclip
        import keyboard

        # Save current clipboard
        try:
            old_clipboard = pyperclip.paste()
        except Exception:
            old_clipboard = ""

        # Copy transcribed text and paste
        pyperclip.copy(text)
        time.sleep(0.05)
        keyboard.press_and_release("ctrl+v")
        time.sleep(0.1)

        if auto_enter:
            keyboard.press_and_release("enter")

        # Restore clipboard after a short delay
        def restore():
            time.sleep(0.5)
            try:
                pyperclip.copy(old_clipboard)
            except Exception:
                pass

        threading.Thread(target=restore, daemon=True).start()
        log(f"Typed: {text[:80]}{'...' if len(text) > 80 else ''}")

    except ImportError:
        # Fallback: use ctypes SendInput on Windows
        log("pyperclip/keyboard not found, using ctypes fallback")
        _type_text_ctypes(text)


def _type_text_ctypes(text: str):
    """Fallback text typing using Windows ctypes."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    
    # Use SendInput with Unicode characters
    INPUT_KEYBOARD = 1
    KEYEVENTF_UNICODE = 0x0004
    KEYEVENTF_KEYUP = 0x0002

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", ctypes.c_long),
            ("dy", ctypes.c_long),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [
            ("uMsg", wintypes.DWORD),
            ("wParamL", wintypes.WORD),
            ("wParamH", wintypes.WORD),
        ]

    class INPUT(ctypes.Structure):
        class _INPUT(ctypes.Union):
            _fields_ = [
                ("ki", KEYBDINPUT),
                ("mi", MOUSEINPUT),
                ("hi", HARDWAREINPUT),
            ]
        _fields_ = [("type", wintypes.DWORD), ("_input", _INPUT)]

    for char in text:
        inputs = (INPUT * 2)()
        # Key down
        inputs[0].type = INPUT_KEYBOARD
        inputs[0]._input.ki.wScan = ord(char)
        inputs[0]._input.ki.dwFlags = KEYEVENTF_UNICODE
        # Key up
        inputs[1].type = INPUT_KEYBOARD
        inputs[1]._input.ki.wScan = ord(char)
        inputs[1]._input.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP

        user32.SendInput(2, ctypes.byref(inputs), ctypes.sizeof(INPUT))
        time.sleep(0.002)


# ---------------------------------------------------------------------------
# Audio feedback (chimes)
# ---------------------------------------------------------------------------
def _generate_tone(frequencies: list[float], duration_ms: int = 120,
                   sample_rate: int = 44100, volume: float = 0.3) -> bytes:
    """Generate a smooth WAV tone in memory.

    Args:
        frequencies: List of Hz values. Multiple = chord; sequential list of
            single-element lists for arpeggiated notes.
        duration_ms: Total duration in milliseconds.
        sample_rate: Audio sample rate.
        volume: Peak amplitude 0.0-1.0.
    """
    import struct, io, wave
    import numpy as np

    n_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples, endpoint=False)

    # Sum sine waves for chord
    signal = np.zeros(n_samples, dtype=np.float64)
    for freq in frequencies:
        signal += np.sin(2 * np.pi * freq * t)
    if frequencies:
        signal /= len(frequencies)

    # Smooth envelope: 10ms attack, 40ms decay at end
    attack = min(int(0.010 * sample_rate), n_samples)
    decay = min(int(0.040 * sample_rate), n_samples)
    envelope = np.ones(n_samples)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-decay:] = np.linspace(1, 0, decay)

    signal = (signal * envelope * volume * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(signal.tobytes())
    return buf.getvalue()


def chime_start():
    """Pleasant ascending two-note chime for recording start."""
    try:
        import winsound
        # C5 (523 Hz) then E5 (659 Hz) — major third, bright and short
        tone1 = _generate_tone([523.25], duration_ms=80, volume=0.25)
        tone2 = _generate_tone([659.25], duration_ms=120, volume=0.3)
        winsound.PlaySound(tone1, winsound.SND_MEMORY)
        winsound.PlaySound(tone2, winsound.SND_MEMORY)
    except Exception:
        pass


def chime_stop():
    """Warm C-major chord that gently fades out — confirmation chime."""
    try:
        import winsound
        # C5 + E5 + G5 played together as a soft major chord
        chord = _generate_tone([523.25, 659.25, 783.99], duration_ms=250, volume=0.2)
        winsound.PlaySound(chord, winsound.SND_MEMORY)
    except Exception:
        pass


def chime_warning():
    """Low double-tap warning when model isn't ready."""
    try:
        import winsound
        tone = _generate_tone([330.0], duration_ms=100, volume=0.2)
        winsound.PlaySound(tone, winsound.SND_MEMORY)
        winsound.PlaySound(tone, winsound.SND_MEMORY)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main application loop
# ---------------------------------------------------------------------------
class DictationApp:
    """Main dictation application with hotkey toggle."""

    MAX_HISTORY = 20

    def __init__(self, config: dict):
        self.config = config
        self.recorder = AudioRecorder(
            sample_rate=config["sample_rate"],
            max_record_seconds=config.get("max_record_seconds"),
        )
        self.whisper = None  # Lazy-loaded
        self.is_recording = False
        self._model_ready = threading.Event()
        self._load_error: str | None = None

        # State machine & callbacks (used by GUI, harmless in CLI mode)
        self._state = AppState.LOADING
        self._callbacks: list = []
        self._history: list[dict] = []

    # -- Callback system -----------------------------------------------

    def add_callback(self, fn):
        """Register a callback ``fn(state: AppState, data: dict)``."""
        self._callbacks.append(fn)

    def _set_state(self, state: AppState, data: dict | None = None):
        """Update state and notify all callbacks."""
        self._state = state
        data = data or {}
        for cb in self._callbacks:
            try:
                cb(state, data)
            except Exception:
                pass  # Never let a broken callback crash the engine

    @property
    def state(self) -> AppState:
        return self._state

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    # -- Model management -----------------------------------------------

    def ensure_model(self):
        """Ensure model is loaded."""
        if self.whisper is None:
            model_path = setup_model(self.config)
            model_info = MODEL_REGISTRY[self.config["model_size"]]
            self.whisper = create_model(
                model_path, device=self.config["device"],
                backend=model_info["backend"],
            )

    def _load_model_background(self):
        """Load model in background thread, setting _model_ready when done."""
        self._set_state(AppState.LOADING)
        try:
            self.ensure_model()
            self._model_ready.set()
            self._set_state(AppState.READY)
            log("Ready! Waiting for hotkey...")
        except Exception as e:
            self._load_error = str(e)
            self._model_ready.set()  # Unblock waiters so they can see the error
            self._set_state(AppState.ERROR, {"error": str(e)})
            log(f"Model loading failed: {e}")

    def fallback_device(self, new_device: str):
        """Clear error state and reload model on a different device."""
        log(f"Falling back to device: {new_device}")
        self._load_error = None
        self._model_ready.clear()
        self.whisper = None
        self.config["device"] = new_device
        threading.Thread(target=self._load_model_background, daemon=True).start()

    # -- Input device enumeration ----------------------------------------

    @staticmethod
    def list_input_devices() -> list[dict]:
        """Return available audio input devices as list of dicts with 'index' and 'name'."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            result = []
            for i, d in enumerate(devices):
                if d["max_input_channels"] > 0:
                    result.append({"index": i, "name": d["name"]})
            return result
        except Exception:
            return []

    # -- Recording -------------------------------------------------------

    def toggle_recording(self):
        """Toggle recording on/off (Supports Tap-to-Toggle and Push-To-Talk)."""
        import time
        import threading
        
        if getattr(self, "_hotkey_held", False):
            # Ignore auto-repeat while the key is physically held
            return
            
        self._hotkey_held = True
        press_time = time.time()

        def _do_stop():
            if not self.is_recording:
                return
            if self.config["beep_on_start"]:
                threading.Thread(target=chime_stop, daemon=True).start()

            audio = self.recorder.stop()
            self.is_recording = False

            if len(audio) < self.config["sample_rate"] * 0.3:
                log("Recording too short, ignoring.")
                self._set_state(AppState.READY)
                return

            self._set_state(AppState.PROCESSING)

            try:
                self.ensure_model()
                text = self.whisper.transcribe(
                    audio,
                    sample_rate=self.config["sample_rate"],
                    language=self.config["language"],
                )
                if text:
                    type_text(text, auto_enter=self.config["auto_enter"])
                    # Store in history
                    self._history.append({
                        "timestamp": datetime.now().isoformat(),
                        "text": text,
                        "duration": len(audio) / self.config["sample_rate"],
                    })
                    if len(self._history) > self.MAX_HISTORY:
                        self._history = self._history[-self.MAX_HISTORY:]
                    self._set_state(AppState.READY, {"text": text})
                else:
                    log("No speech detected.")
                    self._set_state(AppState.READY)
            except RuntimeError as e:
                if "DEVICE_LOST" in str(e) or "device hung" in str(e):
                    self._set_state(AppState.ERROR, {"error": str(e), "device_lost": True})
                else:
                    self._set_state(AppState.ERROR, {"error": str(e)})
            except Exception as e:
                self._set_state(AppState.ERROR, {"error": str(e)})

        def _do_start():
            if not self._model_ready.is_set():
                log("Model still loading, please wait...")
                chime_warning()
                return

            if self._load_error:
                log(f"Cannot record — model failed to load: {self._load_error}")
                return

            self.is_recording = True
            if self.config["beep_on_start"]:
                threading.Thread(target=chime_start, daemon=True).start()

            self.recorder.start()
            self._set_state(AppState.RECORDING)

        def _watch_key(started_recording):
            import keyboard
            # Wait until the hotkey is physically released
            while keyboard.is_pressed(self.config["hotkey"]):
                time.sleep(0.05)
            
            self._hotkey_held = False
            
            # If we started recording and the user held the key for > 0.4s,
            # treat it as Push-To-Talk and stop recording upon release.
            duration = time.time() - press_time
            if started_recording and duration > 0.4:
                _do_stop()

        if self.is_recording:
            # Stop recording immediately in a thread
            threading.Thread(target=_do_stop, daemon=True).start()
            threading.Thread(target=_watch_key, args=(False,), daemon=True).start()
        else:
            # Start recording immediately in a thread
            threading.Thread(target=_do_start, daemon=True).start()
            threading.Thread(target=_watch_key, args=(True,), daemon=True).start()

    # -- Lifecycle -------------------------------------------------------

    def start_background(self):
        """Register hotkey and start model loading without blocking.

        Use this from GUI mode instead of ``run()`` — does NOT call
        ``keyboard.wait()``, so the caller's main loop stays in control.
        """
        import keyboard

        hotkey = self.config["hotkey"]
        keyboard.add_hotkey(hotkey, self.toggle_recording, suppress=True)
        log(f"Hotkey {hotkey} registered.")

        log("Loading model in background (first time may take several minutes)...")
        threading.Thread(target=self._load_model_background, daemon=True).start()

    def stop(self):
        """Clean shutdown — unhook keyboard and stop recording if active."""
        try:
            import keyboard
            keyboard.unhook_all()
        except Exception:
            pass
        if self.is_recording:
            self.recorder.stop()
            self.is_recording = False
        log("Engine stopped.")

    def run(self):
        """Run the main application loop with global hotkey (CLI mode)."""
        import keyboard

        hotkey = self.config["hotkey"]

        log("=" * 60)
        log("NPU Dictation Engine")
        log(f"  Device:  {self.config['device']}")
        log(f"  Model:   {self.config['model_size']}")
        log(f"  Hotkey:  {hotkey}")
        log(f"  Lang:    {self.config['language']}")
        log(f"  Auto-Enter: {self.config['auto_enter']}")
        log("=" * 60)
        log(f"Press {hotkey} to start/stop dictation. Ctrl+C to quit.")
        log("")

        # Register global hotkey immediately so it's responsive during loading
        keyboard.add_hotkey(hotkey, self.toggle_recording, suppress=True)

        # Load model in background so hotkey is responsive during load
        log("Loading model in background (first time may take several minutes)...")
        load_thread = threading.Thread(target=self._load_model_background, daemon=True)
        load_thread.start()

        try:
            keyboard.wait()  # Block forever, handling hotkeys
        except KeyboardInterrupt:
            log("\nShutting down...")
            self.stop()


# ---------------------------------------------------------------------------
# Setup / Install dependencies
# ---------------------------------------------------------------------------
def run_setup():
    """Interactive setup: install dependencies and export model."""
    import subprocess

    log("=" * 60)
    log("NPU Dictation Engine - Setup")
    log("=" * 60)

    # 1. Check Python version
    log(f"Python: {sys.version}")
    
    # 2. Install pip dependencies
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        log("\nInstalling dependencies from requirements.txt...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file), "--quiet"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            log(f"  WARNING: pip install failed: {result.stderr[:500]}")
        else:
            log("  Dependencies installed successfully.")
    else:
        log(f"\nWARNING: requirements.txt not found at {requirements_file}")
        log("  Install manually: pip install -r requirements.txt")

    # 3. Check NPU availability
    log("\nChecking available devices...")
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        log(f"  Available OpenVINO devices: {devices}")
        
        if "NPU" in devices:
            log("  [OK] NPU detected!")
        else:
            log("  [--] NPU not found. Will fall back to GPU or CPU.")
            log("    Make sure Intel NPU driver is installed via Windows Update")
            log("    or from: https://www.intel.com/content/www/us/en/download/794734/")
        
        if "GPU" in devices:
            log("  [OK] GPU detected (Intel iGPU)")
    except ImportError:
        log("  Could not import openvino - installation may have failed")

    # 4. Create default config
    config = load_config()
    
    # Auto-detect best device
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        if "NPU" in devices:
            config["device"] = "NPU"
        elif "GPU" in devices:
            config["device"] = "GPU"
        else:
            config["device"] = "CPU"
    except Exception:
        config["device"] = "CPU"

    save_config(config)

    # 5. Download model
    log(f"\nDownloading {config['model_size']} model for {config['device']}...")
    model_path = None
    try:
        model_path = setup_model(config)
        log(f"Model ready at: {model_path}")
    except Exception as e:
        log(f"Model download failed: {e}")
        log("You can retry later with: python dictation_engine.py --setup")

    # 6. Warm the OpenVINO cache by running a dummy inference
    #    This triggers NPU compilation during setup so the first real use is fast.
    if model_path:
        log(f"\nWarming OpenVINO cache on {config['device']} (first compile may take 5-15 min)...")
        try:
            import numpy as np
            model_info = MODEL_REGISTRY[config["model_size"]]
            model = create_model(model_path, device=config["device"],
                                 backend=model_info["backend"])
            silence = np.zeros(config["sample_rate"], dtype=np.float32)  # 1 second of silence
            model.transcribe(silence, sample_rate=config["sample_rate"], language=config["language"])
            log("Cache warm-up complete — subsequent starts will be fast.")
        except Exception as e:
            log(f"Cache warm-up failed (non-fatal): {e}")
            log("The cache will be built on first real use instead.")

    log("\n" + "=" * 60)
    log("Setup complete!")
    log(f"Config file: {CONFIG_FILE}")
    log(f"Start dictating: python dictation_engine.py")
    log(f"Or use the PowerShell launcher: .\\Start-Dictation.ps1")
    log("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="NPU Dictation Engine")
    parser.add_argument("--setup", action="store_true", help="Run first-time setup")
    parser.add_argument("--device", choices=["NPU", "GPU", "CPU"], help="Override device")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), help="Model size")
    parser.add_argument("--language", type=str, help="Language code (e.g., en, ru, id)")
    parser.add_argument("--auto-enter", action="store_true", help="Press Enter after typing")
    parser.add_argument("--hotkey", type=str, help="Global hotkey (e.g., ctrl+alt+d)")
    args = parser.parse_args()

    if args.setup:
        run_setup()
        return

    config = load_config()

    # Apply CLI overrides
    if args.device:
        config["device"] = args.device
    if args.model:
        config["model_size"] = args.model
    if args.language:
        config["language"] = args.language
    if args.auto_enter:
        config["auto_enter"] = True
    if args.hotkey:
        config["hotkey"] = args.hotkey

    # Auto-select device based on model when --device not explicitly set
    if not args.device:
        model_info = MODEL_REGISTRY[config["model_size"]]
        preferred = model_info["preferred_device"]
        if config["device"] != preferred:
            log(f"Auto-selecting {preferred} for {config['model_size']} "
                f"(override with --device {config['device']})")
            config["device"] = preferred

    validate_config(config)

    app = DictationApp(config)
    app.run()


if __name__ == "__main__":
    main()