"""
Débora Whisper - Local voice-to-text on Intel NPU, GPU, NVIDIA CUDA or CPU
Supports Whisper (via openvino_genai) and Parakeet TDT (via OpenVINO + onnxruntime).

Usage: debora-cli [--setup] [--device NPU|GPU|CPU] [--model base|small|medium|parakeet]
"""

import sys
import os
import time
import json
import re
import argparse
import inspect
import threading
from collections import deque
from enum import Enum
from pathlib import Path
from datetime import datetime

from debora_whisper import paths
from debora_whisper.harness import MemoryHotwords
from debora_whisper.paths import CACHE_DIR, CONFIG_DIR, CONFIG_FILE, LOG_DIR, MODEL_DIR
from debora_whisper.vad_endpoint import AdaptiveEndpoint, VadSegment
from debora_whisper.voice_chat import (VoiceChat, download_llm, ensure_tts_server, is_http_url,
                                    llm_loaded, load_llm)

# Disable HuggingFace symlinks on Windows to avoid WinError 1314
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOG_FILE = LOG_DIR / "app.log"
TELEMETRY_LOG = LOG_DIR / "telemetry.log"
TTS_SERVER_LOG = LOG_DIR / "tts_server.log"
LLM_SERVER_LOG = LOG_DIR / "llm_server.log"
# At startup a larger log moves to <name>.1, replacing the previous one.
LOG_MAX_BYTES = 5_000_000
# An NPU lost to DEVICE_LOST, until a recovery probe or a reboot (npu_lost_this_boot).
NPU_LOST_FILE = CONFIG_DIR / "npu_lost.json"

DEFAULT_CONFIG = {
    "device": "NPU",           # Active device; chosen from device_priority at startup
    # Startup picks the first present device; a lost device falls back to the
    # next healthy one. RTX first: turbo transcribes 5-7x faster there than on
    # the NPU (README, "Benchmark"). Edit config.json to reorder.
    "device_priority": ["CUDA", "NPU", "GPU", "CPU"],
    "model_size": "turbo",     # see MODEL_REGISTRY; turbo: large-v3-turbo
    "language": "en",          # Language code or "auto"
    "hotkey": "ctrl+space",    # Global hotkey to toggle recording
    "auto_enter": False,       # Press Enter after pasting (useful for Claude Code)
    # Continuous drafts: shown in the overlay only, or also typed into the
    # target and rewritten with Shift+Left. Rewriting assumes the editor
    # leaves text and caret alone, which autocomplete, auto-closing pairs,
    # autocorrect and slow targets (browsers, Word, IDEs) do not.
    "inline_drafts": False,
    "beep_on_start": True,     # Audio feedback when recording starts/stops
    "max_record_seconds": 60,  # Max recording length
    "sample_rate": 16000,      # Whisper expects 16kHz
    "show_balloon": True,      # Show text balloon under notch after transcription
    "continuous_listening": False, # Start in continuous (VAD) listening
    "vad_end_silence_seconds": 1.5,
    # Total silence allowed when the latest draft lacks sentence-ending punctuation.
    "vad_incomplete_silence_seconds": 3.0,
    # Hotkey: holding it is always push-to-talk. A tap either starts
    # continuous listening (VAD types each sentence; tap again to stop) or,
    # with "toggle", starts a recording that the next tap stops.
    "tap_action": "continuous",
    # Continuous listening stops by itself after this long without speech
    # (null: never), so a stray tap does not leave the microphone open.
    "continuous_idle_stop_seconds": 120,
    # Voice chat replies are spoken by the Chatterbox TTS server.
    "voice_chat": True,
    "voice_chat_backend": "local",  # local (Qwen/OpenVINO) or claude
    "harness_cwd": None,           # null: the user's home directory
    "harness_model": None,         # null: Claude Code's default
    "harness_permission_mode": "acceptEdits",
    "harness_permission_response": "deny",  # requests not already allowed by Claude
    "harness_allowed_tools": None,  # null uses the packaged read-only diagnostics
    "harness_prompt_file": None,   # null: packaged voice-channel rules
    "harness_memory_file": None,   # null: ~/.debora/harness/voice_memory.md
    "harness_hotwords": True,     # Voice memory hints only for Claude voice chat
    # Silence that ends a sentence in voice chat (dictation: 1.5 s, room to
    # think). The reply cannot start before it has passed.
    "voice_chat_end_silence_seconds": 0.8,
    # Hugging Face repo (OpenVINO IR) or local directory. The int4-cw export
    # answers in ~0.4 s at ~15 tokens/s on a Core Ultra's Arc iGPU.
    "llm_model": "OpenVINO/Qwen3-8B-int4-cw-ov",
    "llm_device": "GPU",           # OpenVINO device; falls back to CPU
    "llm_prompt": None,            # null: voice_chat.DEFAULT_VOICE_CHAT_PROMPT
    # Reference audio for Chatterbox to clone (~10 s of clean speech): a
    # file path, or a name looked up as <name>.wav in the voices folder
    # (paths.VOICES_DIR), then among the bundled voices (carol, debora,
    # debora_v2, isabel, mari). null: Chatterbox's own voice.
    "tts_voice": "debora_v2",
    "tts_url": "http://127.0.0.1:8765",
    "tts_timeout_seconds": 60,
    # Command that starts the TTS server when nothing answers at tts_url.
    # null: uv runs debora_whisper/tts_server.py with tts_voice.
    "tts_server_command": None,
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
        # The FluidInference int4 NPU export hung the NPU (DEVICE_LOST)
        # within a few dozen real-speech inferences, even outside the app,
        # and OpenVINO 2026 refuses to load it (no beam_idx input). The
        # official int8 export ran 80/80 on NPU on both 2025.4 and 2026.4.
        # New local_dir so existing installs download it instead of reusing
        # the int4 files.
        "ov_repo": "OpenVINO/whisper-large-v3-turbo-int8-ov",
        "description": "809M params, 2.3% WER. Best multilingual quality.",
        "preferred_device": "GPU",
        "backend": "whisper",
        "local_dir": "whisper-turbo-int8-openvino",
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
    return model_files_complete(MODEL_DIR / info["local_dir"])


def model_files_complete(path: Path) -> bool:
    """True when path holds a model (Whisper .xml, Parakeet .onnx). Each .xml
    needs its .bin weights: an interrupted download leaves the .xml files
    and missing or empty .bin files, which then fail to load."""
    if not path.exists():
        return False
    xmls = list(path.glob("*.xml"))
    if xmls:
        return all(x.with_suffix(".bin").is_file() and x.with_suffix(".bin").stat().st_size > 0
                   for x in xmls)
    return any(path.glob("*.onnx"))


# ---------------------------------------------------------------------------
# Application state machine
# ---------------------------------------------------------------------------
class AppState(Enum):
    LOADING = "loading"        # Model loading in progress
    READY = "ready"            # Idle, waiting for hotkey
    RECORDING = "recording"    # Microphone active
    PROCESSING = "processing"  # Transcribing audio
    SPEAKING = "speaking"      # Voice chat: the reply is being spoken
    ERROR = "error"            # Device lost or load failed


# On Windows, append mode seeks to the end and then writes: two threads
# logging at once could write at the same offset, and the shorter line
# overwrote the start of the longer one ("ore transcription...").
_log_lock = threading.Lock()


def log(msg: str):
    """Simple logging to file and console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    with _log_lock:
        print(line)
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            target_file = TELEMETRY_LOG if "[Telemetry]" in msg else LOG_FILE
            with open(target_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


def rotate_logs():
    """Keep each log under LOG_MAX_BYTES plus one older file."""
    for path in (LOG_FILE, TELEMETRY_LOG, TTS_SERVER_LOG, LLM_SERVER_LOG):
        try:
            if path.stat().st_size > LOG_MAX_BYTES:
                os.replace(path, path.with_name(path.name + ".1"))
        except OSError:
            pass  # missing, or open in another process


def log_folder_moves():
    """Report the legacy (npu-whisper) folders this run moved or could not move."""
    for line in paths.MIGRATIONS:
        log(line)


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
    valid_devices = set(VALID_DEVICES)
    if config.get("device") not in valid_devices:
        raise ValueError(f"device must be one of {valid_devices}, got '{config.get('device')}'")

    priority = config.get("device_priority")
    if (not isinstance(priority, list) or not priority
            or any(d not in valid_devices for d in priority)):
        raise ValueError(f"device_priority must be a non-empty list of {valid_devices}, got {priority!r}")

    valid_models = set(MODEL_REGISTRY.keys())
    if config.get("model_size") not in valid_models:
        raise ValueError(f"model_size must be one of {valid_models}, got '{config.get('model_size')}'")

    sr = config.get("sample_rate")
    if not isinstance(sr, (int, float)) or sr <= 0:
        raise ValueError(f"sample_rate must be a positive number, got '{sr}'")

    max_rec = config.get("max_record_seconds")
    if max_rec is not None and (not isinstance(max_rec, (int, float)) or max_rec <= 0):
        raise ValueError(f"max_record_seconds must be a positive number or null, got '{max_rec}'")

    for key in ("vad_end_silence_seconds", "vad_incomplete_silence_seconds"):
        seconds = config.get(key, DEFAULT_CONFIG[key])
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not 0 < seconds < float("inf")):
            raise ValueError(f"{key} must be a finite positive number, got {seconds!r}")

    tap_action = config.get("tap_action", "continuous")
    if tap_action not in TAP_ACTIONS:
        raise ValueError(f"tap_action must be one of {TAP_ACTIONS}, got {tap_action!r}")

    for key in ("llm_model", "llm_device"):
        value = config.get(key, DEFAULT_CONFIG[key])
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty string, got {value!r}")
    if config.get("voice_chat_backend", "local") not in ("local", "claude"):
        raise ValueError("voice_chat_backend must be local or claude")
    if not isinstance(config.get("harness_hotwords", True), bool):
        raise ValueError("harness_hotwords must be a bool")
    for key in ("harness_cwd", "harness_model", "harness_prompt_file", "harness_memory_file"):
        value = config.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{key} must be null or a non-empty string, got {value!r}")
    from debora_whisper.harness import PERMISSION_MODES
    if config.get("harness_permission_mode", "acceptEdits") not in PERMISSION_MODES:
        raise ValueError(f"harness_permission_mode must be one of {PERMISSION_MODES}")
    if config.get("harness_permission_response", "deny") not in ("deny", "allow"):
        raise ValueError("harness_permission_response must be deny or allow")
    rules = config.get("harness_allowed_tools")
    if rules is not None and (not isinstance(rules, list) or any(
            not isinstance(rule, str) or not rule.strip() for rule in rules)):
        raise ValueError("harness_allowed_tools must be null or a list of non-empty permission rules")
    voice = config.get("tts_voice")
    if voice is not None and (not isinstance(voice, str) or not voice.strip()):
        raise ValueError(f"tts_voice must be null or a file path or name, got {voice!r}")
    url = config.get("tts_url", DEFAULT_CONFIG["tts_url"])
    if not is_http_url(url):
        raise ValueError(f"tts_url must be an http(s) URL, got {url!r}")
    command = config.get("tts_server_command")
    if command is not None and (not isinstance(command, list) or not command
                                or not all(isinstance(a, str) and a for a in command)):
        raise ValueError(f"tts_server_command must be null or a list of strings, got {command!r}")
    timeout = config.get("tts_timeout_seconds", DEFAULT_CONFIG["tts_timeout_seconds"])
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError(f"tts_timeout_seconds must be a positive number, got {timeout!r}")


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

    if config["device"] == "CUDA":
        return None

    # Check if model already exists (Whisper uses .xml, Parakeet uses .onnx)
    if model_files_complete(model_path):
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
        if model_files_complete(model_path):
            log(f"Model downloaded to {model_path}")
            return model_path
    except Exception as e:
        log(f"Download failed: {e}")
        sys.exit(1)

    log(f"No model files found at {model_path}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Accelerator failure policy
# ---------------------------------------------------------------------------
# OpenCL runtime errors that OpenVINO reports from the GPU plugin. After one of
# these, OpenVINO warns that later OpenCL calls may hang, so the process must not
# touch the GPU (or anything sharing its context) again.
_GPU_FATAL_MARKERS = (
    "CL_OUT_OF_RESOURCES", "CL_OUT_OF_HOST_MEMORY",
    "CL_MEM_OBJECT_ALLOCATION_FAILURE", "CL_DEVICE_NOT_AVAILABLE",
    "CL_INVALID_COMMAND_QUEUE", "subsequent OpenCL calls",
)
_DEVICE_LOSS_MARKERS = ("device_lost", "device lost", "device hung")


class DeviceFailureError(RuntimeError):
    """An accelerator failed at runtime. ``device`` is GPU, NPU or UNKNOWN."""

    def __init__(self, device: str, cause: BaseException):
        self.device = device
        self.detail = _failure_detail(cause)
        super().__init__(f"{device} device failure: {self.detail}")


class RestartRequiredError(RuntimeError):
    """Raised for any load/inference attempt after a fatal device failure."""


_device_failure_lock = threading.Lock()
_device_failure: dict | None = None

# Held by a transcription and by the LLM's compile (in its own process, on
# the Arc iGPU): both times that compile overlapped an NPU inference, the
# NPU was lost. Process-wide, so an engine rebuilt by Settings honours it.
_inference_gate = threading.Lock()


def _exception_chain(exc: BaseException):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def has_nvidia_gpu(return_name: bool = False):
    """Check if an NVIDIA GPU is present and accessible via nvidia-smi.
    If return_name is True, returns the GPU name string or None if not found."""
    import shutil
    import os
    if not shutil.which("nvidia-smi"):
        return None if return_name else False
    try:
        import subprocess
        
        args = ["nvidia-smi"]
        if return_name:
            args = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
            
        output = subprocess.check_output(
            args,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        )
        return output.decode("utf-8").strip() if return_name else True
    except Exception:
        return None if return_name else False


VALID_DEVICES = ("CUDA", "NPU", "GPU", "CPU")
TAP_ACTIONS = ("continuous", "toggle")


def detect_devices() -> set:
    """Devices this machine can run inference on. CPU is always present."""
    import importlib.util
    found = {"CPU"}
    if has_nvidia_gpu():
        # faster-whisper ships in the optional [cuda] extra; without it CUDA
        # would be picked first and then fail to load instead of falling back.
        if importlib.util.find_spec("faster_whisper") is not None:
            found.add("CUDA")
        else:
            log("NVIDIA GPU found but faster-whisper is not installed; "
                "install the [cuda] extra to use it")
    try:
        import openvino as ov
        # "GPU.0"/"GPU.1" on multi-GPU machines; the app addresses "GPU".
        found.update(d.split(".")[0] for d in ov.Core().available_devices)
    except Exception as e:
        log(f"OpenVINO device query failed: {e}")
    return found


def device_supports_model(device: str, model_key: str) -> bool:
    # faster-whisper only runs Whisper; Parakeet has no CUDA backend.
    return device != "CUDA" or MODEL_REGISTRY[model_key]["backend"] == "whisper"


def select_device(config: dict, available: set, exclude=()) -> str | None:
    """First device of config["device_priority"] that is present, runs the
    configured model and is not excluded (e.g. the device that just failed).
    """
    for device in config["device_priority"]:
        if (device in available and device not in exclude
                and device_supports_model(device, config["model_size"])):
            return device
    return None


def apply_device_priority(config: dict):
    """Set config["device"] to the first present device of device_priority."""
    chosen = select_device(config, detect_devices())
    if chosen is None:
        raise SystemExit(f"No device in device_priority {config['device_priority']} "
                         f"can run {config['model_size']}")
    log(f"Device priority {config['device_priority']} -> {chosen} "
        f"for {config['model_size']} (override with --device)")
    config["device"] = chosen


def _failure_detail(exc: BaseException) -> str:
    """Short identifier of the original error (the deepest cause wins)."""
    chain = list(_exception_chain(exc))
    text = " ".join(str(e) for e in chain)
    for marker in _GPU_FATAL_MARKERS[:-1]:
        if marker in text:
            return marker
    root = chain[-1]
    return f"{type(root).__name__}: {str(root)[:120]}"


def classify_device_failure(exc: BaseException, active_devices=()) -> str | None:
    """Return GPU, NPU or UNKNOWN for a fatal accelerator error, else None.

    The decision uses the error text of the whole exception chain and the
    devices the loaded backend really uses (Parakeet runs its decoder on GPU
    even when the configured device is NPU), never the configured device alone.
    """
    if isinstance(exc, DeviceFailureError):
        return exc.device
    text = " ".join(str(e) for e in _exception_chain(exc))
    if any(m in text for m in _GPU_FATAL_MARKERS):
        return "GPU"
    if not any(m in text.lower() for m in _DEVICE_LOSS_MARKERS):
        return None
    if "[GPU]" in text:
        return "GPU"
    if "[NPU]" in text or "ZE_RESULT" in text:
        return "NPU"
    devices = {str(d).upper() for d in active_devices if d}
    if "GPU" in devices:
        # A hybrid NPU+GPU pipeline cannot tell which one was lost: fail closed.
        return "UNKNOWN" if "NPU" in devices else "GPU"
    if "NPU" in devices:
        return "NPU"
    return "UNKNOWN"


def restart_required_message(device: str, detail: str) -> str:
    name = "The accelerator" if device == "UNKNOWN" else f"The {device}"
    return (
        f"{name} failed ({detail}). Dictation is disabled until Débora Whisper "
        f"is restarted, because further calls could hang the driver. Quit and "
        f"restart the app. If it happens again, pick another device in "
        f"Settings before restarting."
    )


def record_device_failure(device: str, exc: BaseException) -> dict:
    """Latch a fatal accelerator failure for the rest of this process."""
    global _device_failure
    with _device_failure_lock:
        if _device_failure is None:
            detail = _failure_detail(exc)
            _device_failure = {
                "device": device,
                "detail": detail,
                "message": restart_required_message(device, detail),
                "exception": exc,
            }
        return dict(_device_failure)


def _boot_time() -> float | None:
    """When Windows last started (seconds since the epoch), or None."""
    if sys.platform != "win32":
        return None
    import ctypes
    ctypes.windll.kernel32.GetTickCount64.restype = ctypes.c_uint64
    return time.time() - ctypes.windll.kernel32.GetTickCount64() / 1000


def remember_npu_loss(detail: str):
    try:
        NPU_LOST_FILE.parent.mkdir(parents=True, exist_ok=True)
        NPU_LOST_FILE.write_text(json.dumps(
            {"time": time.time(), "boot": _boot_time(), "detail": detail}), encoding="utf-8")
    except Exception as e:
        log(f"Cannot record the NPU failure in {NPU_LOST_FILE}: {e}")


def forget_npu_loss():
    """The NPU works again (the recovery probe ran a model on it)."""
    try:
        NPU_LOST_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def npu_lost_this_boot() -> dict | None:
    """The NPU failure recorded since Windows last started, or None.

    After a loss, loading a model on the NPU in the app froze it for
    minutes (console and Ctrl+C included): the load holds the GIL while the
    driver recovers or the model compiles again from scratch (~3 min for
    turbo). Until the recovery probe (npu_probe, its own process) has run
    the model on the NPU, the app starts on another device."""
    try:
        record = json.loads(NPU_LOST_FILE.read_text(encoding="utf-8"))
        boot = _boot_time()
        if boot is not None and record.get("boot") is not None \
                and abs(record["boot"] - boot) < 120:
            return record
    except (OSError, ValueError, AttributeError):
        return None
    try:
        NPU_LOST_FILE.unlink()  # from before the last reboot
    except OSError:
        pass
    return None


def avoid_lost_npu(config: dict) -> bool:
    """Move config["device"] off an NPU lost earlier in this boot; True if
    it moved (the GUI then probes the NPU in the background)."""
    if config["device"] != "NPU" or not (lost := npu_lost_this_boot()):
        return False
    when = datetime.fromtimestamp(lost["time"]).strftime("%H:%M")
    chosen = select_device(config, detect_devices(), exclude={"NPU"}) or "CPU"
    log(f"The NPU failed at {when} ({lost.get('detail')}); starting on {chosen} "
        f"until a check in a separate process finds it working again.")
    config["device"] = chosen
    return True


def device_failure() -> dict | None:
    """The latched fatal failure for this process, or None."""
    with _device_failure_lock:
        return dict(_device_failure) if _device_failure else None


def ensure_devices_usable():
    """Refuse any model load/inference once the process has a fatal failure."""
    failure = device_failure()
    if failure:
        raise RestartRequiredError(failure["message"]) from failure["exception"]


def _reset_device_failure_for_tests():
    global _device_failure
    with _device_failure_lock:
        _device_failure = None


def _model_active_devices(model) -> set:
    if model is None:
        return set()
    getter = getattr(model, "active_devices", None)
    if callable(getter):
        try:
            return set(getter())
        except Exception:
            return set()
    device = getattr(model, "device", None)
    return {device} if isinstance(device, str) else set()


# ---------------------------------------------------------------------------
# Whisper pipeline using Faster Whisper (CUDA)
# ---------------------------------------------------------------------------
class FasterWhisperCUDA:
    """Whisper speech-to-text using faster-whisper on NVIDIA CUDA."""

    def __init__(self, model_size: str, device: str = "cuda"):
        self.device = device
        self.model_size = model_size
        self.pipeline = None
        self._load_pipeline()

    def active_devices(self) -> set:
        return {"CUDA"}

    def _load_pipeline(self):
        ensure_devices_usable()
        
        # On Windows, CTranslate2 needs to find the CUDA runtime DLLs (cublas, cudnn).
        # If installed via pip (nvidia-cublas-cu12, nvidia-cudnn-cu12), we must explicitly
        # add them to the DLL search path.
        if sys.platform == "win32":
            try:
                import site
                for site_pkg in site.getsitepackages():
                    nvidia_base = Path(site_pkg) / "nvidia"
                    if nvidia_base.exists():
                        for lib_dir in nvidia_base.iterdir():
                            bin_path = lib_dir / "bin"
                            if bin_path.exists():
                                os.environ["PATH"] = str(bin_path) + os.pathsep + os.environ.get("PATH", "")
                                if hasattr(os, "add_dll_directory"):
                                    os.add_dll_directory(str(bin_path))
            except Exception as e:
                log(f"Warning: Failed to inject NVIDIA DLL paths: {e}")

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise RuntimeError("faster-whisper is not installed. Please install it to use CUDA backend.")
            
        log(f"Loading faster-whisper pipeline on CUDA ({self.model_size})...")
        
        # compute_type="int8_float16" gives the best performance/VRAM tradeoff on RTX.
        start = time.time()
        self.pipeline = WhisperModel(self.model_size, device="cuda", compute_type="int8_float16")
        hw_name = has_nvidia_gpu(return_name=True) or "NVIDIA GPU"
        log(f"Loaded faster-whisper on {hw_name} in {time.time() - start:.1f}s")

    def supports_hotwords(self) -> bool:
        try:
            parameter = inspect.signature(self.pipeline.transcribe).parameters.get("hotwords")
        except (TypeError, ValueError):
            return False
        return parameter is not None and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)

    def transcribe(self, audio_data, sample_rate: int = 16000, language: str = "en",
                   hotwords: str | None = None) -> str:
        import numpy as np
        start = time.time()

        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        ensure_devices_usable()

        # temperature=0.0 disables faster-whisper's temperature fallback.
        # Dense dictation near 30s has a compression ratio above its 2.4
        # threshold, so the default re-decoded up to 5 times (0.7s -> 3s on
        # an RTX 4070) and returned a temperature-1.0 sample. The OpenVINO
        # path is greedy with no fallback; this matches it.
        hints = {"hotwords": hotwords} if hotwords and self.supports_hotwords() else {}
        segments, info = self.pipeline.transcribe(
            audio_data,
            language=language if language != "auto" else None,
            condition_on_previous_text=False,
            without_timestamps=True,
            temperature=0.0,
            **hints,
        )

        # segments is a lazy generator: decoding happens here.
        text = "".join(segment.text for segment in segments).strip()

        elapsed = time.time() - start
        audio_duration = len(audio_data) / sample_rate
        rtf = elapsed / audio_duration if audio_duration > 0 else 0
        log(f"Transcribed {audio_duration:.1f}s audio in {elapsed:.1f}s (RTF: {rtf:.2f}) on CUDA")

        return text


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

    def active_devices(self) -> set:
        """Devices the loaded pipeline actually runs on."""
        return {self.device}

    def _load_pipeline(self):
        """Load the OpenVINO Whisper pipeline."""
        # Warn about large models on NPU — they may trigger driver instability
        model_name = self.model_path.name.lower()
        if self.device == "NPU" and ("turbo" in model_name or "large" in model_name or "medium" in model_name):
            log(f"NOTE: Large models on NPU may cause driver instability (DEVICE_LOST).")
            log(f"  If inference fails, try rebooting to reset the NPU, or use --device GPU")

        ensure_devices_usable()
        log(f"Loading Whisper pipeline on {self.device}...")
        start = time.time()

        try:
            import openvino_genai as ov_genai
            import openvino as ov
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self.pipeline = ov_genai.WhisperPipeline(
                str(self.model_path), self.device,
                CACHE_DIR=str(CACHE_DIR)
            )
            try:
                hw_name = ov.Core().get_property(self.device, "FULL_DEVICE_NAME")
            except Exception:
                hw_name = self.device
            log(f"Loaded via openvino_genai on {hw_name} in {time.time() - start:.1f}s")
        except Exception as e:
            log(f"Failed to load model: {e}")
            kind = classify_device_failure(e, self.active_devices())
            if kind is not None:
                # A lost device or broken OpenCL context: do not keep loading
                # in this process, not even on CPU.
                raise DeviceFailureError(kind, e) from e
            log("Falling back to CPU...")
            if self.device != "CPU":
                self.device = "CPU"
                self._load_pipeline()
            else:
                raise

    def transcribe(self, audio_data, sample_rate: int = 16000, language: str = "en",
                   hotwords: str | None = None) -> str:
        """Transcribe audio numpy array to text."""
        import numpy as np
        start = time.time()

        # Ensure float32 normalized to [-1, 1]
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # OpenVINO returns a fresh config per call; never store hints on the pipeline.
        config = self.pipeline.get_generation_config()
        config.max_new_tokens = 448
        if language and language != "auto":
            config.language = f"<|{language}|>"
        config.task = "transcribe"
        config.return_timestamps = False
        if hotwords:
            try:
                config.hotwords = hotwords
            except (AttributeError, TypeError):
                config.initial_prompt = hotwords

        ensure_devices_usable()
        try:
            result = self.pipeline.generate(audio_data, config)
        except Exception as e:
            kind = classify_device_failure(e, self.active_devices())
            if kind is None:
                raise
            log(f"{kind} failure during inference on {self.device}: {e}")
            raise DeviceFailureError(kind, e) from e
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
        # Decoder target; it is attempted on GPU even when device is NPU/CPU.
        self.dec_device = "GPU"
        self.enc_time_dim: dict[int, int] = {}  # bucket frame count -> encoder output time dim
        self._load_pipeline()

    def active_devices(self) -> set:
        """Devices the hybrid pipeline uses (encoder plus decoder)."""
        return {self.device, self.dec_device}

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
                f"Re-run setup: debora-cli --model parakeet --setup"
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
                f"Re-run setup: debora-cli --model parakeet --setup"
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
                f"debora-cli --model parakeet --setup"
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
                f"Re-run setup: debora-cli --model parakeet --setup"
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
                kind = classify_device_failure(first_exc, {self.device})
                if kind is not None:
                    raise DeviceFailureError(kind, first_exc) from first_exc
                fallback = "GPU" if self.device == "NPU" else "CPU"
                log(f"  Encoder bucket {bucket} failed on {self.device}: {first_exc}")
                log(f"  Falling back to {fallback}...")
                try:
                    c = core.compile_model(model_to_compile, fallback, cfg)
                    self.device = fallback
                    log(f"  Encoder bucket {bucket} compiled on {fallback}")
                    return c
                except Exception as fallback_exc:
                    # The fallback device can fail fatally too (e.g. NPU hit an
                    # ordinary compile error, then GPU raised
                    # CL_OUT_OF_RESOURCES). Classify it against the device it
                    # ran on and stop before any further compile: the decoder
                    # would otherwise reuse the same failed GPU context.
                    # Raising here keeps first_exc as __context__.
                    # first_exc was already shown non-fatal above, so a match
                    # here comes from the fallback compile. Blame the fallback
                    # device: the chain text may still carry an "[NPU]" or
                    # ZE_RESULT tag from first_exc, which would otherwise read
                    # as an NPU-only loss and send the GUI back onto this GPU.
                    kind = classify_device_failure(fallback_exc, {fallback})
                    if kind is not None:
                        failed = fallback if fallback != "CPU" else kind
                        raise DeviceFailureError(failed, fallback_exc) from fallback_exc
                    log(f"  Encoder bucket {bucket} failed on {fallback}: {fallback_exc}")
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
                kind = classify_device_failure(e, {"GPU"})
                if kind is not None:
                    raise DeviceFailureError(kind, e) from e
                log(f"  Decoder failed on GPU: {e}, falling back to CPU")
                self.dec_compiled = core.compile_model(decoder_model, "CPU")
                self.dec_device = "CPU"
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
            kind = classify_device_failure(e, self.active_devices())
            if kind is not None and not isinstance(e, DeviceFailureError):
                raise DeviceFailureError(kind, e) from e
            raise

        import openvino as ov
        try:
            core = ov.Core()
            enc_hw = core.get_property(self.device, "FULL_DEVICE_NAME")
        except Exception:
            enc_hw = self.device
            
        dec_dev = getattr(self, "dec_device", "GPU")
        try:
            dec_hw = core.get_property(dec_dev, "FULL_DEVICE_NAME")
        except Exception:
            dec_hw = dec_dev
            
        if enc_hw == dec_hw:
            hw_str = enc_hw
        else:
            hw_str = f"{enc_hw} (Encoder) + {dec_hw} (Decoder)"

        log(f"Parakeet pipeline loaded on {hw_str} in {time.time() - start:.1f}s")

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

        ensure_devices_usable()
        try:
            # 2. Encoder (NPU/GPU) — dispatch to the compiled graph for this bucket
            enc_result = self.enc_compiled[bucket]({
                "audio_signal": mel,
                "length": np.array([actual_frames], dtype=np.int64),
            })
            enc_out = enc_result["outputs"]
            enc_len = int(enc_result["encoded_lengths"][0])

            # 3. TDT Decoder (GPU, or CPU after a load-time fallback)
            text = self._tdt_greedy_decode(enc_out, enc_len)
        except Exception as e:
            kind = classify_device_failure(e, self.active_devices())
            if kind is None:
                raise
            log(f"{kind} failure during Parakeet inference: {e}")
            raise DeviceFailureError(kind, e) from e

        elapsed = time.time() - start
        audio_duration = len(audio_data) / sample_rate
        rtf = elapsed / audio_duration if audio_duration > 0 else 0
        log(f"Transcribed {audio_duration:.1f}s audio in {elapsed:.1f}s "
            f"(RTF: {rtf:.2f}) on {self.device}+CPU [Parakeet, bucket={bucket}f]")

        return text


def create_model(model_path: Path, device: str, backend: str, model_size: str = None):
    """Factory function to create the right model class based on backend.

    Args:
        model_path: Path to model directory (can be None for CUDA).
        device: Device string (NPU, GPU, CPU, CUDA).
        backend: "whisper" or "parakeet" from MODEL_REGISTRY.
        model_size: Size of the model (e.g. "turbo"), required for CUDA.

    Returns:
        WhisperNPU, ParakeetNPU, or FasterWhisperCUDA instance.
    """
    # Covers engines rebuilt by Settings: a new instance in the same process
    # must not load anything after a fatal device failure.
    ensure_devices_usable()
    
    if device == "CUDA":
        return FasterWhisperCUDA(model_size, device="cuda")
        
    if backend == "parakeet":
        return ParakeetNPU(model_path, device=device)
    return WhisperNPU(model_path, device=device)


# ---------------------------------------------------------------------------
# Audio recording
# ---------------------------------------------------------------------------
def _log_audio_stream(sd, stream, direction):
    """Report the actual backend instead of assuming Windows uses WASAPI."""
    device = sd.query_devices(stream.device)
    host = sd.query_hostapis(device['hostapi'])['name']
    log(f"Audio {direction}: device={device['name']}, hostapi={host}, "
        f"sample_rate={stream.samplerate}, latency={stream.latency:.3f}s, "
        f"blocksize={stream.blocksize}")


class NeuralVAD:
    """Silero VAD ONNX wrapper for robust voice activity detection."""
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.session = None
        self.state = None
        
        import numpy as np
        import onnxruntime as ort
        
        model_dir = MODEL_DIR / "silero_vad"
        model_path = model_dir / "silero_vad.onnx"
        
        if not model_path.exists():
            model_dir.mkdir(parents=True, exist_ok=True)
            log("Downloading Silero VAD v5 ONNX model...")
            import urllib.request
            url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
            urllib.request.urlretrieve(url, model_path)
            log("Silero VAD downloaded.")
            
        self.session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        self.reset_state()

    # Silero v5 expects each 512-sample chunk prefixed with the last 64
    # samples of the previous one, as its own OnnxWrapper does. Without them
    # the model scores mic-level speech at ~0.001 (seen in every log line).
    CONTEXT_SAMPLES = 64

    def reset_state(self):
        import numpy as np
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros(self.CONTEXT_SAMPLES, dtype=np.float32)

    def process(self, audio_block):
        """Speech probability of one 512-sample (N,) float32 block."""
        import numpy as np
        block = audio_block.reshape(-1).astype(np.float32)
        input_data = np.concatenate((self.context, block)).reshape(1, -1)
        sr = np.array(self.sample_rate, dtype=np.int64)

        out, state = self.session.run(None, {
            'input': input_data,
            'state': self.state,
            'sr': sr
        })
        self.state = state
        self.context = block[-self.CONTEXT_SAMPLES:]
        return float(out[0][0])


def drop_superseded_drafts(items: list) -> list:
    """Keep only the segments worth transcribing, in order.

    Items start with (audio, is_final), optionally followed by VAD metadata.
    A draft followed by
    any later item is stale: the later draft or final covers newer audio, and
    transcribing it would only delay that newer item. Finals are never
    dropped.
    """
    return [item for idx, item in enumerate(items)
            if item[1] or idx == len(items) - 1]


class AudioRecorder:
    """Record audio from microphone using sounddevice. Supports PTT and VAD."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1,
                 max_record_seconds: float = None, on_timeout=None, config=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_record_seconds = max_record_seconds
        self.on_timeout = on_timeout
        self.config = config or {}
        self.recording = False
        self.continuous = self.config.get("continuous_listening", False)
        
        self._frames = []

        import numpy as np
        import queue
        
        self._stream = None
        self._lock = threading.Lock()
        self._timer = None
        self._recording_generation = 0
        self.telemetry = {}
        self._audio_ready = threading.Event()
        self._last_callback = None
        self._expected_adc_time = None
        self._stable_callbacks = 0
        self.paused = False
        # Voice chat speaking: the VAD ignores the microphone so the reply
        # coming out of the speakers is never taken for the user's speech.
        self.muted = False
        # time.time() of the last block the VAD classified as speech.
        self.last_speech_time = 0.0

        # Continuous VAD properties
        self.capacity = int(sample_rate * self.config.get("ring_buffer_seconds", 30))
        self._buffer = np.zeros((self.capacity, channels), dtype=np.float32)
        self._write_pos = 0
        self._read_pos = 0
        self._lookback_count = 0
        self._data_cv = threading.Condition(self._lock)
        self.segment_queue = queue.Queue(maxsize=10)
        # Set by the consumer while it transcribes. An empty queue is not an
        # idle model: the consumer has already taken the item it is working
        # on, so gating drafts on the queue alone snapshots audio that is
        # 1-2s stale by the time the model gets to it.
        self.consumer_busy = threading.Event()
        self.endpoint = AdaptiveEndpoint()
        self._vad_thread = None
        self._stop_vad = False
        self.neural_vad = None
        
        # VAD thresholds
        # Silero speech probability. Like Silero's own get_speech_timestamps,
        # speech starts above the threshold and only ends below it minus
        # 0.15, so a soft syllable does not count as silence.
        self.vad_threshold = self.config.get("vad_speech_threshold", 0.5)
        self.vad_neg_threshold = max(self.vad_threshold - 0.15, 0.01)
        # RMS threshold, used only when Silero cannot be loaded.
        self.energy_threshold = self.config.get("vad_energy_threshold", 0.005)
        self.min_speech_frames = int(sample_rate * self.config.get("vad_min_speech_seconds", 0.4))
        self.max_segment_frames = int(sample_rate * self.config.get("segment_max_seconds", 15))
        self.lookback_frames = int(sample_rate * self.config.get("vad_lookback_seconds", 0.5))
        self.trailing_frames = int(sample_rate * self.config.get("vad_trailing_seconds", 0.3))

    @property
    def end_silence_frames(self) -> int:
        """Silence that ends a segment; shorter in voice chat, read each time
        so switching modes applies to the next sentence."""
        if self.config.get("voice_chat"):
            seconds = self.config.get("voice_chat_end_silence_seconds", 0.8)
        else:
            seconds = self.config.get("vad_end_silence_seconds", 1.5)
            if self.endpoint.incomplete:
                seconds = max(seconds, self.config.get("vad_incomplete_silence_seconds", 3.0))
        return int(self.sample_rate * seconds)

    def warmup(self, timeout=3.0):
        """Open the stream continuously in the background."""
        import sounddevice as sd
        if self._stream is not None:
            self.wait_ready(timeout)
            return

        def callback(indata, frames, time_info, status):
            now = time.perf_counter()
            adc_time = time_info.inputBufferAdcTime
            delivery_delay = max(0.0, time_info.currentTime - adc_time) if adc_time > 0 else 0.0
            
            with self._lock:
                gap = now - self._last_callback if self._last_callback is not None else 0.0
                adc_gap = (max(0.0, adc_time - self._expected_adc_time)
                           if adc_time > 0 and self._expected_adc_time is not None else 0.0)
                self._last_callback = now
                self._expected_adc_time = adc_time + frames / self.sample_rate if adc_time > 0 else None
                self._stable_callbacks = self._stable_callbacks + 1 if gap < 0.5 and not status else 0
                
                if self._stable_callbacks >= 3:
                    self._audio_ready.set()
                else:
                    self._audio_ready.clear()
                    
                # Always write to ring buffer
                capacity = self.capacity
                count = min(frames, capacity)
                data = indata[-count:]
                first = min(count, capacity - self._write_pos)
                self._buffer[self._write_pos:self._write_pos + first] = data[:first]
                self._buffer[:count - first] = data[first:]
                self._write_pos = (self._write_pos + count) % capacity
                self._lookback_count = min(capacity, self._lookback_count + count)
                
                # The VAD thread, once started, follows the ring buffer even
                # while paused, so a later session starts from fresh audio.
                self._data_cv.notify()

                if self.recording and not self.continuous:
                    self.telemetry.setdefault('first_frame', now)
                    self.telemetry['live_frames'] += frames
                    self.telemetry['input_overflows'] += int(status.input_overflow)
                    self.telemetry['max_callback_gap'] = max(self.telemetry['max_callback_gap'], gap)
                    self.telemetry['max_adc_gap'] = max(self.telemetry['max_adc_gap'], adc_gap)
                    self.telemetry['max_delivery_delay'] = max(self.telemetry['max_delivery_delay'], delivery_delay)
                    self._frames.append(indata.copy())

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=0,
                latency="low",
                callback=callback,
            )
            self._stream.start()
            _log_audio_stream(sd, self._stream, "input")
            self.wait_ready(timeout)
            
            if self.continuous:
                self._ensure_vad_thread()

        except Exception:
            self.close()
            raise

    def prepare_vad(self):
        """Load Silero and start the VAD thread (slow on first use).

        Paused unless already continuous: during push-to-talk the thread must
        not cut segments, or they would be typed and then typed again from
        begin_continuous()'s rewind."""
        with self._lock:
            if not self.continuous:
                self.paused = True
        self._ensure_vad_thread()

    def _ensure_vad_thread(self):
        if self.neural_vad is None:
            try:
                self.neural_vad = NeuralVAD(sample_rate=self.sample_rate)
            except Exception as e:
                log(f"Failed to initialize Neural VAD: {e}")
        if self._vad_thread is None or not self._vad_thread.is_alive():
            self._stop_vad = False
            self._vad_thread = threading.Thread(target=self._vad_loop, daemon=True)
            self._vad_thread.start()

    def begin_continuous(self, rewind_seconds: float = 0.0):
        """Turn the push-to-talk recording into continuous VAD listening.

        The push-to-talk audio is dropped; the VAD re-reads the last
        rewind_seconds of the ring buffer instead, so speech that began
        while the hotkey was tapped is not lost.
        """
        with self._lock:
            self.recording = False
            self._frames = []
            if self._timer:
                self._timer.cancel()
                self._timer = None
            rewind = min(int(rewind_seconds * self.sample_rate), self.capacity - 1)
            self._read_pos = (self._write_pos - rewind) % self.capacity
            if self.neural_vad:
                self.neural_vad.reset_state()
            self.continuous = True
            self.paused = False
        self._ensure_vad_thread()

    def end_continuous(self):
        """Stop producing segments. Speech in progress is cut and queued."""
        with self._lock:
            self.paused = True
            self.continuous = False
            self.endpoint.finish()

    def set_muted(self, muted: bool):
        """Mute or unmute the VAD. Unmuting skips everything heard while
        muted, so the reply's tail is not cut as a segment."""
        with self._lock:
            self.muted = muted
            self.endpoint.finish()
            if not muted:
                self._read_pos = self._write_pos
                if self.neural_vad:
                    self.neural_vad.reset_state()

    def wait_ready(self, timeout=3.0):
        if not self._audio_ready.wait(timeout):
            raise RuntimeError("Microphone did not deliver stable audio callbacks during warmup")
        with self._lock:
            if (self._stream is None or not self._stream.active or
                    self._last_callback is None or time.perf_counter() - self._last_callback > 0.5):
                raise RuntimeError("Microphone audio stream is inactive or stalled")

    def _block_is_speech(self, block, is_speaking: bool) -> bool:
        """Classify one 512-sample block; Silero when loaded, else RMS."""
        import numpy as np
        if not self.neural_vad:
            rms = float(np.sqrt(np.mean(block ** 2)))
            return rms > self.energy_threshold

        # DC blocker (zero mean) for mics with an offset.
        samples = block.flatten()
        samples = samples - np.mean(samples)
        prob = self.neural_vad.process(samples)

        # Peak probability and amplitude every 2 s, to tune mics.
        now = time.time()
        if not hasattr(self, "_last_prob_log"):
            self._max_prob, self._max_amp, self._last_prob_log = 0.0, 0.0, now
        self._max_prob = max(self._max_prob, prob)
        self._max_amp = max(self._max_amp, float(np.max(np.abs(samples))))
        if now - self._last_prob_log > 2.0:
            if self._max_prob > 0.01 or self._max_amp > 0.005:
                log(f"[VAD Debug] Max prob: {self._max_prob:.3f}, Max amp: {self._max_amp:.4f}")
            self._max_prob, self._max_amp, self._last_prob_log = 0.0, 0.0, now

        return prob > (self.vad_neg_threshold if is_speaking else self.vad_threshold)

    def _vad_loop(self):
        import numpy as np
        is_speaking = False
        speech_start_pos = 0
        silence_frames = 0
        speech_frames = 0
        last_draft_time = 0.0
        segment_id = 0
        extension_logged = False
        
        log("VAD thread started.")
        try:
            while not self._stop_vad:
                with self._lock:
                    self._data_cv.wait(timeout=0.1)
                    if self._write_pos >= self._read_pos:
                        available = self._write_pos - self._read_pos
                    else:
                        available = self.capacity - self._read_pos + self._write_pos
                        
                    if available == 0:
                        continue
                        
                    if self._write_pos > self._read_pos:
                        new_data = self._buffer[self._read_pos:self._write_pos].copy()
                    else:
                        new_data = np.concatenate((
                            self._buffer[self._read_pos:], 
                            self._buffer[:self._write_pos]
                        ))
                    
                    start_read_pos = self._read_pos
                    self._read_pos = self._write_pos
                    
                # Silero VAD prefers 512 frames for 16kHz
                block_size = 512
                for i in range(0, len(new_data), block_size):
                    block = new_data[i:i+block_size]
                    block_len = len(block)
                    if len(block) < block_size:
                        # Put back leftover frames by winding back read_pos slightly
                        # In practice, we could just ignore or buffer it, but it's easier to just rewind
                        with self._lock:
                            leftover = len(block)
                            self._read_pos = (self._read_pos - leftover) % self.capacity
                        break
                        
                    is_paused = getattr(self, "paused", False)
                    cut_segment = False

                    if self.muted:
                        # Speech in progress is dropped, not cut: it may be
                        # the reply's own audio.
                        is_speaking = False
                        self.endpoint.finish()
                        continue
                    
                    if is_paused:
                        if is_speaking:
                            if speech_frames >= self.min_speech_frames:
                                cut_segment = True
                                log("VAD: Cutting segment due to pause.")
                            else:
                                is_speaking = False
                                self.endpoint.finish()
                                if self.neural_vad: self.neural_vad.reset_state()
                        else:
                            continue
                    else:
                        # Use Neural VAD if available, fallback to RMS
                        is_speech_now = self._block_is_speech(block, is_speaking)
                        if is_speech_now:
                            self.last_speech_time = time.time()

                        if is_speech_now:
                            if not is_speaking:
                                is_speaking = True
                                segment_id = self.endpoint.start()
                                speech_start_pos = (start_read_pos + i - self.lookback_frames) % self.capacity
                                silence_frames = 0
                                speech_frames = self.lookback_frames + block_len
                                last_draft_time = time.time()
                            else:
                                silence_frames = 0
                                speech_frames += block_len
                            self.endpoint.speech(speech_frames)
                            extension_logged = False
                        else:
                            if is_speaking:
                                silence_frames += block_len
                                speech_frames += block_len
                                
                        # Draft logic
                        if is_speaking and not cut_segment:
                            current_time = time.time()
                            if current_time - last_draft_time > 1.0:
                                if (self.segment_queue.empty()
                                        and not self.consumer_busy.is_set()
                                        and speech_frames > self.min_speech_frames):
                                    # Model is idle: snapshot the audio now so
                                    # the draft is as fresh as possible.
                                    with self._lock:
                                        draft_end_pos = (start_read_pos + i + block_len) % self.capacity
                                        if draft_end_pos > speech_start_pos:
                                            draft_audio = self._buffer[speech_start_pos:draft_end_pos].copy()
                                        else:
                                            draft_audio = np.concatenate((
                                                self._buffer[speech_start_pos:],
                                                self._buffer[:draft_end_pos]
                                            ))
                                    self.segment_queue.put(VadSegment(
                                        draft_audio.flatten(), False, segment_id, speech_frames))
                                    last_draft_time = current_time

                        # End of speech conditions
                        silence_limit = self.end_silence_frames
                        base_limit = int(self.sample_rate * self.config.get("vad_end_silence_seconds", 1.5))
                        if (is_speaking and not extension_logged
                                and not self.config.get("voice_chat")
                                and silence_limit > base_limit and silence_frames > base_limit):
                            log(f"VAD: Incomplete draft; allowing {silence_limit / self.sample_rate:g}s "
                                "of total silence.")
                            extension_logged = True
                        if is_speaking and silence_frames > silence_limit:
                            if speech_frames >= self.min_speech_frames:
                                cut_segment = True
                                log("VAD: Cutting segment due to natural silence.")
                            else:
                                # Too short, discard
                                is_speaking = False
                                self.endpoint.finish()
                                if self.neural_vad: self.neural_vad.reset_state()
                                
                        elif is_speaking and speech_frames >= self.max_segment_frames:
                            # Forced cut
                            cut_segment = True
                            log("VAD: Cutting segment due to max duration.")
                            
                    if cut_segment:
                        is_speaking = False
                        self.endpoint.finish()
                        if self.neural_vad: self.neural_vad.reset_state()
                        # Extract segment
                        with self._lock:
                            # Cap the trailing silence to prevent Whisper from dropping words or hallucinating spaces.
                            # We keep up to 1.0s of the silence (which may contain quiet speech) plus trailing_frames.
                            max_silence_keep = int(self.sample_rate * 1.0)
                            extract_frames = speech_frames
                            if silence_frames > max_silence_keep:
                                extract_frames -= (silence_frames - max_silence_keep)
                            extract_frames += self.trailing_frames
                            end_pos = (speech_start_pos + extract_frames) % self.capacity
                            if end_pos > speech_start_pos:
                                audio = self._buffer[speech_start_pos:end_pos].copy()
                            else:
                                audio = np.concatenate((
                                    self._buffer[speech_start_pos:],
                                    self._buffer[:end_pos]
                                ))
                            
                        self.segment_queue.put(VadSegment(
                            audio.flatten(), True, segment_id, speech_frames))
        except Exception as e:
            import traceback
            log(f"CRITICAL ERROR in VAD loop: {e}\n{traceback.format_exc()}")


    def start(self):
        """Start recording (PTT mode)."""
        if self.continuous:
            return # VAD handles recording
            
        with self._lock:
            if self.recording:
                return
            self.telemetry = dict(start_called=time.perf_counter(), live_frames=0,
                                  input_overflows=0, max_callback_gap=0.0,
                                  max_adc_gap=0.0, max_delivery_delay=0.0)
            
            # Extract lookback from continuous buffer
            count = min(self._lookback_count, int(self.sample_rate * 1.5))
            begin = (self._write_pos - count) % self.capacity
            first = min(count, self.capacity - begin)
            self._frames = []
            if first:
                self._frames.append(self._buffer[begin:begin + first].copy())
            if count > first:
                self._frames.append(self._buffer[:count - first].copy())
                
            self._lookback_count = 0
            self.recording = True
            self._recording_generation += 1
            if self.max_record_seconds:
                self._timer = threading.Timer(
                    self.max_record_seconds, self._timeout_stop,
                    args=(self._recording_generation,),
                )
                self._timer.daemon = True
                self._timer.start()

        log("Recording started...")

    def _timeout_stop(self, generation):
        with self._lock:
            if not self.recording or generation != self._recording_generation:
                return
            self.recording = False
            self._timer = None
        log(f"Max recording time ({self.max_record_seconds}s) reached, stopping.")
        if self.on_timeout is not None:
            self.on_timeout(generation)

    def stop(self):
        """Stop recording and return audio as numpy array (PTT mode)."""
        import numpy as np

        if self.continuous:
            return np.array([], dtype=np.float32)

        with self._lock:
            self.recording = False
            frames = self._frames
            self._frames = []
            telemetry = dict(self.telemetry)
            if self._timer:
                self._timer.cancel()
                self._timer = None

        if not frames:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(frames, axis=0).flatten()
        duration = len(audio) / self.sample_rate
        log(f"Recording stopped. Duration: {duration:.1f}s")
        return audio

    def close(self):
        with self._lock:
            self.recording = False
            self._stop_vad = True
            self.endpoint.finish()
            self._data_cv.notify_all()
            if self._timer:
                self._timer.cancel()
                self._timer = None
        stream, self._stream = self._stream, None
        try:
            if stream is not None:
                try:
                    stream.stop()
                finally:
                    stream.close()
        finally:
            with self._lock:
                self.recording = False
                self._frames = []
                self._lookback_count = 0
                self._last_callback = None
                self._expected_adc_time = None
                self._stable_callbacks = 0
                self._audio_ready.clear()

    @property
    def audio_level(self) -> float:
        import numpy as np
        with self._lock:
            if self.continuous:
                if self._lookback_count == 0:
                    return 0.0
                last_frame = self._buffer[(self._write_pos - 1024) % self.capacity : self._write_pos].copy()
            else:
                if not self._frames or not self.recording:
                    return 0.0
                last_frame = self._frames[-1].copy()
                
        if len(last_frame) == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(last_frame ** 2)))
        return min(rms * 25.0, 1.0)


def get_input_target():
    """Where typed keys land: (foreground window, focused control), or None.

    Comparing the pair, not just the window, also catches focus moving to
    another field of the same window in native apps. Browsers and Electron
    apps report one handle per window, so a click elsewhere on the same page
    goes unnoticed.
    """
    try:
        import ctypes
        from ctypes import wintypes

        class GUITHREADINFO(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("hwndActive", wintypes.HWND),
                ("hwndFocus", wintypes.HWND),
                ("hwndCapture", wintypes.HWND),
                ("hwndMenuOwner", wintypes.HWND),
                ("hwndMoveSize", wintypes.HWND),
                ("hwndCaret", wintypes.HWND),
                ("rcCaret", wintypes.RECT),
            ]

        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None
        info = GUITHREADINFO(cbSize=ctypes.sizeof(GUITHREADINFO))
        thread_id = user32.GetWindowThreadProcessId(hwnd, None)
        focus = info.hwndFocus if user32.GetGUIThreadInfo(thread_id, ctypes.byref(info)) else None
        return (hwnd, focus)
    except Exception:
        return None


def same_input_target(a, b) -> bool:
    """Whether keys still land where they landed before.

    GetGUIThreadInfo briefly reports no focused control (seen in about one
    poll in 300), so a missing focus on either side only compares windows.
    """
    if a is None or b is None or a[0] != b[0]:
        return False
    return a[1] is None or b[1] is None or a[1] == b[1]


def type_draft_text(text: str):
    """Paste live drafts through the same serialized path as final text."""
    type_text(text)


def delete_text(count: int):
    """Delete the specified number of characters using Shift+Left selection."""
    if count <= 0:
        return
    try:
        import keyboard
        keyboard.press("shift")
        for _ in range(count):
            keyboard.send("left")
            time.sleep(0.001)
        keyboard.release("shift")
        keyboard.send("backspace")
    except ImportError:
        import ctypes
        user32 = ctypes.windll.user32
        KEYEVENTF_KEYUP = 0x0002
        VK_BACK = 0x08
        VK_SHIFT = 0x10
        VK_LEFT = 0x25
        
        # Press Shift
        user32.keybd_event(VK_SHIFT, 0, 0, 0)
        # Press Left Arrow `count` times
        for _ in range(count):
            user32.keybd_event(VK_LEFT, 0, 0, 0)
            user32.keybd_event(VK_LEFT, 0, KEYEVENTF_KEYUP, 0)
            time.sleep(0.001)
        # Release Shift
        user32.keybd_event(VK_SHIFT, 0, KEYEVENTF_KEYUP, 0)
        # Press Backspace
        user32.keybd_event(VK_BACK, 0, 0, 0)
        user32.keybd_event(VK_BACK, 0, KEYEVENTF_KEYUP, 0)


_clipboard_lock = threading.Lock()


def type_text(text: str, auto_enter: bool = False):
    """Type text into the currently active window using keyboard simulation."""
    if not text:
        return

    try:
        # Try pyperclip + keyboard for reliable pasting
        import pyperclip
        import keyboard

        # Complete restoration before another draft/final uses the clipboard.
        # A detached restore can otherwise overwrite the next text before
        # its Ctrl+V is processed. Unicode key packets are only a fallback:
        # live text should reach the target as one paste, not individual keys.
        with _clipboard_lock:
            try:
                old_clipboard = pyperclip.paste()
            except Exception:
                old_clipboard = None

            pyperclip.copy(text)
            try:
                time.sleep(0.05)
                keyboard.press_and_release("ctrl+v")
                time.sleep(0.1)
                if auto_enter:
                    keyboard.press_and_release("enter")
            finally:
                time.sleep(0.5)
                try:
                    # Don't overwrite something the user copied meanwhile.
                    if old_clipboard is not None and pyperclip.paste() == text:
                        pyperclip.copy(old_clipboard)
                except Exception:
                    pass

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

    # KEYEVENTF_UNICODE takes UTF-16 code units; characters outside the
    # BMP (e.g. emoji) are sent as their surrogate pair.
    data = text.encode("utf-16-le")
    for unit in (int.from_bytes(data[i:i + 2], "little") for i in range(0, len(data), 2)):
        inputs = (INPUT * 2)()
        # Key down
        inputs[0].type = INPUT_KEYBOARD
        inputs[0]._input.ki.wScan = unit
        inputs[0]._input.ki.dwFlags = KEYEVENTF_UNICODE
        # Key up
        inputs[1].type = INPUT_KEYBOARD
        inputs[1]._input.ki.wScan = unit
        inputs[1]._input.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP

        user32.SendInput(2, ctypes.byref(inputs), ctypes.sizeof(INPUT))
        time.sleep(0.002)


# ---------------------------------------------------------------------------
# Audio feedback (chimes)
# ---------------------------------------------------------------------------
def _generate_tone_array(frequencies: list[float], duration_ms: int = 120,
                         sample_rate: int = 44100, volume: float = 0.1):
    """Generate a smooth audio array in memory for playback."""
    import numpy as np

    n_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples, endpoint=False)

    signal = np.zeros(n_samples, dtype=np.float32)
    for freq in frequencies:
        signal += np.sin(2 * np.pi * freq * t)
    if frequencies:
        signal /= len(frequencies)

    # A raised-cosine envelope fades smoothly throughout the short blip,
    # with zero amplitude and slope at both ends to avoid sharp clicks.
    envelope = np.hanning(n_samples).astype(np.float32)

    signal = signal * envelope * volume
    return signal


class ChimePlayer:
    """Keep one output stream running; hotkeys only submit precomputed tones."""

    def __init__(self):
        self._control_lock = threading.Lock()
        self._stream = None
        self._tones = {}
        self._pending = deque(maxlen=1)
        self._tone = None
        self._position = 0
        self._ready = threading.Event()
        self.output_underflows = 0

    def warmup(self, timeout=3.0):
        import sounddevice as sd
        import numpy as np
        if self._stream is not None:
            return
        try:
            device = sd.query_devices(kind="output")
            sample_rate = device['default_samplerate']
            def tone(frequencies, duration, volume):
                return _generate_tone_array(frequencies, duration, sample_rate, volume)
            warning = tone([277.18], 90, 0.07)
            self._tones = {
                'start': tone([440.0], 130, 0.10),
                'stop': tone([330.0], 160, 0.08),
                # Rising pair: continuous listening is on (the tap's own
                # 'start' already played; this tells the modes apart).
                'continuous': np.concatenate([tone([440.0], 90, 0.09),
                                              np.zeros(int(sample_rate * 0.04), dtype=np.float32),
                                              tone([660.0], 110, 0.09)]),
                'warning': np.concatenate([warning, np.zeros(int(sample_rate * 0.05), dtype=np.float32), warning]),
            }
            self._stream = sd.OutputStream(
                samplerate=sample_rate, channels=min(2, device['max_output_channels']),
                dtype="float32", blocksize=0, latency="low", callback=self._callback,
            )
            self._stream.start()
            _log_audio_stream(sd, self._stream, "output")
            if not self._ready.wait(timeout):
                raise RuntimeError("Output stream did not deliver audio callbacks during warmup")
        except Exception as exc:
            log(f"Audio feedback unavailable; continuing without chimes: {exc}")
            try:
                self.close()
            except Exception as cleanup_error:
                log(f"Error closing unavailable audio output: {cleanup_error}")

    def _callback(self, outdata, frames, time_info, status):
        outdata.fill(0)
        self.output_underflows += int(status.output_underflow)
        self._ready.set()
        try:
            # deque append/popleft are thread-safe; keep only the newest request.
            self._tone = self._pending.popleft()
            self._position = 0
        except IndexError:
            pass
        if self._tone is not None:
            count = min(frames, len(self._tone) - self._position)
            outdata[:count] = self._tone[self._position:self._position + count, None]
            self._position += count
            if self._position == len(self._tone):
                self._tone = None

    def play(self, name):
        with self._control_lock:
            if self._ready.is_set() and self._stream is not None and self._stream.active:
                self._pending.append(self._tones[name])

    def close(self):
        with self._control_lock:
            self._ready.clear()
            stream, self._stream = self._stream, None
            try:
                if stream is not None:
                    try:
                        stream.stop()
                    finally:
                        stream.close()
                    log(f"[Telemetry] Output underflows: {self.output_underflows}")
            finally:
                self._ready.clear()
                self._pending.clear()
                self._tone = None
                self._tones = {}


# ---------------------------------------------------------------------------
# Main application loop
# ---------------------------------------------------------------------------
class DictationApp:
    """Main dictation application with hotkey toggle."""

    MAX_HISTORY = 20
    # Upper bound for a recording to wait on a model (re)load before it is
    # discarded with an error, instead of blocking its thread forever.
    MODEL_WAIT_SECONDS = 120

    def __init__(self, config: dict):
        self.config = config
        self.recorder = AudioRecorder(
            sample_rate=config["sample_rate"],
            max_record_seconds=config.get("max_record_seconds"),
            on_timeout=self._finish_recording,
            config=config,
        )
        self.chimes = ChimePlayer()
        self._audio_lifecycle_lock = threading.Lock()
        self._stopping = threading.Event()
        # Orders the final "still running?" check plus paste/history against
        # stop(). Only type_text and the history append run under it, never a
        # state callback, so a callback that calls stop() cannot deadlock.
        self._output_lock = threading.Lock()
        self._transcribing = False
        # Number of model load/warmup reservations in flight (each compiles
        # and infers on the accelerator). A counter, not a flag, so an older
        # loader finishing can never clear a newer loader's busy state.
        # Guarded by _audio_lifecycle_lock.
        self._loading = 0
        self.whisper = None  # Lazy-loaded
        self.is_recording = False
        # Continuous (VAD) listening is on: from --continuous at startup or a
        # hotkey tap. Guarded by _audio_lifecycle_lock.
        self._continuous = bool(config.get("continuous_listening", False))
        self._recording_claude_chat = False
        self._memory_hotwords = MemoryHotwords()
        self._hotword_terms = ()
        self._hotwords_unsupported = set()
        self._continuous_since = 0.0
        self._model_ready = threading.Event()
        self._load_error: str | None = None
        # Serializes model use (warmup/transcribe) against model replacement
        # (fallback, quarantine, NPU recovery hot-swap). Reentrant because
        # _error_payload() may quarantine from a thread already holding it.
        self._model_lock = threading.RLock()
        # Models whose accelerator reported DEVICE_LOST. They are never called
        # again and deliberately never released: calling into a pipeline bound
        # to a lost device crashed the process natively (0xc0000005), and
        # freeing it runs the same driver code. Memory returns on restart.
        self._quarantined_models: list = []
        # Configured device whose model was lost and not yet replaced by a
        # fallback or recovery. ensure_model() refuses to load on it.
        self._lost_device: str | None = None
        # True while the loader's 2 s microphone warmup owns the recorder.
        self._mic_warmup = False

        # State machine & callbacks (used by GUI, harmless in CLI mode)
        self._state = AppState.LOADING
        self._callbacks: list = []
        self._history: list[dict] = []
        # Live draft already typed into the target, guarded by _output_lock.
        self._draft_typed_text = ""
        self._draft_target = None
        self.voice_chat = VoiceChat(config, log=log, tts_log_path=TTS_SERVER_LOG,
                                    llm_log_path=LLM_SERVER_LOG)
        self._voice_chat_loading = threading.Lock()

        self._resource_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self._resource_thread.start()

    def _monitor_resources(self):
        """Continuously log CPU, RAM, and VRAM usage to telemetry."""
        try:
            import psutil
            import sys
            process = psutil.Process()
            while True:
                cpu = psutil.cpu_percent(interval=5.0)
                mem = process.memory_info().rss / (1024 * 1024)
                sys_mem = psutil.virtual_memory().percent
                
                vram_info = ""
                if self.config.get("device") == "CUDA":
                    try:
                        import subprocess
                        output = subprocess.check_output(
                            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
                            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                        ).decode("utf-8").strip()
                        used, total = output.split(", ")
                        vram_info = f" | VRAM: {used}/{total}MB"
                    except Exception:
                        pass
                
                state_name = self._state.value if hasattr(self, "_state") else "UNKNOWN"
                log(f"[Telemetry] State: {state_name} | CPU: {cpu:02.1f}% | App RAM: {mem:.0f}MB | Sys RAM: {sys_mem:02.1f}%{vram_info}")
        except ImportError:
            log("[Telemetry] psutil not found. Resource monitoring disabled.")
        except Exception as e:
            log(f"[Telemetry] Resource monitoring stopped: {e}")

    # -- Callback system -----------------------------------------------

    def add_callback(self, fn):
        """Register a callback ``fn(state: AppState, data: dict)``."""
        self._callbacks.append(fn)

    def _set_state(self, state: AppState, data: dict | None = None):
        """Update state and notify all callbacks."""
        if self._stopping.is_set():
            return
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
        with self._model_lock:
            if self.whisper is None:
                if self._lost_device and self.config["device"] == self._lost_device:
                    raise RuntimeError(
                        f"{self._lost_device} was lost; not reloading on it until "
                        f"a fallback device is selected")
                model_path = setup_model(self.config)
                model_info = MODEL_REGISTRY[self.config["model_size"]]
                self.whisper = create_model(
                    model_path, device=self.config["device"],
                    backend=model_info["backend"],
                    model_size=self.config["model_size"]
                )

    def _quarantine_lost_model(self):
        """Take the model bound to a lost device out of service for good."""
        with self._model_lock:
            self._lost_device = self.config["device"]
            if self.whisper is not None:
                self._quarantined_models.append(self.whisper)
                self.whisper = None
                log(f"Quarantined model on lost {self._lost_device}: it will not "
                    f"be called or released again in this process.")

    def _error_payload(self, exc: BaseException) -> dict:
        """Classify an engine error and build the ERROR state payload.

        A GPU (or unattributable) device failure is latched for the whole
        process: this engine and any engine created later refuse to load or
        infer until the app is restarted. An NPU-only loss keeps the existing
        ``device_lost`` payload so the GUI may move to a healthy GPU.
        """
        if isinstance(exc, RestartRequiredError):
            kind = "LATCHED"
        else:
            kind = classify_device_failure(exc, _model_active_devices(self.whisper))
        if kind is None:
            return {"error": str(exc)}
        cause = f"{type(exc).__name__}: {exc}"
        if kind == "LATCHED":
            failure = device_failure()
            self._load_error = failure["message"]
            self._model_ready.set()
            return {
                "error": failure["message"],
                "restart_required": True,
                "device_failure": failure["device"],
                "cause": cause,
            }

        if kind == "NPU":
            remember_npu_loss(_failure_detail(exc))
        # Never call into the failed pipeline again: a second generate() on
        # the lost NPU crashed the whole process (access violation).
        self._quarantine_lost_model()
        return {"error": str(exc), "device_lost": True,
                "device_failure": kind, "cause": cause}

    def _start_loader(self):
        """Spawn the load thread, marking the engine busy before it starts so
        stop_if_idle() can never miss a load that is about to begin.

        The reservation taken here is released by the new thread only after
        _load_model_background returns, so it covers the whole load even when
        a caller (e.g. a synchronous ERROR callback that falls back to GPU)
        starts the next loader before the previous one has returned.
        """
        with self._audio_lifecycle_lock:
            if self._stopping.is_set():
                return
            self._loading += 1

        def run():
            try:
                self._load_model_background()
            finally:
                self._release_loading()

        try:
            threading.Thread(target=run, daemon=True).start()
        except BaseException:
            # The thread never ran, so its finally never will: undo here.
            self._release_loading()
            raise

    def _release_loading(self):
        with self._audio_lifecycle_lock:
            self._loading -= 1

    def _load_model_background(self):
        """Load model in background thread, setting _model_ready when done.

        Holds its own loading reservation, so direct callers (CLI ``run()``,
        tests) are also reported busy while it runs.
        """
        with self._audio_lifecycle_lock:
            self._loading += 1
        try:
            self._load_model_background_inner()
        finally:
            self._release_loading()

    def _load_model_background_inner(self):
        if self._stopping.is_set():
            return
        if device_failure():
            self._set_state(AppState.ERROR, self._error_payload(
                RestartRequiredError(device_failure()["message"])))
            return
        self._set_state(AppState.LOADING)
        try:
            log("Initializing audio subsystem...")
            with self._audio_lifecycle_lock:
                if self._stopping.is_set():
                    return
                # Open output first so device initialization finishes before
                # capture readiness is checked. Neither stream reopens on a hotkey.
                if self.config["beep_on_start"]:
                    self.chimes.warmup()
                if self._continuous:
                    self._recording_claude_chat = self._claude_voice_chat()
                self.recorder.warmup()

            self.ensure_model()

            # ------------------------------------------------------------------
            # END-TO-END WARMUP (As suggested by the user)
            # Real audio capture to force Windows Audio Engine, drivers, and Whisper
            # to fully initialize and flush any initial delays/buffers.
            # ------------------------------------------------------------------
            with self._audio_lifecycle_lock:
                if self._stopping.is_set():
                    return
                # Recording is allowed while a (fallback) model loads; never
                # hijack the user's recording for the warmup capture.
                capture_warmup = not self.is_recording and not self._continuous
                if capture_warmup:
                    self.recorder.wait_ready()
                    log("Performing real 2-second microphone capture to warm up hardware...")
                    self._mic_warmup = True
                    self.recorder.start()
            if capture_warmup:
                try:
                    if self._stopping.wait(2.0):
                        return
                    with self._audio_lifecycle_lock:
                        if self._stopping.is_set():
                            return
                        warmup_audio = self.recorder.stop()
                        if not self.recorder.telemetry.get('live_frames', 0):
                            raise RuntimeError("Microphone warmup captured no new audio frames")
                finally:
                    self._mic_warmup = False
            else:
                import numpy as np
                log("Recording in progress; warming up inference on silence instead.")
                warmup_audio = np.zeros(int(self.config["sample_rate"]), dtype=np.float32)

            if len(warmup_audio) > 0:
                log("Hardware warmup capture successful. Pre-warming Whisper inference...")
                with self._model_lock:
                    self.ensure_model()
                    _ = self.whisper.transcribe(
                        warmup_audio,
                        sample_rate=self.config["sample_rate"],
                        language=self.config["language"]
                    )
                log("End-to-end warmup complete.")
            else:
                raise RuntimeError("Hardware warmup capture failed (empty buffer)")
            # ------------------------------------------------------------------

            with self._audio_lifecycle_lock:
                if self._stopping.is_set():
                    return
                self.recorder.wait_ready()
                self._model_ready.set()
            # Notify outside the lock: a READY callback may call stop(),
            # busy_reason() or stop_if_idle() (or wait on a thread that does),
            # all of which take _audio_lifecycle_lock. _set_state re-checks
            # _stopping, so a stop that lands here suppresses READY.
            if self._continuous:
                self.is_recording = True
                self._set_state(AppState.RECORDING)
                log("Continuous listening mode active. VAD will process speech.")
            else:
                self._set_state(AppState.READY)
                log("Ready! Waiting for hotkey...")
        except Exception as e:
            import traceback
            log(f"Model loading failed: {e}")
            log(traceback.format_exc())
            # Classify (and latch) even during shutdown: the failed device
            # must stay off-limits for any engine created afterwards.
            payload = self._error_payload(e)
            if self._stopping.is_set():
                return
            if not payload.get("restart_required"):
                self._load_error = str(e)
            self._model_ready.set()  # Unblock waiters so they can see the error
            self._set_state(AppState.ERROR, payload)

    def fallback_device(self, new_device: str):
        """Clear error state and reload model on a different device."""
        failure = device_failure()
        if failure:
            log(f"Refusing to reload on {new_device}: {failure['message']}")
            self._set_state(AppState.ERROR, self._error_payload(
                RestartRequiredError(failure["message"])))
            return
        log(f"Falling back to device: {new_device}")
        with self._model_lock:
            self._load_error = None
            self._model_ready.clear()
            self.whisper = None  # a lost model is already quarantined
            self.config["device"] = new_device
            if new_device != self._lost_device:
                self._lost_device = None
        self._start_loader()

    def inject_recovered_model(self, new_device: str, new_whisper) -> bool:
        """Hot-swap the active model with a recovered one.

        Returns False (nothing changed) while a recording, transcription or
        model load is in flight; the caller may retry later. The replaced
        model is a healthy fallback and is released normally.

        In continuous mode the engine records for its whole lifetime, so an
        open microphone does not block the swap: segments only reach the
        model through _finish_recording, which sets _transcribing and takes
        _model_lock like the swap does.
        """
        with self._audio_lifecycle_lock:
            busy = self._busy_reason_locked()
            if busy == "recording" and self._continuous:
                # Recording is reported first; look past it.
                busy = ("transcription" if self._transcribing
                        else "model loading" if self._loading else None)
            if busy or self._stopping.is_set():
                log(f"Deferring swap to {new_device}: {busy or 'engine stopping'}")
                return False
        with self._model_lock:
            log(f"Seamlessly swapping active engine to {new_device}")
            old = self.whisper
            self.config["device"] = new_device
            self.whisper = new_whisper
            self._lost_device = None
            self._load_error = None
            self._model_ready.set()
        del old
        import gc
        gc.collect()
        # Notify outside every lock (see _load_model_background_inner).
        if self._state in (AppState.ERROR, AppState.LOADING):
            self._set_state(AppState.READY)
        return True

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

    def _claude_voice_chat(self) -> bool:
        return bool(self.config.get("voice_chat")
                    and self.config.get("voice_chat_backend", "local") == "claude")

    def _transcription_hotwords(self) -> dict:
        terms = ()
        if (self._recording_claude_chat and self._claude_voice_chat()
                and self.config.get("harness_hotwords", True)):
            unsupported = (isinstance(self.whisper, ParakeetNPU)
                           or (isinstance(self.whisper, FasterWhisperCUDA)
                               and not self.whisper.supports_hotwords()))
            if unsupported:
                backend = type(self.whisper).__name__
                if backend not in self._hotwords_unsupported:
                    log(f"Voice chat: {backend} has no hotword support; skipping memory hints")
                    self._hotwords_unsupported.add(backend)
            else:
                terms = self._memory_hotwords.terms(self.config, log)
        if terms != self._hotword_terms:
            log(f"Voice chat: using {len(terms)} memory hint terms")
            self._hotword_terms = terms
        return {"hotwords": ", ".join(terms)} if terms else {}

    def _forget_draft_locked(self, erase: bool):
        """Stop tracking the typed draft, first erasing it if asked and keys
        still land where it was typed. Caller holds _output_lock."""
        if erase and self._draft_typed_text:
            if same_input_target(get_input_target(), self._draft_target):
                delete_text(len(self._draft_typed_text))
            else:
                log("Input focus changed; leaving the draft where it was typed.")
        self._draft_typed_text = ""
        self._draft_target = None

    def _finish_recording(self, generation=None, audio=None, is_final=True,
                          segment_id=None, audio_end=None):
        """Consume one recording, whether stopped by the user or its timer,
        or (audio given) one VAD segment of continuous listening."""
        vad_segment = audio is not None
        with self._audio_lifecycle_lock:
            if self._stopping.is_set() or self._transcribing:
                return
            if audio is None:
                if not self.is_recording or (generation is not None and generation != self.recorder._recording_generation):
                    return
                self.is_recording = False
            self._transcribing = True
        try:
            if audio is None:
                audio = self.recorder.stop()
            if self._stopping.is_set():
                return
            if self.config["beep_on_start"] and is_final:
                self.chimes.play('stop')

            if len(audio) < self.config["sample_rate"] * 0.3:
                log("Recording too short, ignoring.")
                if is_final:
                    if self._continuous and self.is_recording:
                        self._set_state(AppState.RECORDING)
                    else:
                        self._set_state(AppState.READY)
                return

            # Continuous mode transcribes all the time while the microphone
            # stays open: keep showing RECORDING (live waveform and draft)
            # instead of a "Transcribing..." balloon per sentence.
            continuous = self._continuous and self.is_recording
            if is_final and not continuous:
                self._set_state(AppState.PROCESSING)

            try:
                if not self._model_ready.is_set():
                    log("Waiting for model to be ready before transcription...")
                if not self._model_ready.wait(self.MODEL_WAIT_SECONDS):
                    raise RuntimeError(
                        f"Model not ready after {self.MODEL_WAIT_SECONDS}s; "
                        f"recording discarded")
                ensure_devices_usable()
                if not _inference_gate.acquire(blocking=False):
                    log("Waiting for the LLM to load before transcribing...")
                    _inference_gate.acquire()
                try:
                    with self._model_lock:
                        self.ensure_model()
                        hints = self._transcription_hotwords()
                        text = self.whisper.transcribe(
                            audio,
                            sample_rate=self.config["sample_rate"],
                            language=self.config["language"],
                            **hints,
                        )
                        if hints and not self._claude_voice_chat():
                            # A toggle during inference must not send hinted text
                            # to dictation or Qwen. Decode again without hints.
                            self._hotword_terms = ()
                            log("Voice chat: using 0 memory hint terms (mode changed during inference)")
                            text = self.whisper.transcribe(
                                audio,
                                sample_rate=self.config["sample_rate"],
                                language=self.config["language"],
                            )
                finally:
                    _inference_gate.release()
                t_lower = text.strip().lower()
                hallucinations = {"obrigado.", "obrigada.", "obrigado", "obrigada", "obrigado!", "obrigada!", "obrigado por assistir.", "obrigada por assistir.", 
"thank you.", "thank you", "thanks for watching.", "obrigado por assistir"}
                if t_lower in hallucinations:
                    log(f"Ignoring hallucination: '{text}'")
                    text = ""

                if (vad_segment and not is_final and segment_id is not None
                        and not self.config.get("voice_chat")):
                    self.recorder.endpoint.update(segment_id, audio_end, text)

                if text.strip() and self.config.get("voice_chat"):
                    # Voice chat types nothing: the text goes to the LLM.
                    self._voice_chat_turn(text.strip(), audio, is_final)
                    return

                if text:
                    # The final stop check and the paste are one step with
                    # respect to stop(): once stop() returns, nothing pastes.
                    # No callback runs while the lock is held.
                    with self._output_lock:
                        if self._stopping.is_set():
                            log("Engine stopped; discarding late transcription.")
                            return
                            
                        # Append a trailing space in continuous mode so next phrase doesn't stick
                        # e.g., "legal" -> "legal ". If they manually type punctuation later,
                        # Windows handles it naturally, but this prevents "legalFicou".
                        if vad_segment:
                            stripped = text.strip()
                            if not is_final and stripped and not re.search(r'[.,!?;\:]$', stripped):
                                text = stripped + '... '
                            elif not text.endswith(' '):
                                text += ' '
                                
                        # Rewrite only what differs from the draft already
                        # typed, if keys still land where it was typed.
                        target = get_input_target()
                        text_to_type = text
                        if self._draft_typed_text:
                            if same_input_target(target, self._draft_target):
                                common = os.path.commonprefix([self._draft_typed_text, text])
                                delete_text(len(self._draft_typed_text) - len(common))
                                text_to_type = text[len(common):]
                            else:
                                log("Input focus changed; leaving the draft where it was typed.")

                        if is_final:
                            if text_to_type:
                                type_text(text_to_type, auto_enter=self.config["auto_enter"])
                            elif self.config["auto_enter"]:
                                # The draft already holds the whole text.
                                import keyboard
                                keyboard.press_and_release("enter")
                            self._draft_typed_text = ""
                            self._draft_target = None
                            log(f"Final transcription: {text.strip()}")

                            self._history.append({
                                "timestamp": datetime.now().isoformat(),
                                "text": text,
                                "duration": len(audio) / self.config["sample_rate"],
                            })
                            if len(self._history) > self.MAX_HISTORY:
                                self._history = self._history[-self.MAX_HISTORY:]
                        elif self.config.get("inline_drafts"):
                            # It's a draft. Type it so the user sees it real-time.
                            type_draft_text(text_to_type)
                            self._draft_typed_text = text
                            self._draft_target = target

                    # Re-read rather than reuse `continuous`: a tap may have
                    # stopped listening while this segment transcribed.
                    listening = self._continuous and self.is_recording
                    if is_final:
                        if listening:
                            # The text is already typed; no "Done" balloon
                            # per sentence while the microphone stays open.
                            self._set_state(AppState.RECORDING, {"draft_text": ""})
                        else:
                            self._set_state(AppState.READY, {"text": text})
                        self.last_draft_text = ""
                    else:
                        self.last_draft_text = text
                        if listening:
                            self._set_state(AppState.RECORDING, {"draft_text": text})
                else:
                    # An empty draft is often a dropped hallucination
                    # mid-sentence: keep the typed draft until the final
                    # decides. An empty final means there was no speech.
                    if is_final:
                        with self._output_lock:
                            self._forget_draft_locked(erase=True)
                        self.last_draft_text = ""
                        log("No speech detected.")
                        if self._continuous and self.is_recording:
                            self._set_state(AppState.RECORDING)
                        else:
                            self._set_state(AppState.READY)
                    elif self._continuous and self.is_recording:
                        # Draft but no text detected
                        self._set_state(AppState.RECORDING, {"draft_text": ""})
            except Exception as e:
                import traceback
                log(f"Error during transcription: {e}")
                log(traceback.format_exc())
                # A failed draft changes nothing: the segment keeps going and
                # its final (maybe on a fallback device) still corrects the
                # typed draft. A failed final ends the segment: leave the
                # draft as the best text we have, but stop tracking it so the
                # next segment does not backspace over it.
                if is_final:
                    with self._output_lock:
                        self._forget_draft_locked(erase=False)
                # Latches GPU failures even if shutdown started meanwhile.
                self._set_state(AppState.ERROR, self._error_payload(e))
        except Exception as exc:
            log(f"Error stopping recording: {exc}")
            self._set_state(AppState.ERROR, {"error": str(exc)})
        finally:
            with self._audio_lifecycle_lock:
                self._transcribing = False

    def _voice_chat_turn(self, text, audio, is_final):
        """Voice chat: show drafts, send the final to the LLM and speak its
        reply. Runs on the transcription thread, so the next segment waits
        until the reply has been spoken."""
        listening = self._continuous and self.is_recording
        if not is_final:
            if listening:
                self._set_state(AppState.RECORDING, {"draft_text": text})
            return
        if self._stopping.is_set():
            return
        log(f"Voice chat: {text!r}")
        # Voice chat was switched on mid-sentence: what was typed of it as
        # dictation goes, the sentence goes to the LLM instead.
        with self._output_lock:
            self._forget_draft_locked(erase=True)
        self._set_state(AppState.PROCESSING)
        # The microphone must not hear the reply.
        self.recorder.set_muted(True)
        try:
            reply = self.voice_chat.respond(
                text, on_reply=lambda r: self._set_state(AppState.SPEAKING, {"text": r}))
        finally:
            self.recorder.set_muted(False)
        if not reply and self.config["beep_on_start"]:
            self.chimes.play('warning')
        with self._output_lock:
            if self._stopping.is_set():
                return
            self._history.append({
                "timestamp": datetime.now().isoformat(),
                "text": f"{text}\n→ {reply}" if reply else text,
                "duration": len(audio) / self.config["sample_rate"],
            })
            if len(self._history) > self.MAX_HISTORY:
                self._history = self._history[-self.MAX_HISTORY:]
        if self._continuous and self.is_recording:
            self._set_state(AppState.RECORDING, {"draft_text": ""})
        else:
            self._set_state(AppState.READY, {"text": reply} if reply else None)

    # A press longer than this is push-to-talk: recording stops on release.
    HOLD_SECONDS = 0.4

    @property
    def continuous_active(self) -> bool:
        return self._continuous

    def toggle_recording(self):
        """Hotkey press. Holding is push-to-talk; a tap starts continuous
        listening (tap_action "continuous") or starts/stops a recording
        ("toggle"). Any press while continuous listening is on stops it."""
        if self._stopping.is_set():
            return
        if getattr(self, "_hotkey_held", False):
            # Ignore auto-repeat while the key is physically held
            return

        press_time = time.time()
        if self.voice_chat.speaking:
            # A press while the reply is on its way only cuts it short.
            self._hotkey_held = True
            self.voice_chat.interrupt()
            threading.Thread(target=self._watch_key, args=(press_time,), daemon=True).start()
            return
        if self._continuous:
            # Allowed while a segment transcribes: stopping must not wait.
            self._hotkey_held = True
            threading.Thread(target=self._end_continuous, daemon=True).start()
            threading.Thread(target=self._watch_key, args=(press_time,), daemon=True).start()
            return

        if self._transcribing:
            return
        self._hotkey_held = True
        if self.is_recording:
            # A "toggle" recording: this press stops it.
            threading.Thread(target=self._finish_recording, daemon=True).start()
            threading.Thread(target=self._watch_key, args=(press_time,), daemon=True).start()
        else:
            starter = threading.Thread(target=self._start_recording,
                                       args=(time.perf_counter(),), daemon=True)
            starter.start()
            threading.Thread(target=self._watch_key, args=(press_time, starter),
                             daemon=True).start()

    def _start_recording(self, press_perf):
        """Hotkey press: start a push-to-talk recording right away (a tap may
        turn it into continuous listening on release)."""
        t_thread_start = time.perf_counter()
        log(f"[Telemetry] Thread _start_recording spawned in {t_thread_start - press_perf:.3f}s after key press")

        failure = device_failure()
        if failure:
            # Another engine in this process may have hit the failure.
            log(f"Cannot record: {failure['message']}")
            self._set_state(AppState.ERROR, self._error_payload(
                RestartRequiredError(failure["message"])))
            return

        if self._load_error:
            log(f"Cannot record — model failed to load: {self._load_error}")
            return

        if self._lost_device and self._lost_device == self.config["device"]:
            log(f"Cannot record — {self._lost_device} was lost and no fallback "
                f"device is active yet.")
            if self.config["beep_on_start"]:
                self.chimes.play('warning')
            return

        if self._mic_warmup:
            log("Cannot record — microphone warmup in progress, try again in a moment.")
            if self.config["beep_on_start"]:
                self.chimes.play('warning')
            return

        t_before_rec = time.perf_counter()
        try:
            with self._audio_lifecycle_lock:
                if self._stopping.is_set() or self._transcribing or self._mic_warmup:
                    return
                try:
                    self.recorder.wait_ready(timeout=0)
                except Exception as e:
                    log(f"Audio stream not ready ({e}), attempting recovery...")
                    self.recorder.close()
                    self.recorder.warmup(timeout=3.0)
                # Routing happens after transcription. Snapshot the voice-chat toggle
                # and backend at recording start, also for a tap's continuous session.
                self._recording_claude_chat = self._claude_voice_chat()
                self.recorder.start()
                self.is_recording = True
        except Exception as exc:
            log(f"Cannot start recording: {exc}")
            self._set_state(AppState.ERROR, {"error": str(exc)})
            return
        t_after_rec = time.perf_counter()
        log(f"[Telemetry] recorder.start() completed in {t_after_rec - t_before_rec:.3f}s")

        if self.config["beep_on_start"]:
            self.chimes.play('start')
            log("[Telemetry] Start chime submitted to persistent output stream")

        self._set_state(AppState.RECORDING)

    def _watch_key(self, press_time, starter=None):
        """Wait for the hotkey release, then settle what a starting press
        meant: held = push-to-talk (stop and transcribe), tap = tap_action."""
        import keyboard
        while keyboard.is_pressed(self.config["hotkey"]):
            time.sleep(0.05)
        if starter is None:
            self._hotkey_held = False
            return
        starter.join()
        if time.time() - press_time > self.HOLD_SECONDS:
            self._hotkey_held = False
            self._finish_recording()
            return
        try:
            if self.config.get("tap_action", "continuous") == "continuous":
                self._begin_continuous(press_time)
        finally:
            self._hotkey_held = False

    def _begin_continuous(self, press_time):
        # Silero loads (first tap: downloads) outside the lifecycle lock, so
        # a stop press, Settings or busy_reason() never wait on it. The
        # push-to-talk capture keeps recording meanwhile.
        self.recorder.prepare_vad()
        with self._audio_lifecycle_lock:
            if self._stopping.is_set() or not self.is_recording or self._transcribing:
                return  # the start failed, or something already stopped it
            try:
                # Re-read from just before the press: speech may start mid-tap.
                self.recorder.begin_continuous(rewind_seconds=time.time() - press_time + 0.3)
            except Exception as e:
                log(f"Cannot start continuous listening: {e}")
                self.recorder.end_continuous()  # VAD paused, cuts nothing
                self.recorder.stop()
                self.is_recording = False
                failed = True
            else:
                self._continuous = True
                self._continuous_since = time.time()
                failed = False
        if failed:
            self._set_state(AppState.READY)
            return
        log("Continuous listening on. Tap the hotkey again to stop.")
        if self.config["beep_on_start"]:
            self.chimes.play('continuous')
        self._set_state(AppState.RECORDING)
        idle = self.config.get("continuous_idle_stop_seconds")
        if idle:
            threading.Thread(target=self._stop_continuous_when_idle,
                             args=(self._continuous_since, idle), daemon=True).start()

    def _stop_continuous_when_idle(self, session, idle_seconds, poll=1.0):
        """A stray tap must not leave the microphone open indefinitely."""
        while not self._stopping.wait(poll):
            if not self._continuous or self._continuous_since != session:
                return
            last_activity = max(session, self.recorder.last_speech_time)
            if time.time() - last_activity > idle_seconds:
                log(f"No speech for {idle_seconds}s; stopping continuous listening.")
                self._end_continuous(session)
                return

    def _end_continuous(self, session=None):
        """Stop continuous listening; with `session`, only that session (a
        tap may have stopped it and started another since it was checked)."""
        with self._audio_lifecycle_lock:
            if not self._continuous:
                return
            if session is not None and self._continuous_since != session:
                return
            self._continuous = False
            self.is_recording = False
            # Speech in progress is still cut, queued and typed; its final
            # sets READY with the text once transcribed.
            self.recorder.end_continuous()
        log("Continuous listening off.")
        if self.config["beep_on_start"]:
            self.chimes.play('stop')
        self._set_state(AppState.READY)

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
        self._start_loader()
        if self.config.get("voice_chat"):
            self._warm_up_voice_chat()
        self._start_segment_consumer()

    def set_voice_chat(self, enabled: bool):
        """Switch voice chat while running. On: the TTS server and the LLM
        load in the background. Off: the reply in progress stops. Both stay
        loaded either way, so switching back is instant."""
        self.config["voice_chat"] = bool(enabled)
        if enabled:
            log("Voice chat on.")
            self._warm_up_voice_chat()
        else:
            log("Voice chat off: dictation types again.")
            self.voice_chat.interrupt()
            if self.config.get("voice_chat_backend", "local") == "local":
                self.voice_chat.reset()

    def _warm_up_voice_chat(self):
        """Start the TTS server and load the LLM ahead of the first reply,
        in the background (the caller may be the Tk thread)."""
        threading.Thread(target=self._load_voice_chat, daemon=True).start()

    def _load_voice_chat(self):
        if not self._voice_chat_loading.acquire(blocking=False):
            return  # already loading
        try:
            # The TTS server (its own process, on the RTX) starts now.
            ensure_tts_server(self.config, log, TTS_SERVER_LOG)
            if self.config.get("voice_chat_backend", "local") == "claude":
                if self.config.get("voice_chat") and not self._stopping.is_set():
                    from debora_whisper.harness import start_harness
                    start_harness(self.config, log)
                    self._notice("Voice chat: Claude Code ready")
                return
            if llm_loaded(self.config):
                self._notice("Voice chat ready")
                return
            self._notice("Voice chat: loading the LLM...")
            # The LLM waits for the speech model's warmup inference, then
            # keeps every transcription waiting while it compiles.
            if not self._wait_speech_model():
                return
            download_llm(self.config, log)  # a first download takes minutes: not gated
            with _inference_gate:
                if load_llm(self.config, log, LLM_SERVER_LOG, stop=self._stopping) is None:
                    return
            if self.config.get("voice_chat"):
                self._notice("Voice chat ready")
                if self.config["beep_on_start"]:
                    self.chimes.play('start')
        except Exception as e:
            log(f"Voice chat: LLM not available ({e})")
            self._notice("Voice chat: the LLM did not load (see app.log)")
        finally:
            self._voice_chat_loading.release()

    def _notice(self, text: str):
        """A passing message for the user; the state does not change."""
        if self._stopping.is_set():
            return
        for cb in self._callbacks:
            try:
                cb(self._state, {"notice": text})
            except Exception:
                pass

    def _wait_speech_model(self) -> bool:
        """Block until the speech model is ready (or failed); False if the
        engine stopped meanwhile."""
        while not self._model_ready.wait(0.5):
            if self._stopping.is_set():
                return False
        return not self._stopping.is_set()

    def _start_segment_consumer(self):
        """Transcribe VAD segments. Always running: any tap may start
        continuous listening."""
        def _continuous_consumer():
            import queue
            import traceback
            pending = []
            while not self._stopping.is_set():
                busy = self.recorder.consumer_busy
                try:
                    if not pending:
                        pending.append(self.recorder.segment_queue.get(timeout=0.5))
                    # Take everything queued meanwhile and skip drafts a
                    # newer item has already superseded.
                    while True:
                        try:
                            pending.append(self.recorder.segment_queue.get_nowait())
                        except queue.Empty:
                            break
                    pending = [i if isinstance(i, tuple) else (i, True)
                               for i in pending if i is not None]
                    pending = drop_superseded_drafts(pending)
                    if not pending:
                        continue
                    segment = pending.pop(0)
                    audio_segment, is_final = segment[:2]

                    if len(audio_segment) > 0:
                        busy.set()
                        try:
                            # Se _transcribing estiver True, _finish_recording retorna.
                            # Mas queremos esperar até que ele termine.
                            while getattr(self, "_transcribing", False) and not self._stopping.is_set():
                                time.sleep(0.1)
                            metadata = ({"segment_id": segment.segment_id,
                                         "audio_end": segment.audio_end}
                                        if isinstance(segment, VadSegment) else {})
                            self._finish_recording(audio=audio_segment, is_final=is_final, **metadata)
                        finally:
                            busy.clear()
                except queue.Empty:
                    pass
                except Exception as e:
                    log(f"CRITICAL ERROR in _continuous_consumer: {e}\n{traceback.format_exc()}")
        
        threading.Thread(target=_continuous_consumer, daemon=True).start()

    def busy_reason(self) -> str | None:
        """What accelerator/microphone work is in flight, or None if idle."""
        with self._audio_lifecycle_lock:
            return self._busy_reason_locked()

    def _busy_reason_locked(self) -> str | None:
        if self.is_recording:
            return "recording"
        if self._transcribing:
            return "transcription"
        if self._loading:
            return "model loading"
        return None

    def stop_if_idle(self) -> str | None:
        """Stop this engine only if no recording, transcription or model load
        is in flight. Returns the busy reason (engine left running) or None
        (engine stopped).

        The idle check and the stop flag are set under the same lock every
        start path (hotkey start, timer/user stop, loader spawn) takes, so no
        new work can begin between the check and the stop. Used before a
        Settings rebuild so a new pipeline is never compiled while the old
        one may still be running on the same device. Never blocks on
        in-flight inference, which may be hung in a driver.
        """
        with self._audio_lifecycle_lock:
            reason = self._busy_reason_locked()
            if reason is not None:
                return reason
            self._stopping.set()
        self.stop()
        return None

    def stop(self):
        """Clean shutdown — unhook keyboard and stop recording if active."""
        # Taking the output lock orders this against a paste in progress:
        # after this line no transcription can be typed or added to history.
        with self._output_lock:
            self._stopping.set()
        self.voice_chat.interrupt()
        try:
            import keyboard
            keyboard.unhook_all()
        except Exception:
            pass
        with self._audio_lifecycle_lock:
            self.is_recording = False
            try:
                self.recorder.close()
            finally:
                self.chimes.close()

        try:
            import pyperclip
            draft = getattr(self, "last_draft_text", "")
            if draft:
                pyperclip.copy(draft)
                log(f"Saved incomplete draft to clipboard: {draft}")
        except Exception:
            pass

        log("Engine stopped.")

    def run(self):
        """Run the main application loop with global hotkey (CLI mode)."""
        import keyboard

        hotkey = self.config["hotkey"]

        log("=" * 60)
        log("Débora Whisper")
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
        self._start_segment_consumer()

        try:
            keyboard.wait()  # Block forever, handling hotkeys
        except KeyboardInterrupt:
            log("\nShutting down...")
            self.stop()


# ---------------------------------------------------------------------------
# Setup / Install dependencies
# ---------------------------------------------------------------------------
def run_setup():
    """Interactive setup: detect devices and download the model.

    Dependencies come from the installer (pip/uv or Start-Dictation.ps1),
    never from here: an installed or frozen app has no requirements.txt.
    """
    log("=" * 60)
    log("Débora Whisper - Setup")
    log("=" * 60)

    # 1. Check Python version
    log(f"Python: {sys.version}")

    # 2. Check NPU availability
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
        
    if has_nvidia_gpu():
        log("  [OK] CUDA detected (NVIDIA GPU)")

    # 3. Create default config
    config = load_config()
    config["device"] = select_device(config, detect_devices()) or "CPU"

    save_config(config)

    # 4. Download model
    log(f"\nDownloading {config['model_size']} model for {config['device']}...")
    model_path = None
    try:
        model_path = setup_model(config)
        log(f"Model ready at: {model_path}")
    except Exception as e:
        log(f"Model download failed: {e}")
        log("You can retry later with: debora-cli --setup")

    # 5. Warm the OpenVINO cache by running a dummy inference
    #    This triggers NPU compilation during setup so the first real use is fast.
    if model_path:
        log(f"\nWarming OpenVINO cache on {config['device']} (first compile may take 5-15 min)...")
        try:
            import numpy as np
            model_info = MODEL_REGISTRY[config["model_size"]]
            model = create_model(model_path, device=config["device"],
                                 backend=model_info["backend"],
                                 model_size=config["model_size"])
            silence = np.zeros(config["sample_rate"], dtype=np.float32)  # 1 second of silence
            model.transcribe(silence, sample_rate=config["sample_rate"], language=config["language"])
            log("Cache warm-up complete — subsequent starts will be fast.")
        except Exception as e:
            log(f"Cache warm-up failed (non-fatal): {e}")
            log("The cache will be built on first real use instead.")

    log("\n" + "=" * 60)
    log("Setup complete!")
    log(f"Config file: {CONFIG_FILE}")
    log("Start dictating: debora (tray app) or debora-cli")
    log("From a source checkout: .\\Start-Dictation.ps1")
    log("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(prog="debora-cli", description="Débora Whisper")
    parser.add_argument("--setup", action="store_true", help="Run first-time setup")
    parser.add_argument("--device", choices=["NPU", "GPU", "CPU", "CUDA"], help="Override device")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), help="Model size")
    parser.add_argument("--language", type=str, help="Language code (e.g., en, ru, id)")
    parser.add_argument("--auto-enter", action="store_true", help="Press Enter after typing")
    parser.add_argument("--hotkey", type=str, help="Global hotkey (e.g., ctrl+alt+d)")
    parser.add_argument("--continuous", action="store_true", help="Enable continuous listening (VAD)")
    parser.add_argument("--voice-chat", action="store_true",
                        help="Talk and hear the reply (voice chat) "
                             "instead of typing")
    parser.add_argument("--voice-chat-backend", choices=["local", "claude"], help="Voice chat backend")
    parser.add_argument("--harness-cwd", help="Claude Code folder (default: home)")
    args = parser.parse_args()
    log_folder_moves()

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
    if args.continuous:
        config["continuous_listening"] = True
    if args.voice_chat:
        config["voice_chat"] = True
    if args.voice_chat_backend:
        config["voice_chat_backend"] = args.voice_chat_backend
    if args.harness_cwd is not None:
        config["harness_cwd"] = args.harness_cwd

    validate_config(config)
    rotate_logs()
    if not args.device:
        apply_device_priority(config)
    avoid_lost_npu(config)

    app = DictationApp(config)
    app.run()


if __name__ == "__main__":
    main()
