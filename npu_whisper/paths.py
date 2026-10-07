"""Where npu-whisper keeps its files.

Config and logs live in ~/.npu-dictation. Models, compiled model caches and
voices go to a shared models folder when MODELS_DIR is set (models under
$MODELS_DIR/npu-whisper, voices under $MODELS_DIR/voices), else also under
~/.npu-dictation. Models taken from the Hugging Face cache (the voice chat
LLM, faster-whisper, Chatterbox) follow HF_HOME instead.
"""
import os
from pathlib import Path

CONFIG_DIR = Path.home() / ".npu-dictation"
CONFIG_FILE = CONFIG_DIR / "config.json"
LOG_DIR = CONFIG_DIR / "logs"

_shared = os.environ.get("MODELS_DIR")
DATA_DIR = Path(_shared) / "npu-whisper" if _shared else CONFIG_DIR
MODEL_DIR = DATA_DIR / "models"
CACHE_DIR = DATA_DIR / "ov-cache"
VOICES_DIR = Path(_shared) / "voices" if _shared else CONFIG_DIR / "voices"
