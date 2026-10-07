"""Where npu-whisper keeps its files.

Config and logs live in ~/.npu-dictation. Models, compiled model caches and
voices go to a shared models folder when MODELS_DIR is set (models under
$MODELS_DIR/npu-whisper, voices under $MODELS_DIR/voices), else also under
~/.npu-dictation. Models taken from the Hugging Face cache (the voice chat
LLM, faster-whisper, Chatterbox) follow HF_HOME instead.
"""
import os
import sys
from pathlib import Path


def user_env(name: str) -> str | None:
    """The variable from this process, or else (Windows) as saved for the
    user: a terminal opened before it was set does not pass it on, and the
    app would quietly use, and download into, other folders."""
    value = os.environ.get(name)
    if value or sys.platform != "win32":
        return value or None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
    except OSError:
        return None
    return os.path.expandvars(str(value)) or None


# huggingface_hub reads HF_HOME when imported, and the TTS server inherits it.
for _name in ("MODELS_DIR", "HF_HOME"):
    if not os.environ.get(_name) and (_value := user_env(_name)):
        os.environ[_name] = _value

CONFIG_DIR = Path.home() / ".npu-dictation"
CONFIG_FILE = CONFIG_DIR / "config.json"
LOG_DIR = CONFIG_DIR / "logs"

_shared = user_env("MODELS_DIR")
DATA_DIR = Path(_shared) / "npu-whisper" if _shared else CONFIG_DIR
MODEL_DIR = DATA_DIR / "models"
CACHE_DIR = DATA_DIR / "ov-cache"
VOICES_DIR = Path(_shared) / "voices" if _shared else CONFIG_DIR / "voices"
