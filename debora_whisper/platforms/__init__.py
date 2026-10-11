"""The provider for the operating system Débora runs on."""
import sys

from debora_whisper.platforms.base import Platform

_current = None


def current() -> Platform:
    global _current
    if _current is None:
        if sys.platform == "win32":
            from debora_whisper.platforms.windows import WindowsPlatform
            _current = WindowsPlatform()
        else:
            _current = Platform()
    return _current
