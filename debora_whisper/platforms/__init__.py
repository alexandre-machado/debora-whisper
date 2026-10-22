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
        elif sys.platform.startswith("linux"):
            from debora_whisper.platforms.linux import LinuxPlatform
            _current = LinuxPlatform()
        else:
            _current = Platform()
    return _current
