"""What Débora asks of the operating system.

Each method has a portable answer, or "unknown" (None) when only the OS can
tell. A provider per OS (windows.py today) overrides what it knows, so the
rest of the app never checks sys.platform itself.
"""
import shutil
from pathlib import Path


class Platform:
    name = "generic"
    no_window = 0

    def nvidia_smi(self):
        """Absolute executable from PATH, never from the current folder."""
        found = shutil.which("nvidia-smi")
        if found is None or not Path(found).is_absolute():
            return None
        path = Path(found).resolve()
        return None if path.parent == Path.cwd().resolve() else str(path)

    def add_library_dir(self, path):
        """Extra library directories are only needed on Windows today."""
        pass

    def kill_tree(self, process):
        """Kill a running process, without waiting for it."""
        if process.poll() is not None:
            return
        process.kill()

    def default_output_name(self):
        """Name of the device her voice plays on, as the audio API shows it."""
        try:
            import sounddevice as sd
            return sd.query_devices(kind="output")["name"]
        except Exception:
            return None

    def output_is_headphones(self, name=None):
        """True for headphones or a headset, False for speakers, None when
        unknown. With headphones her voice cannot reach the microphone."""
        return None
