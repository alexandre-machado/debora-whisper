"""What Débora asks of the operating system.

Each method has a portable answer, or "unknown" (None) when only the OS can
tell. A provider per OS (windows.py today) overrides what it knows, so the
rest of the app never checks sys.platform itself.
"""


class Platform:
    name = "generic"

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
