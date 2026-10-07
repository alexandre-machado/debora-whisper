"""Start Menu / Startup shortcuts for an installed npu-whisper.

`uv tool install` and pip only put commands on PATH. These shortcuts start
the tray app through pythonw, so no console window opens. They point at the
Python of the current install, so re-run `npu-whisper --install-shortcut`
after reinstalling into a different environment, and run
`npu-whisper --remove-shortcut` before `uv tool uninstall npu-whisper`.
"""
import os
import subprocess
import sys
from pathlib import Path

from npu_whisper.dictation_engine import CONFIG_DIR, log

SHORTCUT_NAME = "NPU Whisper.lnk"
ICON_PATH = CONFIG_DIR / "npu-whisper.ico"


def _programs_dir() -> Path:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise RuntimeError("APPDATA is not set; cannot locate the Start Menu.")
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs"


def shortcut_paths() -> dict:
    """Where the shortcuts live: the Start Menu and the per-user Startup folder."""
    programs = _programs_dir()
    return {"start_menu": programs / SHORTCUT_NAME,
            "startup": programs / "Startup" / SHORTCUT_NAME}


def gui_python() -> Path:
    """pythonw.exe next to the running interpreter (no console window)."""
    exe = Path(sys.executable)
    pythonw = exe.with_name("pythonw.exe")
    return pythonw if pythonw.exists() else exe


def _write_icon() -> Path | None:
    try:
        from npu_whisper.ui.icons import render_app_icon
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        render_app_icon(size=256).save(ICON_PATH, sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])
        return ICON_PATH
    except Exception as e:
        log(f"Could not write the shortcut icon: {e}")
        return None


# -I: the app's own environment only. Without it, a npu_whisper.py (or any
# module it imports) in the working directory would shadow the installed one.
SHORTCUT_ARGS = "-I -m npu_whisper"

# Values reach PowerShell as environment variables, never spliced into the
# script, so paths with quotes or `$` cannot change what it runs.
_CREATE_LNK = (
    "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:NPW_LNK);"
    "$s.TargetPath = $env:NPW_TARGET;"
    "$s.Arguments = $env:NPW_ARGS;"
    "$s.WorkingDirectory = $env:NPW_WORKDIR;"
    "$s.Description = 'Local voice dictation (npu-whisper)';"
    "if ($env:NPW_ICON) { $s.IconLocation = $env:NPW_ICON };"
    "$s.Save()"
)


def _powershell() -> str:
    """By absolute path, so a powershell.exe on PATH or in the cwd is never run."""
    root = os.environ.get("SystemRoot", r"C:\Windows")
    return str(Path(root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe")


def _create_lnk(path: Path, target: Path, icon: Path | None):
    path.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "NPW_LNK": str(path), "NPW_TARGET": str(target),
           "NPW_ARGS": SHORTCUT_ARGS, "NPW_WORKDIR": str(CONFIG_DIR),
           "NPW_ICON": str(icon) if icon else ""}
    subprocess.run([_powershell(), "-NoProfile", "-NonInteractive", "-Command", _CREATE_LNK],
                   env=env, check=True, capture_output=True, text=True)


def install(autostart: bool = False) -> list[Path]:
    """Create the Start Menu shortcut, plus a Startup one if autostart."""
    paths = shortcut_paths()
    wanted = [paths["start_menu"]] + ([paths["startup"]] if autostart else [])
    target, icon = gui_python(), _write_icon()
    for path in wanted:
        _create_lnk(path, target, icon)
        log(f"Created shortcut: {path}")
    if not autostart and paths["startup"].exists():
        log(f"Start with Windows stays on ({paths['startup']}); "
            f"--remove-shortcut removes it.")
    return wanted


def remove() -> list[Path]:
    """Delete both shortcuts and the icon. Missing ones are not an error."""
    removed = []
    for path in shortcut_paths().values():
        if path.exists():
            path.unlink()
            removed.append(path)
            log(f"Removed shortcut: {path}")
    ICON_PATH.unlink(missing_ok=True)
    if not removed:
        log("No npu-whisper shortcuts to remove.")
    return removed
