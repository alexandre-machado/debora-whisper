"""Start Menu / Startup shortcuts for an installed debora-whisper.

`uv tool install` and pip only put commands on PATH. These shortcuts start
the tray app through pythonw, so no console window opens. They point at the
Python of the current install, so re-run `debora --install-shortcut`
after reinstalling into a different environment, and run
`debora --remove-shortcut` before `uv tool uninstall debora-whisper`.
"""
import os
import subprocess
import sys
from pathlib import Path

from debora_whisper.dictation_engine import CONFIG_DIR, log

SHORTCUT_NAME = "Débora Whisper.lnk"
ICON_PATH = CONFIG_DIR / "debora-whisper.ico"
# Created before the rename; they start a module that no longer exists.
LEGACY_SHORTCUT_NAME = "NPU Whisper.lnk"
LEGACY_ICON_PATH = CONFIG_DIR / "npu-whisper.ico"


def _programs_dir() -> Path:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise RuntimeError("APPDATA is not set; cannot locate the Start Menu.")
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs"


def shortcut_paths(name: str = SHORTCUT_NAME) -> dict:
    """Where the shortcuts live: the Start Menu and the per-user Startup folder."""
    programs = _programs_dir()
    return {"start_menu": programs / name,
            "startup": programs / "Startup" / name}


def _remove_legacy() -> list[Path]:
    """Delete the pre-rename shortcuts and icon."""
    removed = []
    for path in shortcut_paths(LEGACY_SHORTCUT_NAME).values():
        if path.exists():
            path.unlink()
            removed.append(path)
            log(f"Removed shortcut: {path}")
    LEGACY_ICON_PATH.unlink(missing_ok=True)
    return removed


def gui_python() -> Path:
    """pythonw.exe next to the running interpreter (no console window)."""
    exe = Path(sys.executable)
    pythonw = exe.with_name("pythonw.exe")
    return pythonw if pythonw.exists() else exe


def _write_icon() -> Path | None:
    try:
        from debora_whisper.ui.icons import render_app_icon
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        render_app_icon(size=256).save(ICON_PATH, sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])
        return ICON_PATH
    except Exception as e:
        log(f"Could not write the shortcut icon: {e}")
        return None


# -I: the app's own environment only. Without it, a debora_whisper.py (or any
# module it imports) in the working directory would shadow the installed one.
SHORTCUT_ARGS = "-I -m debora_whisper"

# Values reach PowerShell as environment variables, never spliced into the
# script, so paths with quotes or `$` cannot change what it runs.
_CREATE_LNK = (
    "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:NPW_LNK);"
    "$s.TargetPath = $env:NPW_TARGET;"
    "$s.Arguments = $env:NPW_ARGS;"
    "$s.WorkingDirectory = $env:NPW_WORKDIR;"
    "$s.Description = 'Local voice dictation (debora-whisper)';"
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
    """Create the Start Menu shortcut, plus a Startup one if autostart (or
    if a pre-rename Startup shortcut was replaced)."""
    paths = shortcut_paths()
    autostart = autostart or shortcut_paths(LEGACY_SHORTCUT_NAME)["startup"] in _remove_legacy()
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
    """Delete both shortcuts and the icon, pre-rename ones too. Missing ones
    are not an error."""
    removed = _remove_legacy()
    for path in shortcut_paths().values():
        if path.exists():
            path.unlink()
            removed.append(path)
            log(f"Removed shortcut: {path}")
    ICON_PATH.unlink(missing_ok=True)
    if not removed:
        log("No debora-whisper shortcuts to remove.")
    return removed
