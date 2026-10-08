"""Child processes the app starts and must be able to end."""
import subprocess
import sys
from pathlib import Path

# No console window for children of the windowless (pythonw) app.
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def python_executable() -> str:
    """This interpreter for a child talking over stdin/stdout: python.exe
    rather than the Start Menu's pythonw.exe, which may have no standard
    streams. NO_WINDOW keeps it windowless."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").is_file():
        return str(exe.with_name("python.exe"))
    return sys.executable


def kill_tree(process: subprocess.Popen):
    """Kill process and its children, without waiting for them.

    On Windows a venv's python.exe and uv both run the real interpreter as a
    child: killing only the parent leaves that child running, holding its
    GPU or NPU."""
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)],
                       capture_output=True, creationflags=NO_WINDOW)
    else:
        process.kill()
