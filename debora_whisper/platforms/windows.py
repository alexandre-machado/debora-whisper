"""Windows provider.

Each playback endpoint's form factor is in the registry, so telling
headphones from speakers needs no extra package.
"""
import os
import subprocess
import ctypes
import threading
import time
from pathlib import Path

from debora_whisper.platforms.base import CaretReadStatus, Platform
from debora_whisper.platforms import windows_text, windows_uia

_RENDER = r"SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Render"
_DESCRIPTION = "{a45c254e-df1c-4efd-8020-67d146a850e0},2"  # "Headphones"
_DEVICE_NAMES = ("{b3f8fa53-0004-438e-9003-51a46e139bfc},26",  # "LE-HD 450BT"
                 "{b3f8fa53-0004-438e-9003-51a46e139bfc},6")   # "Realtek(R) Audio"
_FORM_FACTOR = "{1da5d803-d492-4edd-8c23-e0c0ffee7f0e},0"
_ACTIVE = 1
_HEADPHONES, _HEADSET = 3, 5
# MME, sounddevice's default host API, cuts device names at 31 characters.
_MME_NAME_CHARS = 31


class KeyboardHookWatchdog:
    """Confirm missing hook events against independent Windows input state.

    Mouse input alone never triggers recovery. Give a detected key one second
    to reach the hook, then replace the native listener as well as our bindings:
    keyboard.add_hotkey alone cannot revive a WH_KEYBOARD_LL removed by Windows.
    The same check covers input after sleep/unlock without a hidden window.
    """
    STALE_SECONDS = 3.0
    RETRY_SECONDS = 30.0

    def __init__(self, stopping, recover, log, *, clock=time.monotonic, probe=None):
        self._stopping, self._recover, self._log = stopping, recover, log
        self._clock = clock
        self._probe = probe or self._input_state
        self._last_event = clock()
        self._last_attempt = float("-inf")
        self._suspect = None
        self._keys_down = set()
        self._thread = None

    def observe(self, event):
        self._last_event = self._clock()

    def _input_state(self):
        from ctypes import wintypes

        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

        user32 = ctypes.windll.user32
        info = LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(info)
        if not user32.GetLastInputInfo(ctypes.byref(info)):
            return float("inf"), False
        # Both counters are DWORD milliseconds, including across the 49-day wrap.
        age = ((ctypes.windll.kernel32.GetTickCount() - info.dwTime) & 0xffffffff) / 1000
        states = {key: user32.GetAsyncKeyState(key) for key in range(8, 255)}
        down = {key for key, state in states.items() if state & 0x8000}
        # The low bit can miss a quick tap (another process may consume it).
        # Newly held keys provide an independent fallback; mouse buttons excluded.
        pressed = bool(down - self._keys_down) or any(state & 1 for state in states.values())
        self._keys_down = down
        return age, pressed

    def poll(self):
        now = self._clock()
        if self._stopping.is_set():
            return
        age, key_pressed = self._probe()
        if now - self._last_event < self.STALE_SECONDS:
            self._suspect = None
            return
        if now - self._last_attempt < self.RETRY_SECONDS:
            return
        if self._suspect is None:
            if key_pressed and age < 1.0:
                self._suspect = (now, self._last_event)
            return
        since, last_event = self._suspect
        if self._last_event != last_event:
            self._suspect = None
            return
        if now - since < 1.0:
            return
        self._last_attempt = now
        self._suspect = None
        if self._recover():
            self._last_event = now
            self._log("Keyboard hook lost; hotkeys re-registered.")

    def _run(self):
        while not self._stopping.wait(0.5):
            try:
                self.poll()
            except Exception as exc:
                now = self._clock()
                if now - getattr(self, "_last_error", float("-inf")) >= self.RETRY_SECONDS:
                    self._last_error = now
                    self._log(f"Keyboard hook recovery failed: {exc}")

    def start(self):
        self._thread = threading.Thread(target=self._run, name="keyboard-watchdog", daemon=True)
        self._thread.start()

    def join(self, timeout):
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)


def render_endpoints():
    """[(name as audio APIs show it, form factor)] of active playback devices."""
    import winreg
    endpoints = []
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _RENDER)
    except OSError:
        return []
    with root:
        index = 0
        while True:
            try:
                guid = winreg.EnumKey(root, index)
            except OSError:
                break
            index += 1
            try:
                with winreg.OpenKey(root, guid) as key:
                    if winreg.QueryValueEx(key, "DeviceState")[0] != _ACTIVE:
                        continue
                with winreg.OpenKey(root, guid + r"\Properties") as props:
                    values = {}
                    for name in (_DESCRIPTION, _FORM_FACTOR, *_DEVICE_NAMES):
                        try:
                            values[name] = winreg.QueryValueEx(props, name)[0]
                        except OSError:
                            pass
            except OSError:
                continue
            device = next((values[n] for n in _DEVICE_NAMES if values.get(n)), None)
            if values.get(_DESCRIPTION) and device:
                endpoints.append((f"{values[_DESCRIPTION]} ({device})",
                                  values.get(_FORM_FACTOR)))
    return endpoints


def is_headphones(name, endpoints):
    """True/False by the matching endpoint's form factor, None if not found."""
    if not name:
        return None
    matches = {factor for full, factor in endpoints
               if full == name or (len(name) >= _MME_NAME_CHARS and full.startswith(name))}
    if len(matches) != 1:
        return None
    return matches.pop() in (_HEADPHONES, _HEADSET)


class WindowsPlatform(Platform):
    name = "windows"

    def keyboard_watchdog(self, stopping, recover, log):
        return KeyboardHookWatchdog(stopping, recover, log)

    def restart_keyboard_listener(self, keyboard, stopping):
        # keyboard 0.13.x owns one native hook on listening_thread. Its public
        # unhook APIs only remove Python callbacks. Exiting the installing thread
        # removes that thread's native hook; retain the processing thread and
        # unrelated registrations. WM_NULL wakes 0.13.x's inverted GetMessage
        # loop, WM_QUIT also handles a corrected message loop.
        listener = keyboard._listener
        with listener.lock:
            old = listener.listening_thread
            if old.is_alive():
                post = ctypes.windll.user32.PostThreadMessageW
                post.argtypes = [ctypes.c_ulong, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t]
                post.restype = ctypes.c_int
                post(old.native_id, 0, 0, 0)  # WM_NULL
                post(old.native_id, 0x0012, 0, 0)  # WM_QUIT
                old.join(1.0)
                if old.is_alive():
                    raise RuntimeError("keyboard listener did not stop")
            # Lost up events otherwise leave Ctrl/Alt or a suppressed hotkey stuck.
            with keyboard._pressed_events_lock:
                keyboard._pressed_events.clear()
                keyboard._logically_pressed_keys.clear()
            listener.active_modifiers.clear()
            listener.modifier_states.clear()
            backend = keyboard._os_keyboard
            backend.shift_is_pressed = backend.altgr_is_pressed = False
            backend.ignore_next_right_alt = False
            if stopping.is_set():
                return
            listener.listening_thread = threading.Thread(
                target=listener.listen, name="keyboard-listener", daemon=True)
            listener.listening_thread.start()

    def type_text(self, text: str, auto_enter: bool = False):
        return windows_text.type_text(text, auto_enter)

    def _type_text_ctypes(self, text: str):
        return windows_text._type_text_ctypes(text)

    def type_draft(self, text: str):
        return windows_text.type_draft(text)

    def delete_text(self, count: int):
        return windows_text.delete_text(count)

    def _delete_text_ctypes(self, count: int):
        return windows_text._delete_text_ctypes(count)

    def _backspace_text_ctypes(self, count: int):
        return windows_text._backspace_text_ctypes(count)

    def input_target(self):
        return windows_text.input_target()

    def text_before_caret(self, count: int) -> str | CaretReadStatus | None:
        return windows_uia.text_before_caret(count)

    def same_input_target(self, a, b) -> bool:
        return windows_text.same_input_target(a, b)

    def nvidia_smi(self):
        system = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        if system.is_absolute() and system.is_file():
            return str(system)
        return super().nvidia_smi()

    def add_library_dir(self, path):
        os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(path))

    @property
    def no_window(self):
        return subprocess.CREATE_NO_WINDOW

    def kill_tree(self, process):
        """Kill process and its children, without waiting for them.

        On Windows a venv's python.exe and uv both run the real interpreter as a
        child: killing only the parent leaves that child running, holding its
        GPU or NPU."""
        if process.poll() is not None:
            return
        result = subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)],
                                capture_output=True, creationflags=self.no_window)
        if result.returncode != 0 and process.poll() is None:
            # taskkill may be denied process enumeration even though our
            # existing Popen handle can terminate the child. Otherwise its
            # pipe readers survive the shutdown as well.
            process.kill()

    def output_is_headphones(self, name=None):
        name = self.default_output_name() if name is None else name
        return is_headphones(name, render_endpoints())
