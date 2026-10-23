"""Windows provider.

Each playback endpoint's form factor is in the registry, so telling
headphones from speakers needs no extra package.
"""
import os
import subprocess
import ctypes
import threading
import time
from ctypes import wintypes
from functools import lru_cache
from pathlib import Path
from types import FunctionType, SimpleNamespace

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
_user32 = ctypes.WinDLL("user32", use_last_error=True) if os.name == "nt" else None
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True) if os.name == "nt" else None
_advapi32 = ctypes.WinDLL("advapi32", use_last_error=True) if os.name == "nt" else None
if _user32 is not None:
    _user32.PostThreadMessageW.argtypes = [ctypes.c_ulong, ctypes.c_uint,
                                         ctypes.c_size_t, ctypes.c_ssize_t]
    _user32.PostThreadMessageW.restype = ctypes.c_int
    _user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND,
                                   wintypes.UINT, wintypes.UINT]
    _user32.GetMessageW.restype = wintypes.BOOL
    _user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
    _user32.TranslateMessage.restype = wintypes.BOOL
    _user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
    _user32.DispatchMessageW.restype = ctypes.c_ssize_t
    _user32.GetForegroundWindow.argtypes = []
    _user32.GetForegroundWindow.restype = wintypes.HWND
    _user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    _user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                         ctypes.POINTER(wintypes.HANDLE)]
    _advapi32.OpenProcessToken.restype = wintypes.BOOL
    _advapi32.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                            wintypes.LPVOID, wintypes.DWORD,
                                            ctypes.POINTER(wintypes.DWORD)]
    _advapi32.GetTokenInformation.restype = wintypes.BOOL


def _process_elevated(pid):
    """Return None on failure, so inaccessible tokens don't suppress recovery."""
    if _kernel32 is None:
        return None
    try:
        process = _kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not process:
            return None
        try:
            token = wintypes.HANDLE()
            if not _advapi32.OpenProcessToken(process, 0x0008, ctypes.byref(token)):  # TOKEN_QUERY
                return None
            try:
                elevated, size = wintypes.DWORD(), wintypes.DWORD()
                if not _advapi32.GetTokenInformation(
                        token, 20, ctypes.byref(elevated), ctypes.sizeof(elevated),
                        ctypes.byref(size)):  # TokenElevation
                    return None
                return bool(elevated.value)
            finally:
                _kernel32.CloseHandle(token)
        finally:
            _kernel32.CloseHandle(process)
    except OSError:
        return None


@lru_cache(maxsize=1)
def _self_elevated():
    return _process_elevated(os.getpid())


def _foreground_elevated():
    if _user32 is None:
        return False
    try:
        window = _user32.GetForegroundWindow()
        pid = wintypes.DWORD()
        if not window or not _user32.GetWindowThreadProcessId(window, ctypes.byref(pid)) or not pid.value:
            return False
        return _process_elevated(pid.value) is True
    except OSError:
        return False


def _keyboard_hook_args(backend, callback):
    # keyboard 0.13.x builds its decoder closure inside prepare_intercept and
    # retains each native callback via atexit. Run that builder with private
    # globals to capture the install arguments without installing or registering
    # cleanup. No package globals are patched; the listener retains ONE callback
    # for every recovery, and our loop owns the hook handle and its cleanup.
    captured = []
    namespace = dict(backend.prepare_intercept.__globals__)
    namespace.update(SetWindowsHookEx=lambda *args: captured.append(args),
                     atexit=SimpleNamespace(register=lambda *args: None))
    FunctionType(backend.prepare_intercept.__code__, namespace)(callback)
    return captured[0], namespace


def _listen_safely(backend, listener):
    if not hasattr(listener, "_debora_hook_args"):
        listener._debora_hook_args, listener._debora_hook_globals = _keyboard_hook_args(
            backend, listener.direct_callback)
    # The decoder closure uses these private globals for its modifier state.
    listener._debora_hook_globals.update(shift_is_pressed=False, altgr_is_pressed=False,
                                         ignore_next_right_alt=False)
    hook = backend.SetWindowsHookEx(*listener._debora_hook_args)
    if not hook:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        message = wintypes.MSG()
        while True:
            result = _user32.GetMessageW(ctypes.byref(message), None, 0, 0)
            if result == 0:  # WM_QUIT
                return
            if result == -1:
                raise ctypes.WinError(ctypes.get_last_error())
            _user32.TranslateMessage(ctypes.byref(message))
            _user32.DispatchMessageW(ctypes.byref(message))
    finally:
        backend.UnhookWindowsHookEx(hook)


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
        self._elevated = _self_elevated()

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
        # Derive only pressed/not-pressed; raw key states are never logged or
        # stored beyond this poll (held keys are compared with the next poll).
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
        if self._elevated is False and _foreground_elevated():
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
        self._watchdog = KeyboardHookWatchdog(stopping, recover, log)
        return self._watchdog

    def restart_keyboard_listener(self, keyboard, stopping):
        # Keep the processing thread and unrelated registrations. Only the first
        # listener uses keyboard 0.13.x's NULL LPMSG: wake it with WM_NULL and
        # quietly consume its expected ctypes OSError during shutdown. All our
        # replacements own a real MSG and unhook cleanly on WM_QUIT.
        listener = keyboard._listener
        with listener.lock:
            old = listener.listening_thread
            if old.is_alive():
                legacy = getattr(old, "_debora_message_loop", False) is not True
                previous_hook = threading.excepthook

                def quiet_old_listener(args):
                    if args.thread is old and issubclass(args.exc_type, OSError):
                        return
                    previous_hook(args)

                try:
                    if legacy:
                        threading.excepthook = quiet_old_listener
                        _user32.PostThreadMessageW(old.native_id, 0, 0, 0)  # WM_NULL
                    _user32.PostThreadMessageW(old.native_id, 0x0012, 0, 0)  # WM_QUIT
                    old.join(1.0)
                finally:
                    if legacy:
                        threading.excepthook = previous_hook
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
                target=_listen_safely, args=(backend, listener),
                name="keyboard-listener", daemon=True)
            listener.listening_thread._debora_message_loop = True
            listener.listening_thread.start()

    def type_text(self, text: str, auto_enter: bool = False):
        return self._observe_typing(windows_text.type_text, text, auto_enter)

    def _observe_typing(self, type_text, *args):
        watchdog = getattr(self, "_watchdog", None)
        # Replayed keys and VK_PACKET bypass the passive keyboard observer.
        if watchdog is not None:
            watchdog.observe(None)
        try:
            return type_text(*args)
        finally:
            if watchdog is not None:
                watchdog.observe(None)

    def _type_text_ctypes(self, text: str):
        return self._observe_typing(windows_text._type_text_ctypes, text)

    def type_draft(self, text: str):
        return self._observe_typing(windows_text.type_draft, text)

    def delete_text(self, count: int):
        return self._observe_typing(windows_text.delete_text, count)

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
