"""Windows hotkeys and passive key events, without a low-level keyboard hook."""
import ctypes
import os
import threading
from ctypes import wintypes
from types import SimpleNamespace

from debora_whisper.solo_key import SoloKeyTap

MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, MOD_NOREPEAT = 1, 2, 4, 8, 0x4000
WM_INPUT, WM_HOTKEY, WM_QUIT = 0x00ff, 0x0312, 0x0012
RIDEV_INPUTSINK, RIDEV_REMOVE, RID_INPUT = 0x100, 1, 0x10000003
RI_KEY_BREAK, RI_KEY_E0, RI_KEY_E1 = 1, 2, 4
HOTKEY_ID = 1

_MODIFIERS = {"alt": MOD_ALT, "ctrl": MOD_CONTROL, "control": MOD_CONTROL,
              "shift": MOD_SHIFT, "win": MOD_WIN, "windows": MOD_WIN}
_KEYS = {"space": 0x20, "tab": 0x09, "enter": 0x0d, "return": 0x0d,
         "esc": 0x1b, "escape": 0x1b, "backspace": 0x08, "delete": 0x2e,
         "insert": 0x2d, "home": 0x24, "end": 0x23, "page up": 0x21,
         "page down": 0x22, "up": 0x26, "down": 0x28, "left": 0x25,
         "right": 0x27, "caps lock": 0x14, "num lock": 0x90,
         "scroll lock": 0x91, "pause": 0x13, "print screen": 0x2c}
_SIDE_KEYS = {"left ctrl": 0xa2, "right ctrl": 0xa3, "left shift": 0xa0,
              "right shift": 0xa1, "left alt": 0xa4, "right alt": 0xa5,
              "alt gr": 0xa5, "left windows": 0x5b, "right windows": 0x5c}
_MODIFIER_KEYS = {MOD_ALT: (0xa4, 0xa5), MOD_CONTROL: (0xa2, 0xa3),
                  MOD_SHIFT: (0xa0, 0xa1), MOD_WIN: (0x5b, 0x5c)}


def key_to_vk(key):
    key = key.strip().lower()
    if key in _KEYS:
        return _KEYS[key]
    if len(key) == 1 and key in "abcdefghijklmnopqrstuvwxyz0123456789":
        return ord(key.upper())
    if key.startswith("f") and key[1:].isdigit() and 1 <= int(key[1:]) <= 24:
        return 0x70 + int(key[1:]) - 1
    raise ValueError(f"unsupported key {key!r}")


def parse_hotkey(hotkey):
    """Return RegisterHotKey modifiers and VK for a single chord."""
    if not isinstance(hotkey, str) or not hotkey.strip() or "," in hotkey:
        raise ValueError("expected one chord, such as ctrl+space; sequences are unsupported")
    modifiers, keys = 0, []
    for key in hotkey.lower().split("+"):
        key = key.strip()
        if key in _MODIFIERS:
            modifiers |= _MODIFIERS[key]
        elif key in _SIDE_KEYS:
            raise ValueError("RegisterHotKey cannot distinguish left/right modifiers")
        else:
            keys.append(key_to_vk(key))
    if len(keys) != 1:
        raise ValueError("expected exactly one non-modifier key")
    if keys[0] == 0x7b:
        raise ValueError("F12 is reserved by Windows for the debugger")
    return modifiers | MOD_NOREPEAT, keys[0]


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [("usUsagePage", wintypes.USHORT), ("usUsage", wintypes.USHORT),
                ("dwFlags", wintypes.DWORD), ("hwndTarget", wintypes.HWND)]


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [("dwType", wintypes.DWORD), ("dwSize", wintypes.DWORD),
                ("hDevice", wintypes.HANDLE), ("wParam", wintypes.WPARAM)]


class RAWKEYBOARD(ctypes.Structure):
    _fields_ = [("MakeCode", wintypes.USHORT), ("Flags", wintypes.USHORT),
                ("Reserved", wintypes.USHORT), ("VKey", wintypes.USHORT),
                ("Message", wintypes.UINT), ("ExtraInformation", wintypes.ULONG)]


WNDPROC = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)(
    ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)


class WNDCLASS(ctypes.Structure):
    _fields_ = [("style", wintypes.UINT), ("lpfnWndProc", WNDPROC),
                ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
                ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HICON),
                ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH),
                ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR)]


_user32 = ctypes.WinDLL("user32", use_last_error=True) if os.name == "nt" else None
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True) if os.name == "nt" else None
if _user32 is not None:
    for name, args, result in (
        ("RegisterHotKey", [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT], wintypes.BOOL),
        ("UnregisterHotKey", [wintypes.HWND, ctypes.c_int], wintypes.BOOL),
        ("RegisterClassW", [ctypes.POINTER(WNDCLASS)], wintypes.ATOM),
        ("UnregisterClassW", [wintypes.LPCWSTR, wintypes.HINSTANCE], wintypes.BOOL),
        ("CreateWindowExW", [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR,
                             wintypes.DWORD, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             ctypes.c_int, wintypes.HWND, wintypes.HMENU,
                             wintypes.HINSTANCE, wintypes.LPVOID], wintypes.HWND),
        ("DestroyWindow", [wintypes.HWND], wintypes.BOOL),
        ("DefWindowProcW", [wintypes.HWND, wintypes.UINT, wintypes.WPARAM,
                            wintypes.LPARAM], ctypes.c_ssize_t),
        ("RegisterRawInputDevices", [ctypes.POINTER(RAWINPUTDEVICE), wintypes.UINT,
                                     wintypes.UINT], wintypes.BOOL),
        ("GetRawInputData", [wintypes.HANDLE, wintypes.UINT, wintypes.LPVOID,
                             ctypes.POINTER(wintypes.UINT), wintypes.UINT], wintypes.UINT),
        ("GetMessageW", [ctypes.POINTER(wintypes.MSG), wintypes.HWND,
                         wintypes.UINT, wintypes.UINT], wintypes.BOOL),
        ("DispatchMessageW", [ctypes.POINTER(wintypes.MSG)], ctypes.c_ssize_t),
        ("PostThreadMessageW", [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM,
                                wintypes.LPARAM], wintypes.BOOL),
        ("GetAsyncKeyState", [ctypes.c_int], ctypes.c_short),
    ):
        function = getattr(_user32, name)
        function.argtypes, function.restype = args, result
    _kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    _kernel32.GetModuleHandleW.restype = wintypes.HMODULE


class WindowsHotkeys:
    """One owner thread for RegisterHotKey and a Raw Input message-only window.

    Callbacks only enqueue actions or record release state. Raw events are
    discarded after dispatch; only SoloKeyTap's current held-key state remains.
    """

    def __init__(self, hotkey, on_press, on_release, voice_key, on_voice, log):
        self.hotkey = hotkey
        self._on_press, self._on_release, self._log = on_press, on_release, log
        self._modifiers, self._vk = 0, None
        self._solo = None
        self._voice_vks = ()
        self._registered = False
        self._ready = threading.Event()
        self._stopping = threading.Event()
        self._thread = None
        try:
            self._modifiers, self._vk = parse_hotkey(hotkey)
        except ValueError as exc:
            log(f"Hotkey {hotkey!r} disabled: {exc}.")
        voice_key = voice_key.strip().lower()
        if voice_key:
            try:
                if voice_key in _MODIFIERS:
                    self._voice_vks = _MODIFIER_KEYS[_MODIFIERS[voice_key]]
                else:
                    self._voice_vks = (_SIDE_KEYS[voice_key] if voice_key in _SIDE_KEYS
                                       else key_to_vk(voice_key),)
                self._solo = SoloKeyTap(voice_key, on_voice)
                self._voice_key = voice_key
            except ValueError as exc:
                log(f"Voice chat hotkey {voice_key!r} disabled: {exc}.")

    def start(self):
        self._thread = threading.Thread(target=self._run, name="windows-hotkeys", daemon=True)
        self._thread.start()
        if not self._ready.wait(5):
            self.stop()
            raise RuntimeError("Windows hotkey thread did not initialize")

    def stop(self):
        self._stopping.set()
        thread = self._thread
        if thread is None or thread is threading.current_thread():
            return
        # Initialization may still be in progress. _run checks _stopping after
        # creating the queue, so a failed post here cannot strand that thread.
        if thread.is_alive():
            _user32.PostThreadMessageW(thread.native_id, WM_QUIT, 0, 0)
            thread.join(2)
        if thread.is_alive():
            raise RuntimeError("Windows hotkey thread did not stop")

    def is_pressed(self):
        # Also catches releases lost across lock/unlock or device removal.
        return (self._registered and bool(_user32.GetAsyncKeyState(self._vk) & 0x8000)
                and all(any(_user32.GetAsyncKeyState(vk) & 0x8000 for vk in keys)
                        for flag, keys in _MODIFIER_KEYS.items() if self._modifiers & flag))

    def _dispatch_hotkey(self, identifier):
        if identifier == HOTKEY_ID and self._registered and not self._stopping.is_set():
            self._on_press()

    def _handle_key(self, key):
        if self._stopping.is_set() or key.VKey == 0xff or key.MakeCode == 0xff:
            return
        vk = key.VKey
        extended = bool(key.Flags & RI_KEY_E0)
        if vk == 0x10:
            vk = 0xa1 if key.MakeCode == 0x36 else 0xa0
        elif vk == 0x11:
            vk = 0xa3 if extended else 0xa2
        elif vk == 0x12:
            vk = 0xa5 if extended else 0xa4
        released = bool(key.Flags & RI_KEY_BREAK)
        if released and (vk == self._vk or any(
                vk in keys for flag, keys in _MODIFIER_KEYS.items() if self._modifiers & flag)):
            self._on_release()
        if self._solo is not None:
            self._solo.handle(SimpleNamespace(
                name=self._voice_key if vk in self._voice_vks else "",
                scan_code=(vk, key.MakeCode, key.Flags & (RI_KEY_E0 | RI_KEY_E1)),
                event_type="up" if released else "down"))

    def _read_input(self, handle):
        size = wintypes.UINT()
        header_size = ctypes.sizeof(RAWINPUTHEADER)
        if _user32.GetRawInputData(handle, RID_INPUT, None, ctypes.byref(size), header_size) != 0:
            raise ctypes.WinError(ctypes.get_last_error())
        buffer = ctypes.create_string_buffer(size.value)
        received = _user32.GetRawInputData(handle, RID_INPUT, buffer, ctypes.byref(size), header_size)
        if received == 0xffffffff:
            raise ctypes.WinError(ctypes.get_last_error())
        if received < header_size + ctypes.sizeof(RAWKEYBOARD):
            return
        header = RAWINPUTHEADER.from_buffer(buffer)
        if header.dwType == 1:  # RIM_TYPEKEYBOARD
            self._handle_key(RAWKEYBOARD.from_buffer(buffer, header_size))

    def _window_proc(self, hwnd, message, wparam, lparam):
        if message == WM_INPUT:
            try:
                self._read_input(lparam)
            except Exception:
                # Never include raw event data in logs, even on decoder errors.
                if not getattr(self, "_input_error", False):
                    self._input_error = True
                    self._log("Windows Raw Input read failed; release polling remains active.")
        return _user32.DefWindowProcW(hwnd, message, wparam, lparam)

    def _run(self):
        hwnd, atom, raw = None, None, False
        instance = None
        class_name = f"DeboraHotkeys_{id(self)}"
        callback = WNDPROC(self._window_proc)
        try:
            instance = _kernel32.GetModuleHandleW(None)
            window_class = WNDCLASS(lpfnWndProc=callback, hInstance=instance,
                                    lpszClassName=class_name)
            atom = _user32.RegisterClassW(ctypes.byref(window_class))
            if not atom:
                raise ctypes.WinError(ctypes.get_last_error())
            hwnd = _user32.CreateWindowExW(0, class_name, "", 0, 0, 0, 0, 0,
                                           wintypes.HWND(-3), None, instance, None)
            if not hwnd:
                raise ctypes.WinError(ctypes.get_last_error())
            device = RAWINPUTDEVICE(1, 6, RIDEV_INPUTSINK, hwnd)
            raw = bool(_user32.RegisterRawInputDevices(ctypes.byref(device), 1, ctypes.sizeof(device)))
            if not raw:
                raise ctypes.WinError(ctypes.get_last_error())
            if self._vk is not None:
                self._registered = bool(_user32.RegisterHotKey(
                    None, HOTKEY_ID, self._modifiers, self._vk))
                if self._registered:
                    self._log(f"Hotkey {self.hotkey} registered.")
                else:
                    error = ctypes.get_last_error()
                    self._log(f"Hotkey {self.hotkey!r} disabled: RegisterHotKey failed "
                              f"({error}: {ctypes.FormatError(error).strip()}); "
                              "another application may own it.")
            if self._solo is not None:
                self._log(f"Voice chat hotkey {self._voice_key} registered (tap it alone).")
            self._ready.set()
            message = wintypes.MSG()
            while not self._stopping.is_set():
                result = _user32.GetMessageW(ctypes.byref(message), None, 0, 0)
                if result == 0:
                    break
                if result == -1:
                    raise ctypes.WinError(ctypes.get_last_error())
                if message.message == WM_HOTKEY:
                    self._dispatch_hotkey(message.wParam)
                else:
                    _user32.DispatchMessageW(ctypes.byref(message))
        except Exception as exc:
            self._log(f"Windows hotkeys {self.hotkey!r} stopped: {exc}.")
        finally:
            if self._registered:
                _user32.UnregisterHotKey(None, HOTKEY_ID)
                self._registered = False
            if raw:
                device = RAWINPUTDEVICE(1, 6, RIDEV_REMOVE, None)
                _user32.RegisterRawInputDevices(ctypes.byref(device), 1, ctypes.sizeof(device))
            if hwnd:
                _user32.DestroyWindow(hwnd)
            if atom:
                _user32.UnregisterClassW(class_name, instance)
            self._ready.set()
