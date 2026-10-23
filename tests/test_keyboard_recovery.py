"""Keyboard recovery contracts without installing hooks or inspecting real tokens."""
import ctypes
import threading
import queue
from ctypes import wintypes
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from debora_whisper import dictation_engine as de
from debora_whisper.platforms import windows


@pytest.mark.parametrize("missing", [False, True])
def test_recover_stop_rebuild_delivers_hotkey_once(monkeypatch, keyboard, missing):
    import sys

    events = queue.Queue()
    callbacks = []
    threads = []

    def listen(backend, listener):
        threads.append(threading.current_thread())
        while events.get() != "quit":
            for callback in list(callbacks):
                callback()

    def register(key, callback, **kwargs):
        callbacks.append(callback)
        return lambda: callbacks.remove(callback)

    monkeypatch.setattr(windows, "_listen_safely", listen)
    monkeypatch.setattr(windows, "_post_keyboard_message",
                        lambda thread, message: events.put("quit"))
    keyboard._listener.listening_thread.is_alive.side_effect = None
    keyboard._listener.listening_thread.is_alive.return_value = False
    keyboard.parse_hotkey = lambda key: (((29,), (57,)),)
    keyboard.hook = lambda *args, **kwargs: lambda: None
    keyboard.add_hotkey = register
    keyboard.unhook_all = callbacks.clear
    monkeypatch.setitem(sys.modules, "keyboard", keyboard)
    provider = windows.WindowsPlatform()
    monkeypatch.setattr(provider, "keyboard_watchdog", lambda *args: None)
    monkeypatch.setattr(de.platforms, "current", lambda: provider)
    config = {**de.DEFAULT_CONFIG, "voice_chat_hotkey": "", "beep_on_start": False}
    first, second = de.DictationApp(config), de.DictationApp(config)
    delivered = threading.Event()
    second.toggle_recording = Mock(side_effect=delivered.set)
    try:
        assert first._recover_hotkeys()
        recovered = keyboard._listener.listening_thread
        first.stop()
        assert not recovered.is_alive()
        if missing:
            del keyboard._listener.listening_thread
        second._start_hotkeys(keyboard)
        replacement = keyboard._listener.listening_thread
        second._start_hotkeys(keyboard)
        events.put("key")
        assert delivered.wait(2)
        second.toggle_recording.assert_called_once_with()
        assert threads == [recovered, replacement]
        assert replacement.is_alive()
    finally:
        first.stop_monitoring()
        second.stop()


def test_final_shutdown_wakes_legacy_listener_without_replacement(monkeypatch, keyboard):
    original = threading.excepthook
    old = keyboard._listener.listening_thread

    def post(thread_id, message, *args):
        assert threading.excepthook is not original
        assert thread_id == old.native_id
        return 1

    windows._user32.PostThreadMessageW.side_effect = post
    windows.WindowsPlatform().stop_keyboard_listener(keyboard, final=True)
    assert threading.excepthook is original
    assert keyboard._listener.listening_thread is old
    assert windows._user32.PostThreadMessageW.call_args_list == [
        call(42, 0, 0, 0), call(42, 0x12, 0, 0)]
    old.join.assert_called_once_with(1.0)


@pytest.fixture
def keyboard(monkeypatch):
    old = Mock(native_id=42)
    old.is_alive.side_effect = [True, False]
    listener = SimpleNamespace(lock=threading.Lock(), listening_thread=old,
                               active_modifiers=set(), modifier_states={},
                               direct_callback=Mock())
    backend = SimpleNamespace(SetWindowsHookEx=Mock(return_value=123),
                              UnhookWindowsHookEx=Mock())
    api = Mock()
    api.GetMessageW.return_value = 0
    monkeypatch.setattr(windows, "_user32", api)
    monkeypatch.setattr(windows, "_keyboard_hook_args",
                        Mock(return_value=((13, object(), None, 0), {})))
    monkeypatch.setattr(windows, "_self_elevated", lambda: True)
    return SimpleNamespace(_listener=listener, _os_keyboard=backend,
                           _pressed_events_lock=threading.Lock(),
                           _pressed_events={}, _logically_pressed_keys={})


@pytest.mark.parametrize("failure", [None, "post", "join", "hung"])
def test_legacy_exception_filter_is_thread_scoped_and_restored(monkeypatch, keyboard, failure):
    old = keyboard._listener.listening_thread
    original = Mock()
    monkeypatch.setattr(threading, "excepthook", original)
    unrelated = SimpleNamespace(thread=object(), exc_type=OSError)
    unexpected = SimpleNamespace(thread=old, exc_type=ValueError)

    def post(*args):
        threading.excepthook(SimpleNamespace(thread=old, exc_type=OSError))
        threading.excepthook(unrelated)
        threading.excepthook(unexpected)
        if failure == "post":
            raise OSError("post failed")
        return True

    windows._user32.PostThreadMessageW.side_effect = post
    if failure == "join":
        old.join.side_effect = RuntimeError("join failed")
    if failure == "hung":
        old.is_alive.side_effect = [True, True]
    stop = threading.Event()
    stop.set()
    if failure:
        with pytest.raises((OSError, RuntimeError)):
            windows.WindowsPlatform().restart_keyboard_listener(keyboard, stop)
    else:
        windows.WindowsPlatform().restart_keyboard_listener(keyboard, stop)
    assert threading.excepthook is original
    assert original.call_args_list == [call(unrelated), call(unexpected)] * (1 if failure == "post" else 2)


def test_replacement_shutdown_posts_only_quit_without_exception_filter(monkeypatch, keyboard):
    old = keyboard._listener.listening_thread
    old._debora_message_loop = True
    original = threading.excepthook

    def post(*args):
        assert threading.excepthook is original
        return True

    windows._user32.PostThreadMessageW.side_effect = post
    stop = threading.Event()
    stop.set()
    windows.WindowsPlatform().restart_keyboard_listener(keyboard, stop)
    windows._user32.PostThreadMessageW.assert_called_once_with(42, 0x12, 0, 0)


@pytest.mark.parametrize("message", [0, 0x12])
@pytest.mark.parametrize("ready", [True, False])
def test_post_retries_until_queue_exists_or_timeout(monkeypatch, keyboard, message, ready):
    elapsed = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        elapsed[0] += seconds

    monkeypatch.setattr(windows, "time", SimpleNamespace(
        monotonic=lambda: elapsed[0], sleep=sleep))
    post = windows._user32.PostThreadMessageW
    post.side_effect = [0, 0, 1] if ready else None
    post.return_value = 0
    assert windows._post_keyboard_message(keyboard._listener.listening_thread, message) is ready
    assert 0.02 <= elapsed[0] <= 0.51
    if not ready:
        assert elapsed[0] >= 0.5
    assert sleeps == [0.01] * len(sleeps)
    assert post.call_args_list == [call(42, message, 0, 0)] * (len(sleeps) + 1)


@pytest.mark.parametrize("kind", ["legacy", "unstarted", "dead"])
def test_shutdown_leaves_unowned_or_inactive_listener_alone(keyboard, kind):
    thread = keyboard._listener.listening_thread
    if kind == "unstarted":
        del keyboard._listener.listening_thread
    elif kind == "dead":
        thread._debora_message_loop = True
        thread.is_alive.side_effect = None
        thread.is_alive.return_value = False
    windows.WindowsPlatform().stop_keyboard_listener(keyboard)
    windows._user32.PostThreadMessageW.assert_not_called()
    thread.join.assert_not_called()


def test_shutdown_exits_owned_loop_and_unhooks(keyboard):
    entered, quit = threading.Event(), threading.Event()

    def get(*args):
        entered.set()
        assert quit.wait(2)
        return 0

    def post(thread_id, message, *args):
        assert thread_id == thread.native_id and message == 0x12
        quit.set()
        return 1

    windows._user32.GetMessageW.side_effect = get
    windows._user32.PostThreadMessageW.side_effect = post
    thread = threading.Thread(target=windows._listen_safely,
                              args=(keyboard._os_keyboard, keyboard._listener), daemon=True)
    thread._debora_message_loop = True
    keyboard._listener.listening_thread = thread
    thread.start()
    try:
        assert entered.wait(2)
        windows.WindowsPlatform().stop_keyboard_listener(keyboard)
        assert not thread.is_alive()
        keyboard._os_keyboard.UnhookWindowsHookEx.assert_called_once_with(123)
    finally:
        quit.set()
        thread.join(2)


def test_restart_leaves_backend_modifier_flags_untouched(keyboard):
    backend = keyboard._os_keyboard
    backend.shift_is_pressed = backend.altgr_is_pressed = True
    backend.ignore_next_right_alt = True
    stop = threading.Event()
    stop.set()
    windows.WindowsPlatform().restart_keyboard_listener(keyboard, stop)
    assert backend.shift_is_pressed and backend.altgr_is_pressed and backend.ignore_next_right_alt


@pytest.mark.parametrize("failure", [None, "get", "dispatch", "install"])
def test_message_loop_uses_real_buffer_and_unhooks_on_exit(keyboard, monkeypatch, failure):
    api, backend = windows._user32, keyboard._os_keyboard
    seen = []

    def get(message, window, minimum, maximum):
        assert isinstance(message._obj, wintypes.MSG)
        seen.append(message._obj)
        message._obj.message = 0x12 if len(seen) == 2 else 0x100
        return -1 if failure == "get" else (1 if len(seen) == 1 else 0)

    api.GetMessageW.side_effect = get
    monkeypatch.setattr(ctypes, "WinError", lambda code: OSError("win32 failed"), raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)
    if failure == "dispatch":
        api.DispatchMessageW.side_effect = OSError("dispatch failed")
    if failure == "install":
        backend.SetWindowsHookEx.return_value = 0
    if failure:
        with pytest.raises(OSError):
            windows._listen_safely(backend, keyboard._listener)
    else:
        windows._listen_safely(backend, keyboard._listener)
        assert seen[0] is seen[1]
        api.TranslateMessage.assert_called_once()
        api.DispatchMessageW.assert_called_once()
    if failure == "install":
        backend.UnhookWindowsHookEx.assert_not_called()
        api.GetMessageW.assert_not_called()
    else:
        backend.UnhookWindowsHookEx.assert_called_once_with(123)


def test_listener_reuses_callback_and_resets_decoder_modifiers(keyboard):
    listener, backend = keyboard._listener, keyboard._os_keyboard
    windows._listen_safely(backend, listener)
    first_callback = backend.SetWindowsHookEx.call_args.args[1]
    listener._debora_hook_globals.update(shift_is_pressed=True, altgr_is_pressed=True,
                                         ignore_next_right_alt=True)
    windows._listen_safely(backend, listener)
    assert backend.SetWindowsHookEx.call_args.args[1] is first_callback
    windows._keyboard_hook_args.assert_called_once_with(backend, listener.direct_callback)
    assert not any(listener._debora_hook_globals.values())
    assert backend.UnhookWindowsHookEx.call_count == 2


def test_builder_captures_callback_without_native_install_or_atexit_registration():
    # Model the package's nested decoder and CFUNCTYPE creation, without a hook.
    namespace = dict(SetWindowsHookEx=Mock(), atexit=Mock(),
                     LowLevelKeyboardProc=ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int),
                     UnhookWindowsHookEx=Mock(), shift_is_pressed=False)
    exec("""
def prepare_intercept(callback):
    def handler(value):
        global shift_is_pressed
        shift_is_pressed = True
        return callback(value)
    native_callback = LowLevelKeyboardProc(handler)
    SetWindowsHookEx(13, native_callback, None, 0)
    atexit.register(UnhookWindowsHookEx, native_callback)
""", namespace)
    backend = SimpleNamespace(prepare_intercept=namespace["prepare_intercept"])
    args, decoder_globals = windows._keyboard_hook_args(backend, lambda value: value + 1)
    assert args[0] == 13
    assert args[1](4) == 5
    assert decoder_globals["shift_is_pressed"] is True
    assert namespace["shift_is_pressed"] is False
    namespace["SetWindowsHookEx"].assert_not_called()
    namespace["atexit"].register.assert_not_called()


def test_successful_recovery_emits_exactly_one_log_line(keyboard, monkeypatch):
    log = Mock()
    monkeypatch.setattr(de, "log", log)
    provider, now = windows.WindowsPlatform(), [0.0]

    def start(**kwargs):
        return Mock(start=lambda: kwargs["target"](*kwargs["args"]))

    monkeypatch.setattr(threading, "Thread", start)
    stop = threading.Event()

    def recover():
        provider.restart_keyboard_listener(keyboard, stop)
        return True

    keyboard._listener.listening_thread.join.side_effect = lambda timeout: threading.excepthook(
        SimpleNamespace(thread=keyboard._listener.listening_thread, exc_type=OSError))
    watchdog = windows.KeyboardHookWatchdog(stop, recover, log, clock=lambda: now[0],
                                           probe=lambda: (0.0, True))
    for now[0] in (4.0, 5.0):
        watchdog.poll()
    log.assert_called_once_with("Keyboard hook lost; hotkeys re-registered.")


@pytest.mark.parametrize("self_elevated,foreground,skipped", [
    (False, True, True), (True, True, False), (None, True, False), (False, False, False),
])
def test_watchdog_skips_only_known_elevated_foreground(monkeypatch, self_elevated, foreground, skipped):
    monkeypatch.setattr(windows, "_self_elevated", lambda: self_elevated)
    query = Mock(return_value=foreground)
    monkeypatch.setattr(windows, "_foreground_elevated", query)
    now, recover = [0.0], Mock(return_value=True)
    watchdog = windows.KeyboardHookWatchdog(threading.Event(), recover, Mock(),
                                           clock=lambda: now[0], probe=lambda: (0.0, True))
    watchdog._suspect = (3.0, 0.0)
    now[0] = 4.0
    watchdog.poll()
    assert recover.call_count == (0 if skipped else 1)
    assert watchdog._suspect is None
    if self_elevated is not False:
        query.assert_not_called()
    if skipped:
        query.return_value = False
        now[0] = 5.0
        watchdog.poll()
        recover.assert_not_called()  # Must start a fresh confirmation interval.
        now[0] = 6.0
        watchdog.poll()
        recover.assert_called_once()


@pytest.mark.parametrize("failure", [None, "process", "token", "info", "exception"])
@pytest.mark.parametrize("elevated", [False, True])
def test_token_elevation_closes_all_acquired_handles(monkeypatch, failure, elevated):
    kernel, advapi = Mock(), Mock()
    # Pointer-sized handle values catch accidental truncation in signatures.
    process_handle, token_handle = 0x123456789, 0x234567890
    kernel.OpenProcess.return_value = 0 if failure == "process" else process_handle

    def open_token(process, access, token):
        token._obj.value = token_handle
        return failure != "token"

    def info(token, kind, value, size, returned):
        assert token.value == token_handle
        assert kind == 20
        assert size == ctypes.sizeof(wintypes.DWORD)
        if failure == "exception":
            raise OSError("token failed")
        value._obj.value = elevated
        returned._obj.value = size
        return failure != "info"

    advapi.OpenProcessToken.side_effect = open_token
    advapi.GetTokenInformation.side_effect = info
    monkeypatch.setattr(windows, "_kernel32", kernel)
    monkeypatch.setattr(windows, "_advapi32", advapi)
    assert windows._process_elevated(321) is (None if failure else elevated)
    kernel.OpenProcess.assert_called_once_with(0x1000, False, 321)
    closed = [args.args[0].value if isinstance(args.args[0], wintypes.HANDLE)
              else args.args[0] for args in kernel.CloseHandle.call_args_list]
    assert closed == ([] if failure == "process" else [process_handle] if failure == "token"
                      else [token_handle, process_handle])


def test_self_elevation_is_cached(monkeypatch):
    query = Mock(return_value=False)
    monkeypatch.setattr(windows, "_process_elevated", query)
    windows._self_elevated.cache_clear()
    try:
        assert windows._self_elevated() is False
        assert windows._self_elevated() is False
        query.assert_called_once()
    finally:
        windows._self_elevated.cache_clear()


@pytest.mark.parametrize("failure", [None, "window", "pid", "zero_pid", "token", "exception"])
def test_foreground_query_falls_back_on_failure(monkeypatch, failure):
    api = Mock()
    api.GetForegroundWindow.return_value = 0 if failure == "window" else 123

    def pid(window, result):
        assert window == 123
        if failure == "exception":
            raise OSError("window disappeared")
        result._obj.value = 0 if failure == "zero_pid" else 321
        return 0 if failure == "pid" else 456

    api.GetWindowThreadProcessId.side_effect = pid
    query = Mock(return_value=None if failure == "token" else True)
    monkeypatch.setattr(windows, "_user32", api)
    monkeypatch.setattr(windows, "_process_elevated", query)
    assert windows._foreground_elevated() is (failure is None)
    if failure in (None, "token"):
        query.assert_called_once_with(321)
    else:
        query.assert_not_called()


@pytest.mark.skipif(windows._kernel32 is None, reason="Windows API declarations")
def test_native_api_signatures_preserve_handle_and_message_widths():
    assert windows._kernel32.OpenProcess.restype is wintypes.HANDLE
    assert windows._kernel32.CloseHandle.argtypes == [wintypes.HANDLE]
    assert windows._advapi32.OpenProcessToken.argtypes == [
        wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    assert windows._advapi32.GetTokenInformation.argtypes == [
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD)]
    assert windows._user32.GetMessageW.argtypes[0] == ctypes.POINTER(wintypes.MSG)
