"""Native Windows hotkey contracts without real hooks, windows or key input."""
import ctypes
import queue
import threading
from unittest.mock import Mock, call, patch

import pytest

from debora_whisper import dictation_engine as de
from debora_whisper.platforms import windows_hotkeys as wh
from debora_whisper.platforms.windows import WindowsPlatform


@pytest.mark.parametrize("hotkey,modifiers,vk", [
    ("ctrl+space", wh.MOD_CONTROL, 0x20),
    (" CTRL + Shift + D ", wh.MOD_CONTROL | wh.MOD_SHIFT, ord("D")),
    ("alt+f9", wh.MOD_ALT, 0x78),
    ("windows+7", wh.MOD_WIN, ord("7")),
    ("control+page down", wh.MOD_CONTROL, 0x22),
    ("f24", 0, 0x87),
    ("shift+escape", wh.MOD_SHIFT, 0x1b),
])
def test_hotkey_maps_to_modifiers_and_vk(hotkey, modifiers, vk):
    assert wh.parse_hotkey(hotkey) == (modifiers | wh.MOD_NOREPEAT, vk)


@pytest.mark.parametrize("hotkey,reason", [
    ("ctrl", "non-modifier"), ("ctrl+shift", "non-modifier"),
    ("ctrl+a+b", "non-modifier"), ("ctrl+a, b", "sequences"),
    ("", "one chord"), (None, "one chord"),
    ("ctrl+", "unsupported key"), ("ctrl+not-a-key", "unsupported key"),
    ("ctrl+f25", "unsupported key"), ("right ctrl+space", "left/right"),
    ("ctrl+f12", "reserved"), ("ctrl+/", "unsupported key"),
])
def test_unmappable_hotkeys_fail_clearly(hotkey, reason):
    with pytest.raises(ValueError, match=reason):
        wh.parse_hotkey(hotkey)


@pytest.fixture
def native_api(monkeypatch):
    messages = queue.Queue()
    api = Mock()
    api.RegisterClassW.return_value = 1
    api.CreateWindowExW.return_value = 123
    api.RegisterRawInputDevices.return_value = True
    api.RegisterHotKey.return_value = True
    api.DefWindowProcW.return_value = 0
    api.GetAsyncKeyState.return_value = 0

    def get_message(pointer, *args):
        message, identifier = messages.get(timeout=2)
        if message in (0, -1):
            return message
        pointer._obj.message = message
        pointer._obj.wParam = identifier
        return 1

    def post_message(thread_id, message, *args):
        messages.put((0 if message == wh.WM_QUIT else message, 0))
        return True

    api.GetMessageW.side_effect = get_message
    api.PostThreadMessageW.side_effect = post_message
    monkeypatch.setattr(wh, "_user32", api)
    monkeypatch.setattr(wh, "_kernel32", Mock(GetModuleHandleW=Mock(return_value=42)))
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 1409, raising=False)
    monkeypatch.setattr(ctypes, "FormatError", lambda error: "Hot key already registered", raising=False)
    monkeypatch.setattr(ctypes, "WinError", lambda error: OSError(error, "native failure"), raising=False)
    return api, messages


def service(hotkey="ctrl+space", voice_key="right alt", **callbacks):
    return wh.WindowsHotkeys(hotkey, callbacks.get("on_press", Mock()),
                             callbacks.get("on_release", Mock()), voice_key,
                             callbacks.get("on_voice", Mock()), callbacks.get("log", Mock()))


def key(vk, scan, flags=0):
    return wh.RAWKEYBOARD(MakeCode=scan, VKey=vk, Flags=flags)


def test_owner_thread_registers_dispatches_and_unregisters(native_api):
    api, messages = native_api
    delivered = threading.Event()
    pressed = Mock(side_effect=delivered.set)
    owner_threads = []
    api.RegisterHotKey.side_effect = lambda *args: owner_threads.append(threading.get_ident()) or True
    api.UnregisterHotKey.side_effect = lambda *args: owner_threads.append(threading.get_ident()) or True
    hotkeys = service(on_press=pressed)
    try:
        hotkeys.start()
        api.RegisterHotKey.assert_called_once_with(None, wh.HOTKEY_ID,
                                                   wh.MOD_CONTROL | wh.MOD_NOREPEAT, 0x20)
        messages.put((wh.WM_HOTKEY, wh.HOTKEY_ID))
        assert delivered.wait(2)
        pressed.assert_called_once_with()
    finally:
        hotkeys.stop()
    assert owner_threads == [hotkeys._thread.ident, hotkeys._thread.ident]
    assert owner_threads[0] != threading.get_ident()
    assert not hotkeys._thread.is_alive()
    api.UnregisterHotKey.assert_called_once_with(None, wh.HOTKEY_ID)
    api.DestroyWindow.assert_called_once_with(123)
    api.UnregisterClassW.assert_called_once()
    assert api.RegisterRawInputDevices.call_count == 2
    registration, removal = [entry.args[0]._obj for entry in api.RegisterRawInputDevices.call_args_list]
    assert (registration.usUsagePage, registration.usUsage, registration.dwFlags,
            registration.hwndTarget) == (1, 6, wh.RIDEV_INPUTSINK, 123)
    assert removal.dwFlags == wh.RIDEV_REMOVE and removal.hwndTarget is None
    hotkeys.stop()  # idempotent, no second unregister
    api.UnregisterHotKey.assert_called_once()


def test_wm_hotkey_enqueues_toggle_once(native_api):
    app = de.DictationApp({**de.DEFAULT_CONFIG, "beep_on_start": False})
    app.stop_monitoring()  # Inspect the action queue without a racing consumer.
    app._stopping.clear()
    hotkeys = service(on_press=app.toggle_recording)
    hotkeys._registered = True
    try:
        hotkeys._dispatch_hotkey(999)
        assert app._hotkey_actions.empty()
        hotkeys._dispatch_hotkey(wh.HOTKEY_ID)
        hotkeys._dispatch_hotkey(wh.HOTKEY_ID)  # engine also guards a held press
        callback, args = app._hotkey_actions.get_nowait()
        assert callback == app._handle_hotkey_press and len(args) == 2
        assert app._hotkey_actions.empty()
        hotkeys._stopping.set()
        app._hotkey_held = False
        hotkeys._dispatch_hotkey(wh.HOTKEY_ID)
        assert app._hotkey_actions.empty()
    finally:
        app.stop_monitoring()


@pytest.mark.parametrize("vk,scan,flags", [(0x20, 57, 0), (0x11, 29, 0), (0x11, 29, wh.RI_KEY_E0)])
def test_raw_release_ends_push_to_talk(native_api, vk, scan, flags):
    app = de.DictationApp({**de.DEFAULT_CONFIG, "beep_on_start": False})
    hotkeys = service(on_release=app._release_hotkey)
    app._native_hotkeys = hotkeys
    app._hotkey_held = True
    starter = Mock()
    try:
        with patch.object(de.time, "time", return_value=101.0):
            hotkeys._handle_key(key(vk, scan, flags | wh.RI_KEY_BREAK))
        assert app._hotkey_released_at == 101.0
        with patch.object(app, "_finish_recording") as finish, \
                patch.object(app, "_begin_continuous") as begin:
            app._watch_key(100.0, starter)
        starter.join.assert_called_once()
        finish.assert_called_once_with()
        begin.assert_not_called()
        assert not app._hotkey_held
    finally:
        app.stop_monitoring()


@pytest.mark.parametrize("held", [{0x20, 0xa2}, {0x20, 0xa3}, {0x20}, {0xa2}, set()])
def test_release_poll_uses_physical_modifier_sides(native_api, held):
    api, _ = native_api
    api.GetAsyncKeyState.side_effect = lambda vk: 0x8000 if vk in held else 0
    hotkeys = service()
    hotkeys._registered = True
    assert hotkeys.is_pressed() == (0x20 in held and bool(held & {0xa2, 0xa3}))


def test_poll_release_settles_hold_without_keyboard_listener(native_api):
    app = de.DictationApp({**de.DEFAULT_CONFIG, "beep_on_start": False})
    app._native_hotkeys = service()
    app._native_hotkeys._registered = True
    app._hotkey_held = True
    try:
        with patch.object(de.time, "time", return_value=101.0), \
                patch.object(app, "_finish_recording") as finish, \
                patch("keyboard.is_pressed") as old_poll:
            app._watch_key(100.0, Mock())
        finish.assert_called_once_with()
        old_poll.assert_not_called()
    finally:
        app.stop_monitoring()


def test_unrelated_raw_release_does_not_end_a_hold(native_api):
    released = Mock()
    hotkeys = service(on_release=released)
    hotkeys._handle_key(key(ord("A"), 30, wh.RI_KEY_BREAK))
    hotkeys._handle_key(key(0x20, 57))
    released.assert_not_called()


def test_raw_tap_keeps_tap_action_after_delayed_audio_start(native_api):
    app = de.DictationApp({**de.DEFAULT_CONFIG, "beep_on_start": False})
    app._hotkey_held = True
    app._native_hotkeys = service(on_release=app._release_hotkey)
    try:
        with patch.object(de.time, "time", return_value=100.1):
            app._native_hotkeys._handle_key(key(0x20, 57, wh.RI_KEY_BREAK))
        with patch.object(de.time, "time", return_value=105.0), \
                patch.object(app, "_finish_recording") as finish, \
                patch.object(app, "_begin_continuous") as begin:
            app._watch_key(100.0, Mock())
        begin.assert_called_once_with(100.0)
        finish.assert_not_called()
    finally:
        app.stop_monitoring()


@pytest.mark.parametrize("voice_key", ["right alt", "alt gr"])
@pytest.mark.parametrize("chord", ["solo", "q", "ctrl"])
def test_raw_events_feed_solo_tap_without_suppressing_altgr(native_api, voice_key, chord):
    tapped = Mock()
    hotkeys = service(voice_key=voice_key, on_voice=tapped)
    if chord == "ctrl":
        hotkeys._handle_key(key(0x11, 29))
    hotkeys._handle_key(key(0x12, 56, wh.RI_KEY_E0))
    hotkeys._handle_key(key(0x12, 56, wh.RI_KEY_E0))  # repeat
    if chord == "q":
        hotkeys._handle_key(key(ord("Q"), 16))
        hotkeys._handle_key(key(ord("Q"), 16, wh.RI_KEY_BREAK))
    hotkeys._handle_key(key(0x12, 56, wh.RI_KEY_E0 | wh.RI_KEY_BREAK))
    assert tapped.call_count == (1 if chord == "solo" else 0)


def test_wm_input_decodes_extended_key_and_calls_default_proc(native_api):
    api, _ = native_api
    tapped = Mock()
    hotkeys = service(on_voice=tapped)
    data = b""

    def read(handle, command, buffer, size, header_size):
        assert handle == 77 and command == wh.RID_INPUT
        assert header_size == ctypes.sizeof(wh.RAWINPUTHEADER)
        size._obj.value = len(data)
        if buffer is None:
            return 0
        ctypes.memmove(buffer, data, len(data))
        return len(data)

    api.GetRawInputData.side_effect = read
    for flags in (wh.RI_KEY_E0, wh.RI_KEY_E0 | wh.RI_KEY_BREAK):
        event = key(0x12, 56, flags)
        header = wh.RAWINPUTHEADER(dwType=1, dwSize=ctypes.sizeof(wh.RAWINPUTHEADER) + ctypes.sizeof(event))
        data = bytes(header) + bytes(event)
        hotkeys._window_proc(123, wh.WM_INPUT, 1, 77)
    tapped.assert_called_once_with()
    assert api.DefWindowProcW.call_args_list == [call(123, wh.WM_INPUT, 1, 77)] * 2


def test_raw_read_failure_is_logged_once_without_event_data(native_api):
    api, _ = native_api
    log = Mock()
    hotkeys = service(log=log)
    api.GetRawInputData.side_effect = RuntimeError("private event data")
    for _ in range(2):
        hotkeys._window_proc(123, wh.WM_INPUT, 1, 77)
    log.assert_called_once_with("Windows Raw Input read failed; release polling remains active.")
    assert api.DefWindowProcW.call_count == 2


def test_invalid_voice_key_leaves_main_hotkey_available(native_api):
    api, messages = native_api
    messages.put((0, 0))
    log = Mock()
    hotkeys = service(voice_key="not-a-key", log=log)
    hotkeys._run()
    assert hotkeys._solo is None
    api.RegisterHotKey.assert_called_once()
    assert any("Voice chat hotkey 'not-a-key' disabled" in entry.args[0]
               for entry in log.call_args_list)


@pytest.mark.parametrize("failure", ["parse", "conflict", "raw", "window", "class", "get"])
def test_failures_log_and_release_owned_resources(native_api, failure):
    api, messages = native_api
    log = Mock()
    if failure == "conflict":
        api.RegisterHotKey.return_value = False
    elif failure == "raw":
        api.RegisterRawInputDevices.return_value = False
    elif failure == "window":
        api.CreateWindowExW.return_value = None
    elif failure == "class":
        api.RegisterClassW.return_value = 0
    messages.put((-1 if failure == "get" else 0, 0))
    configured = "ctrl" if failure == "parse" else "ctrl+space"
    hotkeys = service(hotkey=configured, voice_key="", log=log)
    hotkeys._run()
    lines = "\n".join(c.args[0] for c in log.call_args_list)
    assert configured in lines
    assert "disabled" in lines or "stopped" in lines
    if failure == "conflict":
        assert "1409" in lines and "another application" in lines
    assert not hotkeys._registered and hotkeys._ready.is_set()
    assert api.UnregisterHotKey.call_count == (1 if failure == "get" else 0)
    assert api.DestroyWindow.call_count == (0 if failure in ("class", "window") else 1)
    assert api.UnregisterClassW.call_count == (0 if failure == "class" else 1)


def test_settings_rebuild_unregisters_before_registering_new_hotkey(native_api, monkeypatch):
    api, _ = native_api
    provider, keyboard = WindowsPlatform(), Mock()
    monkeypatch.setattr(provider, "keyboard", lambda: keyboard)
    monkeypatch.setattr(de.platforms, "current", lambda: provider)
    config = {**de.DEFAULT_CONFIG, "beep_on_start": False, "voice_chat_hotkey": ""}
    first = de.DictationApp(dict(config))
    second = de.DictationApp({**config, "hotkey": "ctrl+shift+d"})
    try:
        first._start_hotkeys(keyboard)
        assert first.stop_if_idle() is None
        second._start_hotkeys(keyboard)
        second._start_hotkeys(keyboard)
        relevant = [entry for entry in api.mock_calls if entry[0] in ("RegisterHotKey", "UnregisterHotKey")]
        assert relevant == [
            call.RegisterHotKey(None, wh.HOTKEY_ID, wh.MOD_CONTROL | wh.MOD_NOREPEAT, 0x20),
            call.UnregisterHotKey(None, wh.HOTKEY_ID),
            call.RegisterHotKey(None, wh.HOTKEY_ID,
                                wh.MOD_CONTROL | wh.MOD_SHIFT | wh.MOD_NOREPEAT, ord("D")),
        ]
        keyboard.add_hotkey.assert_not_called()
        keyboard.hook.assert_not_called()
        keyboard.is_pressed.assert_not_called()
        keyboard.unhook_all.assert_not_called()
    finally:
        first.stop()
        second.stop()
    assert not first._native_hotkeys._thread.is_alive()
    assert not second._native_hotkeys._thread.is_alive()
