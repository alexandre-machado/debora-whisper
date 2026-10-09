"""DPI regressions: monitor changes and Tk's independent font scaling."""

import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

tk = pytest.importorskip("tkinter")
from debora_whisper.ui.overlay import OverlayWindow


@pytest.mark.parametrize("dpi", [96, 120, 144, 192, 240])
def test_scale_uses_window_dpi_and_pointer_sized_handle(monkeypatch, dpi):
    get_dpi = Mock(return_value=dpi)
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(
        user32=SimpleNamespace(GetDpiForWindow=get_dpi)), raising=False)
    overlay = OverlayWindow.__new__(OverlayWindow)
    # A handle wider than 32 bits must survive the ctypes call.
    overlay._win = Mock()
    overlay._win.winfo_id.return_value = 0x123456789
    overlay._root = Mock()

    assert overlay._get_scale() == dpi / 96
    get_dpi.assert_called_once_with(0x123456789)
    assert ctypes.sizeof(get_dpi.argtypes[0]) == ctypes.sizeof(ctypes.c_void_p)
    overlay._root.winfo_fpixels.assert_not_called()


@pytest.mark.parametrize("failure", [None, OSError("DPI unavailable")])
def test_invalid_dpi_falls_back_to_tk(monkeypatch, failure):
    get_dpi = Mock(return_value=0, side_effect=failure)
    monkeypatch.setattr(ctypes, "windll", SimpleNamespace(
        user32=SimpleNamespace(GetDpiForWindow=get_dpi)), raising=False)
    overlay = OverlayWindow.__new__(OverlayWindow)
    overlay._win = None
    overlay._root = Mock()
    overlay._root.winfo_fpixels.return_value = 144
    assert overlay._get_scale() == 1.5
    overlay._root.winfo_fpixels.return_value = 0
    assert overlay._get_scale() == 1.0


@pytest.mark.parametrize("scale", [1.0, 1.25, 1.5, 2.0, 2.5])
def test_fonts_use_pixels_with_one_dpi_conversion(scale):
    assert OverlayWindow._font_pixels(16, scale) == -round(16 * 96 / 72 * scale)


def test_monitor_change_rescales_even_without_animation():
    overlay = OverlayWindow.__new__(OverlayWindow)
    overlay._scale = 2.0
    overlay._cur_w, overlay._cur_h = 300, 76
    overlay._tgt_base_w, overlay._tgt_base_h = 450, 38
    overlay._get_scale = Mock(return_value=1.25)
    overlay._position = Mock()
    overlay._redraw = Mock()
    overlay._balloon_win = object()
    overlay._balloon_text = "Visible transcription"
    overlay._show_balloon_popup = Mock()

    overlay._refresh_scale()

    assert (overlay._cur_w, overlay._cur_h) == (187.5, 47.5)
    assert (overlay._tgt_w, overlay._tgt_h) == (562.5, 47.5)
    overlay._show_balloon_popup.assert_called_once_with("Visible transcription")
    overlay._refresh_scale()
    overlay._redraw.assert_called_once()
    overlay._position.assert_called_once()


def _clickable_overlay(scale, state="ready"):
    overlay = OverlayWindow.__new__(OverlayWindow)
    overlay._scale = scale
    overlay._state = state
    overlay._on_toggle = Mock()
    overlay._on_pos_changed = Mock()
    overlay._cur_w = 150 * scale
    overlay._win = Mock(winfo_x=Mock(return_value=100), winfo_y=Mock(return_value=10))
    return overlay


@pytest.mark.parametrize("scale", [1.0, 1.25, 1.5, 2.0])
@pytest.mark.parametrize("x", [2, 75, 148])
def test_a_click_anywhere_toggles(scale, x):
    overlay = _clickable_overlay(scale)
    press = SimpleNamespace(x=x * scale, y=10)
    overlay._on_drag_start(press)
    # A small jitter within the slop is still a click.
    overlay._on_drag_move(SimpleNamespace(x=x * scale + 3 * scale, y=10))
    overlay._on_drag_end(press)
    overlay._on_toggle.assert_called_once()
    overlay._on_pos_changed.assert_not_called()


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_a_drag_moves_without_toggling(scale):
    overlay = _clickable_overlay(scale)
    overlay._on_drag_start(SimpleNamespace(x=20, y=10))
    overlay._on_drag_move(SimpleNamespace(x=20 + 30 * scale, y=10))
    overlay._on_drag_end(SimpleNamespace(x=20 + 30 * scale, y=10))
    overlay._on_toggle.assert_not_called()
    overlay._on_pos_changed.assert_called_once()


@pytest.mark.parametrize("state", ["loading", "processing", "error"])
def test_a_click_does_nothing_while_busy(state):
    overlay = _clickable_overlay(1.0, state)
    overlay._on_drag_start(SimpleNamespace(x=20, y=10))
    overlay._on_drag_end(SimpleNamespace(x=20, y=10))
    overlay._on_toggle.assert_not_called()


@pytest.mark.parametrize("state, draft, label", [
    ("ready", "", "Ready"),
    ("recording", "", ""),  # the mascot alone says it is listening
    ("recording", "ola tudo bem", "ola tudo bem"),
    ("processing", "", "Transcribing..."),
    ("speaking", "", "Speaking..."),
    ("error", "", "Error"),
])
def test_states_are_told_in_words_only(state, draft, label):
    overlay = _clickable_overlay(1.0, state)
    overlay._hover = False
    overlay._draft_text = draft
    assert overlay._label()[0] == label


def test_text_is_a_soft_white_in_a_legible_font(monkeypatch):
    import tkinter.font as tkfont
    overlay = OverlayWindow.__new__(OverlayWindow)
    overlay._root = None
    assert OverlayWindow.TEXT.upper() != "#FFFFFF"
    monkeypatch.setattr(tkfont, "families", lambda root=None: ["Segoe UI", "Segoe UI Semibold",
                                                                "Segoe UI Variable Text",
                                                                "Segoe UI Variable Text Semibold"])
    assert overlay._font(14) == ("Segoe UI Variable Text", 14)
    assert overlay._font(14, semibold=True) == ("Segoe UI Variable Text Semibold", 14)


def test_font_falls_back_to_segoe_ui(monkeypatch):
    import tkinter.font as tkfont
    overlay = OverlayWindow.__new__(OverlayWindow)
    overlay._root = None
    monkeypatch.setattr(tkfont, "families", lambda root=None: ["Segoe UI", "Arial"])
    assert overlay._font(14, semibold=True) == ("Segoe UI Semibold", 14)


def test_panel_is_flat():
    from debora_whisper.ui.glass import render_pill
    overlay = OverlayWindow.__new__(OverlayWindow)
    img = render_pill(150, 38, radius=OverlayWindow.RADIUS, **overlay._flat())
    # One color edge to edge: no border, gradient or highlight.
    inner = img.crop((8, 2, 142, 36)).convert("RGB")
    assert len(set(inner.getdata())) == 1
    assert img.getpixel((75, 0))[:3] == img.getpixel((75, 19))[:3]


def test_panel_is_translucent_and_slightly_rounded():
    assert 0.5 < OverlayWindow.OPACITY < 1.0
    assert OverlayWindow.RADIUS < OverlayWindow.COMPACT_H // 4
    assert OverlayWindow.BALLOON_RADIUS == OverlayWindow.RADIUS


@pytest.fixture
def tk_root(monkeypatch):
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    # Exercise real Tk measurement/rendering without showing test windows.
    toplevel = tk.Toplevel

    def hidden_toplevel(*args, **kwargs):
        window = toplevel(*args, **kwargs)
        window.withdraw()
        return window

    monkeypatch.setattr(tk, "Toplevel", hidden_toplevel)
    yield root
    for timer in root.tk.call("after", "info"):
        root.after_cancel(timer)
    root.destroy()


@pytest.mark.parametrize("scale", [1.0, 1.25, 1.5, 2.0])
def test_real_tk_text_and_balloon_ignore_other_monitors_font_scale(
        tk_root, monkeypatch, scale):
    monkeypatch.setattr(OverlayWindow, "_get_scale", lambda self: scale)
    overlay = OverlayWindow(tk_root)
    assert overlay._cur_w == 150 * scale
    assert overlay._cur_h == 38 * scale
    text = "Texto de exemplo para testar a proporção do balão em outra tela."
    measurements = []
    for tk_scale in (96 / 72, 192 / 72):
        tk_root.tk.call("tk", "scaling", tk_scale)
        overlay._redraw()
        text_id = next(item for item in overlay._canvas.find_all()
                       if overlay._canvas.type(item) == "text")
        pill_bbox = overlay._canvas.bbox(text_id)
        assert pill_bbox[3] - pill_bbox[1] < overlay._cur_h
        overlay._show_balloon_popup(text)
        canvas = overlay._balloon_win.winfo_children()[0]
        text_id = next(item for item in canvas.find_all()
                       if canvas.type(item) == "text")
        bbox = canvas.bbox(text_id)
        width, height = int(canvas.cget("width")), int(canvas.cget("height"))
        assert 0 <= bbox[0] < bbox[2] <= width
        assert 0 <= bbox[1] < bbox[3] <= height
        measurements.append((pill_bbox, bbox, width, height))
    assert measurements[0] == measurements[1]
    overlay._dismiss_balloon()


def test_speaking_stays_until_the_next_state_and_shows_the_reply():
    overlay = OverlayWindow.__new__(OverlayWindow)
    overlay._state = "processing"
    overlay._show_balloon = True
    overlay._cancel_timers = Mock()
    overlay._animate = Mock()
    overlay._show_balloon_popup = Mock()
    overlay._root = Mock()

    overlay.show_speaking("Oi,")
    overlay.show_speaking("Oi, tudo bem!")

    assert overlay._state == "speaking"
    overlay._cancel_timers.assert_called_once()
    overlay._animate.assert_called_once()
    overlay._root.after.assert_not_called()  # no auto-return to Ready
    assert overlay._show_balloon_popup.call_args.args == ("Oi, tudo bem!",)


def test_mascot_ships_inside_the_package():
    from PIL import Image
    from debora_whisper.ui.overlay import MASCOT_PATH
    assert "debora_whisper" in MASCOT_PATH.parts
    with Image.open(MASCOT_PATH) as img:
        assert img.size[0] == img.size[1] >= 64


@pytest.mark.parametrize("clip", ["loop", "zoom"])
def test_mascot_clips_ship_and_load(clip):
    from debora_whisper.ui import overlay as ov
    path = ov.MASCOT_LOOP_PATH if clip == "loop" else ov.MASCOT_ZOOM_PATH
    overlay = OverlayWindow.__new__(OverlayWindow)
    assert "debora_whisper" in path.parts
    frames = overlay._mascot_frames(34, radius=6, clip=clip)
    assert len(frames) > 1
    assert all(f.size == (60, 34) and f.mode == "RGBA" for f in frames)  # 16:9
    assert frames[0].getpixel((0, 0))[3] < 16  # outside the rounded corner


def test_mascot_falls_back_to_the_still_image(monkeypatch, tmp_path):
    import debora_whisper.ui.overlay as ov
    monkeypatch.setattr(ov, "MASCOT_LOOP_PATH", tmp_path / "missing.webp")
    overlay = OverlayWindow.__new__(OverlayWindow)
    frames = overlay._mascot_frames(34)
    assert len(frames) == 1 and frames[0].size == (60, 34)  # square still, cropped


@pytest.mark.parametrize("state, animated", [
    ("recording", True), ("speaking", True),
    ("ready", False), ("processing", False), ("loading", False), ("error", False),
])
def test_mascot_moves_only_while_listening_or_speaking(state, animated):
    overlay = OverlayWindow.__new__(OverlayWindow)
    overlay._root = Mock()
    overlay._root.after.return_value = "timer"
    overlay._state = state
    overlay._mascot_index = 5
    overlay._mascot_clip = "loop"
    overlay._mascot_anim_id = None
    overlay._sync_mascot_animation()
    assert overlay._root.after.called is animated
    if not animated:
        assert overlay._mascot_index == 0


def test_leaving_an_animated_state_stops_the_timer():
    overlay = OverlayWindow.__new__(OverlayWindow)
    overlay._root = Mock()
    overlay._state = "ready"
    overlay._mascot_index = 7
    overlay._mascot_clip = "zoom"
    overlay._mascot_anim_id = "timer"
    overlay._sync_mascot_animation()
    overlay._root.after_cancel.assert_called_once_with("timer")
    assert overlay._mascot_anim_id is None and overlay._mascot_index == 0
    assert overlay._mascot_clip == "loop"


def test_speaking_repeats_the_zoom():
    overlay = OverlayWindow.__new__(OverlayWindow)
    overlay._state = "recording"
    overlay._show_balloon = False
    overlay._cancel_timers = Mock()
    overlay._animate = Mock()
    overlay._redraw = Mock()
    overlay.show_speaking("Oi!")
    assert (overlay._mascot_clip, overlay._mascot_index) == ("zoom", 0)
    zoom_len = len(overlay._mascot_frames(1, clip="zoom"))
    for _ in range(zoom_len):
        overlay._mascot_tick()
    assert (overlay._mascot_clip, overlay._mascot_index) == ("zoom", 0)
    # More of the reply does not restart the zoom.
    overlay._mascot_tick()
    overlay.show_speaking("Oi! Tudo bem?")
    assert (overlay._mascot_clip, overlay._mascot_index) == ("zoom", 1)


def test_mascot_edge_fades_into_the_panel():
    overlay = OverlayWindow.__new__(OverlayWindow)
    hard = overlay._mascot_frames(34, radius=6)[0]
    soft = overlay._mascot_frames(34, feather=2, radius=6)[0]
    # The center stays opaque; the rim is fainter than a hard edge's.
    assert soft.getpixel((30, 17))[3] == 255
    rim = [(30, 0), (0, 17), (59, 17), (30, 33)]
    assert sum(soft.getpixel(p)[3] for p in rim) < sum(hard.getpixel(p)[3] for p in rim) / 2
