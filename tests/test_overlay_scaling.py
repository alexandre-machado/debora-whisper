"""DPI regressions: monitor changes and Tk's independent font scaling."""

import ctypes
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

tk = pytest.importorskip("tkinter")
from npu_whisper.ui.overlay import OverlayWindow


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


@pytest.mark.parametrize("scale", [1.0, 1.25, 1.5, 2.0])
def test_dot_hit_area_tracks_its_rendered_size(scale):
    overlay = OverlayWindow.__new__(OverlayWindow)
    overlay._scale = scale
    overlay._state = "ready"
    overlay._on_toggle = Mock()
    overlay._on_drag_start(SimpleNamespace(x=38 * scale, y=10))
    overlay._on_toggle.assert_called_once()
    overlay._on_drag_start(SimpleNamespace(x=38 * scale + 1, y=10))
    assert not overlay._drag_is_click
    overlay._on_toggle.assert_called_once()


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
