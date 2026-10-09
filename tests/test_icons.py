"""Tests for icon generation."""

import pytest
from PIL import Image

from debora_whisper.ui.icons import (
    icon_loading, icon_ready, icon_recording, icon_processing, icon_error,
    icon_speaking, STATE_ICONS, ICON_SIZE,
)


class TestIconGeneration:
    """Verify each icon returns a 64x64 RGBA image."""

    @pytest.mark.parametrize("fn", [
        icon_loading, icon_ready, icon_recording, icon_processing, icon_error,
        icon_speaking,
    ])
    def test_icon_size_and_mode(self, fn):
        img = fn()
        assert isinstance(img, Image.Image)
        assert img.size == (ICON_SIZE, ICON_SIZE)
        assert img.mode == "RGBA"

    def test_state_icons_dict_has_all_states(self):
        expected = {"loading", "ready", "recording", "processing", "error", "speaking"}
        assert set(STATE_ICONS.keys()) == expected

    def test_state_icons_callable(self):
        for name, fn in STATE_ICONS.items():
            img = fn()
            assert isinstance(img, Image.Image), f"STATE_ICONS['{name}'] did not return an Image"

    def test_icons_are_not_fully_transparent(self):
        """Each icon should have some non-transparent pixels."""
        for name, fn in STATE_ICONS.items():
            img = fn()
            alpha = img.split()[3]  # Alpha channel
            assert alpha.getextrema()[1] > 0, f"Icon '{name}' is fully transparent"

    def test_error_icon_differs_from_recording(self):
        """Error icon (X overlay) should differ from plain recording icon."""
        rec = icon_recording()
        err = icon_error()
        assert rec.tobytes() != err.tobytes()


def test_each_bars_glow_leaves_the_previous_bar_whole():
    """Each bar is composited on its own layer: drawn straight onto the icon,
    the next bar's glow overwrote the right edge of the one before it."""
    from debora_whisper.ui.icons import render_bars
    size = 256
    img = render_bars("#06B6D4", [1.0] * 5, size=size)
    bar_w, spacing = size * 0.12, size * 0.06
    start_x = (size - (5 * bar_w + 4 * spacing)) / 2
    for i in range(5):
        right_edge = start_x + i * (bar_w + spacing) + bar_w
        r, g, b, a = img.getpixel((int(right_edge - bar_w * 0.25), size // 2))
        assert a > 200 and g > 150, f"bar {i} cut at its right edge: {(r, g, b, a)}"
