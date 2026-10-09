"""Floating overlay for dictation state: a translucent, slightly rounded
panel that says what the app is doing, in words only.

Uses PIL supersampled rendering via ui.glass for anti-aliased shapes.
"""

import tkinter as tk

from debora_whisper.ui.glass import (
    TRANSPARENT_COLOR, PillCache, composite_on_transparent, pil_to_photo,
    render_pill, _hex_to_rgba,
)

# Color used for window transparency (never appears in UI)
_TRANSPARENT = TRANSPARENT_COLOR


class OverlayWindow:
    """Always-visible floating panel that shows dictation state as text.

    Compact when idle, wider on hover and while recording. A click anywhere
    toggles recording; a drag moves it.
    """

    # --- Dimensions ---
    COMPACT_W = 150
    COMPACT_H = 38
    HOVER_H = 38
    EXPANDED_W = 450
    EXPANDED_H = 38
    RADIUS = 6   # a slightly rounded rectangle, not a capsule
    # Whole-window opacity: Tk has no per-pixel alpha, so text fades too.
    OPACITY = 0.85

    # --- iOS-inspired dark palette ---
    BG = "#0A0A0A"
    BG_HOVER = "#0A0A0A"
    TEXT = "#D8D8DC"      # a slightly gray white, softer than pure white
    TEXT_DIM = "#9A9AA0"
    FONT_SIZE = 11
    # Windows 11's text-optimized Segoe; Segoe UI where it is missing.
    FONTS = (("Segoe UI Variable Text", "Segoe UI Variable Text Semibold"),
             ("Segoe UI", "Segoe UI Semibold"))
    GREEN = "#30D158"
    RED = "#FF453A"
    VIOLET = "#8B5CF6"
    AMBER = "#FF9F0A"
    BLUE = "#0A84FF"
    GRAY = "#48484A"

    # States in which a click toggles recording (speaking: cuts the reply).
    _CLICK_STATES = ("ready", "recording", "speaking")
    # A press that moves less than this (logical px) is a click, not a drag.
    _CLICK_SLOP = 4

    # --- Balloon dimensions ---
    BALLOON_MAX_W = 360
    BALLOON_PAD = 12
    BALLOON_GAP = 20      # gap between pill and balloon
    BALLOON_RADIUS = 6
    BALLOON_FONT_SIZE = 16
    BALLOON_DURATION = 6000  # ms before auto-dismiss

    def __init__(self, root, on_toggle=None, pos_x=None, pos_y=10, on_pos_changed=None):
        """
        Args:
            root: Parent Tk/CTk window.
            on_toggle: Callback() to toggle recording on dot click.
            pos_x: Initial logical center X position.
            pos_y: Initial top Y position.
            on_pos_changed: Callback(x, y) when dragged to a new position.
        """
        self._root = root
        self._on_toggle = on_toggle
        self._on_pos_changed = on_pos_changed
        self._win: tk.Toplevel | None = None
        self._canvas: tk.Canvas | None = None

        # Position is in desktop pixels; dimensions below are logical (96 DPI).
        self._pos_x: int | None = pos_x
        self._pos_y: int = pos_y

        # Animated size
        self._tgt_base_w = float(self.COMPACT_W)
        self._tgt_base_h = float(self.COMPACT_H)
        s = self._scale = self._get_scale()
        self._cur_w = self._tgt_base_w * s
        self._cur_h = self._tgt_base_h * s
        self._tgt_w = self._tgt_base_w * s
        self._tgt_h = self._tgt_base_h * s

        # Drag state
        self._drag_offset_x = 0
        self._drag_offset_y = 0

        # State
        self._state = "loading"
        self._hover = False
        self._result_text = ""

        # Timer IDs
        self._anim_id = None
        self._auto_hide_id = None

        # Balloon
        self._balloon_win: tk.Toplevel | None = None
        self._balloon_id = None
        self._balloon_text = ""
        self._show_balloon = True

        # PIL rendering state
        self._pill_cache = PillCache()
        self._photo_refs: list = []  # prevent GC of PhotoImages
        

        self._build()


    def _get_scale(self) -> float:
        """Read the effective DPI of this window, including Windows display scaling."""
        try:
            import ctypes
            from ctypes import wintypes

            get_dpi = ctypes.windll.user32.GetDpiForWindow
            get_dpi.argtypes = [wintypes.HWND]
            get_dpi.restype = wintypes.UINT
            window = self._win if self._win is not None else self._root
            dpi = get_dpi(window.winfo_id())
            if dpi > 0:
                return dpi / 96.0
        except (AttributeError, OSError, tk.TclError):
            pass
        try:
            dpi = float(self._root.winfo_fpixels("1i"))
            if dpi > 0:
                return dpi / 96.0
        except (AttributeError, ValueError, tk.TclError):
            pass
        return 1.0

    @staticmethod
    def _font_pixels(points, scale):
        # Negative Tk sizes are pixels. Positive sizes would apply Tk's global
        # points-to-pixels scaling AGAIN, which may belong to another monitor.
        return -max(1, round(points * 96 / 72 * scale))

    def _refresh_scale(self, event=None):
        """Resize all artwork together when moving displays or changing DPI."""
        if event is not None and event.widget is not self._win:
            return
        scale = self._get_scale()
        if scale == self._scale:
            return
        ratio = scale / self._scale
        self._scale = scale
        self._cur_w *= ratio
        self._cur_h *= ratio
        self._tgt_w = self._tgt_base_w * scale
        self._tgt_h = self._tgt_base_h * scale
        self._position()
        self._redraw()
        if self._balloon_win is not None:
            self._show_balloon_popup(self._balloon_text)

    # --- Window setup -----------------------------------------------------

    def _build(self):
        self._win = tk.Toplevel(self._root)
        self._win.overrideredirect(True)
        self._win.attributes("-topmost", True)
        self._win.configure(bg=_TRANSPARENT)

        try:
            self._win.attributes("-transparentcolor", _TRANSPARENT)
        except Exception:
            pass  # Non-Windows fallback: square corners
        try:
            self._win.attributes("-alpha", self.OPACITY)
        except Exception:
            pass

        # Re-assert topmost periodically so the Windows 11 taskbar doesn't cover it
        def _force_topmost():
            if self._win and self._win.winfo_exists():
                self._win.attributes("-topmost", True)
                self._win.lift()
                self._refresh_scale()
                self._root.after(500, _force_topmost)

        self._canvas = tk.Canvas(
            self._win, bg=_TRANSPARENT, highlightthickness=0,
            width=self.COMPACT_W, height=self.COMPACT_H,
        )
        self._canvas.pack(fill="both", expand=True)

        self._canvas.bind("<Enter>", lambda e: self._set_hover(True))
        self._canvas.bind("<Leave>", self._on_leave)
        self._canvas.bind("<Motion>", self._on_mouse_move)
        self._canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self._canvas.bind("<B1-Motion>", self._on_drag_move)
        self._canvas.bind("<ButtonRelease-1>", self._on_drag_end)

        self._win.update_idletasks()
        self._position()
        self._win.update_idletasks()
        self._win.bind("<Configure>", self._refresh_scale)
        _force_topmost()
        self._apply_window_flags()
        self._redraw()

    def _apply_window_flags(self):
        """Prevent focus stealing and hide from taskbar (Windows)."""
        try:
            import ctypes
            GWL_EXSTYLE = -20
            WS_EX_NOACTIVATE = 0x08000000
            WS_EX_TOOLWINDOW = 0x00000080
            WS_EX_APPWINDOW = 0x00040000
            WS_EX_TOPMOST = 0x00000008
            hwnd = ctypes.windll.user32.GetParent(self._win.winfo_id())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            style = (style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST) & ~WS_EX_APPWINDOW
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
        except Exception:
            pass

    def _get_screen_width(self) -> int:
        """Get the primary screen width reliably."""
        try:
            import ctypes
            return ctypes.windll.user32.GetSystemMetrics(0)
        except Exception:
            return self._win.winfo_screenwidth()

    def _position(self):
        """Position the window, keeping it centered on its anchor point."""
        w = int(self._cur_w)
        h = int(self._cur_h)
        if self._pos_x is None:
            # Default: center horizontally at top of screen
            x = (self._get_screen_width() - w) // 2
        else:
            # Keep centered on the user's chosen position
            x = self._pos_x - w // 2
        y = self._pos_y

        # Clamp to the working area of the monitor where the window currently belongs
        try:
            import ctypes
            from ctypes import wintypes
            class POINT(ctypes.Structure):
                _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]
            class RECT(ctypes.Structure):
                _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                            ("right", wintypes.LONG), ("bottom", wintypes.LONG)]
            class MONITORINFO(ctypes.Structure):
                _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", RECT),
                            ("rcWork", RECT), ("dwFlags", wintypes.DWORD)]
            
            user32 = ctypes.windll.user32
            user32.MonitorFromPoint.argtypes = [POINT, wintypes.DWORD]
            user32.MonitorFromPoint.restype = wintypes.HANDLE
            user32.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(MONITORINFO)]
            user32.GetMonitorInfoW.restype = wintypes.BOOL
            pt = POINT(x + w // 2, y + h // 2)
            hMonitor = user32.MonitorFromPoint(pt, 2) # MONITOR_DEFAULTTONEAREST
            
            mi = MONITORINFO()
            mi.cbSize = ctypes.sizeof(MONITORINFO)
            if user32.GetMonitorInfoW(hMonitor, ctypes.byref(mi)):
                wl, wt, wr, wb = mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right, mi.rcMonitor.bottom
                if x < wl: x = wl
                if x + w > wr: x = wr - w
                if y < wt: y = wt
                if y + h > wb: y = wb - h
                
                # Update logical position so it doesn't try to escape on next resize
                self._pos_x = x + w // 2
                self._pos_y = y
        except Exception:
            pass

        self._win.geometry(f"{w}x{h}+{x}+{y}")
        self._canvas.configure(width=w, height=h)

    # --- PIL-based drawing ------------------------------------------------
    # All elements are composited onto the pill RGBA image first, then
    # the final result is flattened onto TRANSPARENT_COLOR and placed as
    # one single canvas image.  This avoids the transparent-color
    # punch-through that happens when layering separate images.

    @staticmethod
    def _paste_centered(base, overlay, cx, cy):
        """Alpha-composite overlay onto base, centered at (cx, cy)."""
        ow, oh = overlay.size
        x = cx - ow // 2
        y = cy - oh // 2
        base.alpha_composite(overlay, (x, y))

    def _flat(self) -> dict:
        """render_pill arguments for a flat panel: one color, no border."""
        bg = _hex_to_rgba(self.BG)[:3]
        return {"bg_top": bg, "bg_bottom": bg, "border_width": 0}

    def _font(self, size: int, semibold: bool = False) -> tuple:
        """(family, size) in the most legible installed family."""
        families = getattr(self, "_font_families", None)
        if families is None:
            try:
                import tkinter.font as tkfont
                installed = set(tkfont.families(self._root))
            except Exception:
                installed = set()
            families = next((pair for pair in self.FONTS if pair[0] in installed),
                            self.FONTS[-1])
            self._font_families = families
        return (families[1] if semibold else families[0], size)

    def _label(self) -> tuple[str, str, bool]:
        """(text, color, bold) for the current state."""
        state = self._state
        if state == "loading":
            return "Loading...", self.TEXT_DIM, False
        if state == "ready":
            if self._hover and self._cur_w > (self.COMPACT_W + 20) * self._scale:
                return "Start recording", self.TEXT_DIM, False
            return "Ready", self.TEXT, True
        if state == "recording":
            draft = getattr(self, "_draft_text", "")
            if draft:
                return draft[-40:], self.TEXT, False
            return "", self.TEXT_DIM, False
        if state == "processing":
            return "Transcribing...", self.TEXT, True
        if state == "speaking":
            return "Speaking...", self.TEXT, True
        if state == "result":
            return "Done", self.TEXT, True
        if state == "error":
            return "Error", self.RED, True
        return "", self.TEXT, False

    def _redraw(self):
        from PIL import Image
        c = self._canvas
        c.delete("all")
        self._photo_refs.clear()
        s = self._scale
        font_size = self._font_pixels(self.FONT_SIZE, s)
        w, h = int(self._cur_w), int(self._cur_h)
        r = min(int(self.RADIUS * s), h // 2)
        mid = h // 2

        # Start with glass pill as the base image
        pill = self._pill_cache.get(w, h, radius=r, **self._flat())
        # Work on a copy so the cache stays clean
        frame = pill.copy()

        # Draw the mascot thumbnail on the left
        try:
            if not hasattr(self, "_mascot_img"):
                from PIL import Image
                import os
                path = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "assets", "branding", "mascot_v2_thumbnail.png")
                self._mascot_img = Image.open(path).convert("RGBA")
            
            icon_size = h - int(4 * s) # 2px padding top/bottom
            if getattr(self, "_last_icon_size", 0) != icon_size:
                from PIL import Image, ImageDraw
                img = self._mascot_img.resize((icon_size, icon_size), Image.LANCZOS)
                mask = Image.new('L', (icon_size, icon_size), 0)
                ImageDraw.Draw(mask).ellipse((0, 0, icon_size, icon_size), fill=255)
                circular_img = Image.new('RGBA', (icon_size, icon_size), (0, 0, 0, 0))
                circular_img.paste(img, (0, 0), mask)
                self._mascot_thumb = circular_img
                self._last_icon_size = icon_size
            
            frame.alpha_composite(self._mascot_thumb, (int(2 * s), int(2 * s)))
        except Exception as e:
            pass

        label, fill, bold = self._label()
        text_items = []  # (x, y, text, fill, font, anchor) — drawn after image
        if label:
            font = self._font(font_size, semibold=bold)
            # Shift text center to account for the mascot on the left
            icon_w = h
            remaining_w = w - icon_w
            text_cx = icon_w + remaining_w // 2
            text_items.append((text_cx, mid, label, fill, font, "center"))

        # Flatten to RGB on transparent background and place as one image
        composited = composite_on_transparent(frame)
        photo = pil_to_photo(composited)
        self._photo_refs.append(photo)
        c.create_image(0, 0, image=photo, anchor="nw")

        # Draw text on top (ClearType AA handled by tkinter)
        # Apply a 1-pixel optical correction upwards when scaled to keep it centered
        optical_offset = 1 if s > 1.0 else 0
        for tx, ty, text, fill, font, anchor in text_items:
            c.create_text(tx, ty - optical_offset, text=text, fill=fill, font=font,
                          anchor=anchor)

    # --- Drag to reposition -----------------------------------------------

    def _on_drag_start(self, event):
        """Remember the press; it becomes a click or a drag."""
        self._press_x, self._press_y = event.x, event.y
        self._dragging = False
        self._drag_offset_x = event.x
        self._drag_offset_y = event.y

    def _on_drag_move(self, event):
        """Move window to follow the mouse, once it left the click slop."""
        if not self._dragging:
            slop = self._CLICK_SLOP * self._scale
            if (abs(event.x - self._press_x) <= slop
                    and abs(event.y - self._press_y) <= slop):
                return
            self._dragging = True
        x = self._win.winfo_x() + event.x - self._drag_offset_x
        y = self._win.winfo_y() + event.y - self._drag_offset_y
        w = int(self._cur_w)
        # Store the center x so resizing stays anchored to the drag position
        self._pos_x = x + w // 2
        self._pos_y = y
        self._win.geometry(f"+{x}+{y}")

    def _on_drag_end(self, event):
        """A drag saves the position; a click toggles recording."""
        if getattr(self, "_dragging", False):
            self._dragging = False
            if self._on_pos_changed:
                self._on_pos_changed(self._pos_x, self._pos_y)
        elif self._state in self._CLICK_STATES and self._on_toggle:
            self._on_toggle()

    # --- Hover & cursor ---------------------------------------------------

    def _on_leave(self, event):
        self._canvas.configure(cursor="")
        self._set_hover(False)

    def _on_mouse_move(self, event):
        """Hand cursor wherever a click toggles recording."""
        if self._state in self._CLICK_STATES:
            self._canvas.configure(cursor="hand2")
        else:
            self._canvas.configure(cursor="")

    def _set_hover(self, hovered):
        if self._state != "ready":
            return
        self._hover = hovered
        if hovered:
            self._animate(self.EXPANDED_W, self.HOVER_H)
        else:
            self._animate(self.COMPACT_W, self.COMPACT_H)

    # --- Animation --------------------------------------------------------

    def _animate(self, tw, th):
        self._tgt_base_w = float(tw)
        self._tgt_base_h = float(th)
        if self._anim_id:
            self._root.after_cancel(self._anim_id)
        self._anim_tick()

    def _anim_tick(self):
        s = self._scale
        self._tgt_w = self._tgt_base_w * s
        self._tgt_h = self._tgt_base_h * s
        dw = self._tgt_w - self._cur_w
        dh = self._tgt_h - self._cur_h
        if abs(dw) < 1.5 and abs(dh) < 1.5:
            self._cur_w = self._tgt_w
            self._cur_h = self._tgt_h
            self._position()
            self._redraw()
            self._anim_id = None
            return
        self._cur_w += dw * 0.25
        self._cur_h += dh * 0.25
        self._position()
        self._redraw()
        self._anim_id = self._root.after(16, self._anim_tick)

    # --- Public state API -------------------------------------------------

    def _cancel_timers(self):
        for attr in ("_anim_id", "_auto_hide_id"):
            tid = getattr(self, attr, None)
            if tid:
                self._root.after_cancel(tid)
                setattr(self, attr, None)
        self._dismiss_balloon()

    def show_loading(self):
        """Gray dot — model loading."""
        self._cancel_timers()
        self._state = "loading"
        self._hover = False
        self._animate(self.COMPACT_W, self.COMPACT_H)

    def show_ready(self):
        """Green dot — idle, waiting for hotkey."""
        self._cancel_timers()
        self._state = "ready"
        self._hover = False
        self._animate(self.COMPACT_W, self.COMPACT_H)

    def show_recording(self, draft_text=""):
        """Wider panel: "Listening..." or the draft so far."""
        if self._state != "recording":
            self._cancel_timers()
            self._state = "recording"
            self._animate(self.EXPANDED_W, self.EXPANDED_H)
        self._draft_text = draft_text
        self._redraw()

    def show_processing(self):
        """Amber dot — transcribing."""
        self._cancel_timers()
        self._state = "processing"
        self._animate(self.EXPANDED_W, self.COMPACT_H)

    def show_result(self, text: str):
        """Green dot — transcription done. Text shown in balloon only."""
        self._cancel_timers()
        self._state = "result"
        self._result_text = text
        self._animate(self.COMPACT_W, self.COMPACT_H)
        self._auto_hide_id = self._root.after(2500, self.show_ready)
        if self._show_balloon and text.strip():
            self._show_balloon_popup(text)

    def show_speaking(self, text: str):
        """Blue dot — voice chat speaks its reply (shown in the balloon). It
        stays until the next state: the reply is not done until it is said."""
        if self._state != "speaking":
            self._cancel_timers()
            self._state = "speaking"
            self._animate(self.EXPANDED_W, self.COMPACT_H)
        if self._show_balloon and text.strip():
            self._show_balloon_popup(text)

    def show_notice(self, text: str):
        """A passing message in the balloon; the pill keeps its state."""
        if self._show_balloon and text.strip():
            self._show_balloon_popup(text)

    def show_error(self):
        """Red dot — error state."""
        self._cancel_timers()
        self._state = "error"
        self._animate(self.COMPACT_W, self.COMPACT_H)

    def set_show_balloon(self, enabled: bool):
        """Enable or disable the text balloon under the notch."""
        self._show_balloon = enabled

    def set_balloon_font_size(self, size: int):
        """Set the font size for balloon text."""
        self.BALLOON_FONT_SIZE = max(10, min(size, 32))

    # --- Balloon popup ----------------------------------------------------

    def _show_balloon_popup(self, text: str):
        """Show a dark tooltip-style balloon below the pill with full text."""
        self._dismiss_balloon()
        self._balloon_text = text

        bw = self._balloon_win = tk.Toplevel(self._root)
        bw.overrideredirect(True)
        bw.attributes("-topmost", True)
        bw.configure(bg=_TRANSPARENT)
        try:
            bw.attributes("-transparentcolor", _TRANSPARENT)
        except Exception:
            pass
        try:
            bw.attributes("-alpha", self.OPACITY)
        except Exception:
            pass

        s = self._scale
        pad = int(self.BALLOON_PAD * s)
        r = int(self.BALLOON_RADIUS * s)
        max_w = int(self.BALLOON_MAX_W * s)
        fsize = self._font_pixels(self.BALLOON_FONT_SIZE, s)

        canvas = tk.Canvas(bw, bg=_TRANSPARENT, highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        # Measure text to determine balloon size
        tmp_id = canvas.create_text(
            0, 0, text=text, font=self._font(fsize),
            width=max_w - 2 * pad, anchor="nw",
        )
        bbox = canvas.bbox(tmp_id)
        canvas.delete(tmp_id)

        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        bw_w = text_w + 2 * pad
        bw_h = text_h + 2 * pad
        # Clamp minimum width
        bw_w = max(bw_w, int(120 * s))

        canvas.configure(width=bw_w, height=bw_h)

        # Glass pill background for balloon (PIL-rendered)
        balloon_pill = render_pill(bw_w, bw_h, radius=r, **self._flat())
        composited = composite_on_transparent(balloon_pill)
        photo = pil_to_photo(composited)
        # Store reference to prevent GC
        canvas._photo_ref = photo
        canvas.create_image(0, 0, image=photo, anchor="nw")

        # Draw text
        canvas.create_text(
            pad, pad, text=text, font=self._font(fsize),
            fill=self.TEXT, width=max_w - 2 * pad, anchor="nw",
        )

        # Click anywhere on balloon to dismiss
        canvas.bind("<ButtonPress-1>", lambda e: self._dismiss_balloon())

        # Position below the pill
        pill_x = self._win.winfo_x()
        pill_y = self._win.winfo_y()
        pill_w = int(self._cur_w)
        pill_h = int(self._cur_h)
        bx = pill_x + (pill_w - bw_w) // 2
        by = pill_y + pill_h + int(self.BALLOON_GAP * s)
        bw.geometry(f"{bw_w}x{bw_h}+{bx}+{by}")

        # Apply no-focus flags
        try:
            import ctypes
            GWL_EXSTYLE = -20
            WS_EX_NOACTIVATE = 0x08000000
            WS_EX_TOOLWINDOW = 0x00000080
            WS_EX_APPWINDOW = 0x00040000
            hwnd = ctypes.windll.user32.GetParent(bw.winfo_id())
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            style = (style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW) & ~WS_EX_APPWINDOW
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
        except Exception:
            pass

        # Auto-dismiss
        self._balloon_id = self._root.after(self.BALLOON_DURATION, self._dismiss_balloon)

    def _dismiss_balloon(self):
        """Destroy the balloon popup if it exists."""
        self._balloon_text = ""
        if self._balloon_id:
            self._root.after_cancel(self._balloon_id)
            self._balloon_id = None
        if self._balloon_win:
            try:
                self._balloon_win.destroy()
            except Exception:
                pass
            self._balloon_win = None

    def hide(self):
        """Dynamic Island is always visible — hide means go to ready."""
        self.show_ready()
