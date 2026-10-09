"""One translucent window for the mascot and the current conversation.

Uses PIL supersampled rendering via ui.glass for anti-aliased shapes.
"""

import tkinter as tk
from pathlib import Path
from time import monotonic

from debora_whisper.ui.glass import (
    TRANSPARENT_COLOR, PillCache, composite_on_transparent, pil_to_photo,
    _hex_to_rgba,
)

# Débora's face, shown at the panel's left edge.
MASCOT_PATH = Path(__file__).parent / "assets" / "mascot.png"
# Her animated bust, 12 fps, 228x128, cut from a generated video: a calm
# loop (played forward then back, so it has no seam) and a zoom into her face
# and back, repeated while recording or speaking.
MASCOT_LOOP_PATH = Path(__file__).parent / "assets" / "mascot_loop.webp"
MASCOT_ZOOM_PATH = Path(__file__).parent / "assets" / "mascot_zoom.webp"

# Color used for window transparency (never appears in UI)
_TRANSPARENT = TRANSPARENT_COLOR


class OverlayWindow:
    """A fixed mascot anchor with an optional, clipped conversation to its right."""

    # --- Dimensions ---
    COMPACT_W = 64
    COMPACT_H = 38
    RADIUS = 6   # a slightly rounded rectangle, not a capsule
    # Whole-window opacity: Tk has no per-pixel alpha, so text fades too.
    OPACITY = 0.85

    # --- iOS-inspired dark palette ---
    BG = "#0A0A0A"
    TEXT = "#D8D8DC"      # a slightly gray white, softer than pure white
    TEXT_DIM = "#9A9AA0"
    # Windows 11's text-optimized Segoe; Segoe UI where it is missing.
    FONTS = (("Segoe UI Variable Text", "Segoe UI Variable Text Semibold"),
             ("Segoe UI", "Segoe UI Semibold"))
    GREEN = "#30D158"
    RED = "#FF453A"
    VIOLET = "#8B5CF6"
    AMBER = "#FF9F0A"
    BLUE = "#0A84FF"
    GRAY = "#48484A"

    # The mascot moves only while she listens or speaks.
    _MASCOT_ANIMATED_STATES = ("recording", "speaking")
    MASCOT_FPS = 12
    MASCOT_FEATHER = 2  # logical px of haze where the mascot meets the panel
    MASCOT_ASPECT = 16 / 9  # the whole video frame, uncropped

    # States in which a click toggles recording (speaking: cuts the reply).
    _CLICK_STATES = ("ready", "recording", "speaking")
    # A press that moves less than this (logical px) is a click, not a drag.
    _CLICK_SLOP = 4

    # --- Balloon dimensions ---
    BALLOON_DEFAULT_W = 360
    BALLOON_MIN_W = 200
    BALLOON_PAD = 6
    BALLOON_GAP = 8
    BALLOON_FONT_SIZE = 16
    BALLOON_DURATION = 2500  # the former result -> ready auto-dismiss delay
    SLIDE_SECONDS = 0.2

    def __init__(self, root, on_toggle=None, pos_x=None, pos_y=10, on_pos_changed=None,
                 balloon_width=None, on_width_changed=None):
        """
        Args:
            root: Parent Tk/CTk window.
            on_toggle: Callback() to toggle recording on mascot click.
            pos_x: Initial mascot center X in desktop pixels.
            pos_y: Initial top Y position.
            on_pos_changed: Callback(x, y) when dragged to a new position.
            balloon_width: Preferred text area width in logical pixels, or None.
            on_width_changed: Callback(width) after resizing the text area.
        """
        self._root = root
        self._on_toggle = on_toggle
        self._on_pos_changed = on_pos_changed
        self._on_width_changed = on_width_changed
        self._balloon_width = balloon_width
        self._win: tk.Toplevel | None = None
        self._canvas: tk.Canvas | None = None

        # Position is in desktop pixels; dimensions below are logical (96 DPI).
        self._pos_x: int | None = pos_x
        self._pos_y: int = pos_y

        s = self._scale = self._get_scale()
        self._cur_w = self.COMPACT_W * s
        self._cur_h = self.COMPACT_H * s
        self._window_x = None

        # Drag state
        self._drag_offset_x = 0
        self._drag_offset_y = 0

        # State
        self._state = "loading"
        self._mascot_clip = "loop"
        self._mascot_index = 0
        self._mascot_anim_id = None
        self._user_text = ""
        self._reply_text = ""
        self._voice_turn = False
        self._turn_finished = False
        self._text_dismissed = False
        self._balloon_id = None
        self._show_balloon = True
        self._slide_id = None
        self._slides = {}

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
        self._scale = scale
        self._reset_slides()
        self._position()
        self._redraw()

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
        # A child canvas is a real clipping viewport, not a second Toplevel.
        # It has the same flat glass color and inherits the window's opacity.
        self._text_canvas = tk.Canvas(
            self._canvas, bg=self.BG, highlightthickness=0, borderwidth=0)
        self._text_canvas.bind("<ButtonPress-1>", lambda e: self._dismiss_balloon())
        self._text_canvas.configure(cursor="hand2")

        self._canvas.bind("<Leave>", self._on_leave)
        self._canvas.bind("<Motion>", self._on_mouse_move)
        self._canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self._canvas.bind("<B1-Motion>", self._on_drag_move)
        self._canvas.bind("<ButtonRelease-1>", self._on_drag_end)

        self._win.update_idletasks()
        self._restore_position()
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

    def _monitor_work_area(self, x, y, full=False):
        """Find the monitor containing the mascot's anchor.
        full: the whole monitor, taskbar included, not just the work area."""
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
            pt = POINT(x, y)
            hMonitor = user32.MonitorFromPoint(pt, 2) # MONITOR_DEFAULTTONEAREST
            
            mi = MONITORINFO()
            mi.cbSize = ctypes.sizeof(MONITORINFO)
            if user32.GetMonitorInfoW(hMonitor, ctypes.byref(mi)):
                rect = mi.rcMonitor if full else mi.rcWork
                return rect.left, rect.top, rect.right, rect.bottom
        except Exception:
            pass
        return 0, 0, self._win.winfo_screenwidth(), self._win.winfo_screenheight()

    def _mascot_geometry(self):
        """Use exactly the same rounded pixel dimensions as the drawn clip."""
        pad = round(2 * self._scale)
        height = round(self.COMPACT_H * self._scale) - 2 * pad
        right = pad + round(height * self.MASCOT_ASPECT)
        return pad, height, right, right + pad

    def _text_geometry(self, width):
        mascot_right = self._mascot_geometry()[2]
        gap = round(self.BALLOON_GAP * self._scale)
        bar_width = max(1, round(self._scale))
        bar_left = width - round(3 * self._scale) - bar_width
        text_left = mascot_right + gap
        text_right = max(text_left, bar_left - gap)
        return text_left, text_right, bar_left, bar_width

    def _conversation_lines(self):
        if not self._show_balloon or self._text_dismissed:
            return []
        if self._voice_turn:
            lines = [("user", f"Você: {self._user_text}", self.TEXT_DIM)]
            if self._reply_text:
                lines.append(("reply", f"Débora: {self._reply_text}", self.TEXT))
            return lines if self._user_text or self._reply_text else []
        return [("user", self._user_text, self.TEXT)] if self._user_text else []

    def _text_font(self):
        import tkinter.font as tkfont
        return tkfont.Font(root=self._root, font=self._font(
            self._font_pixels(self.BALLOON_FONT_SIZE, self._scale)))

    def _restore_position(self):
        # The root may be on another monitor. First move this window, then
        # restore the saved mascot center using the destination monitor's DPI.
        center, top = self._pos_x, self._pos_y
        self._position()
        self._win.update_idletasks()
        scale = self._get_scale()
        if scale != self._scale:
            self._scale = scale
            self._window_x = None
            self._pos_x, self._pos_y = center, top
            self._position()

    def _position(self):
        """Only the mascot is the position anchor; text never recenters it."""
        mascot_w = self._mascot_geometry()[3]
        mascot_h = round(self.COMPACT_H * self._scale)
        if self._window_x is None:
            center = self._get_screen_width() // 2 if self._pos_x is None else self._pos_x
            self._window_x = center - mascot_w // 2
        x, y = self._window_x, self._pos_y
        wl, wt, wr, wb = self._monitor_work_area(
            x + mascot_w // 2, y + mascot_h // 2, full=True)
        x = max(wl, min(x, wr - mascot_w))
        y = max(wt, min(y, wb - mascot_h))
        self._window_x, self._pos_x, self._pos_y = x, x + mascot_w // 2, y
        w, h = mascot_w, mascot_h
        lines = self._conversation_lines()
        if lines:
            requested = self._balloon_width or self.BALLOON_DEFAULT_W
            w += min(round(requested * self._scale), max(0, wr - x - mascot_w))
            h = max(h, len(lines) * self._text_font().metrics("linespace")
                    + 2 * round(self.BALLOON_PAD * self._scale))
        self._cur_w, self._cur_h = w, h
        # +negative coordinates are absolute desktop positions in Tk geometry.
        self._win.geometry(f"{w}x{h}+{x}+{y}")
        self._canvas.configure(width=w, height=h)

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

    def _redraw(self):
        c = self._canvas
        c.delete("all")
        self._photo_refs.clear()
        s = self._scale
        w, h = int(self._cur_w), int(self._cur_h)
        frame = self._pill_cache.get(w, h, radius=round(self.RADIUS * s),
                                     **self._flat()).copy()
        self._sync_mascot_animation()
        pad, mascot_h, _, _ = self._mascot_geometry()
        frames = self._mascot_frames(mascot_h, max(1, round(self.MASCOT_FEATHER * s)),
                                     max(1, round(self.RADIUS * s)), clip=self._mascot_clip)
        if frames:
            frame.alpha_composite(frames[self._mascot_index % len(frames)], (pad, pad))
        photo = pil_to_photo(composite_on_transparent(frame))
        self._photo_refs.append(photo)
        c.create_image(0, 0, image=photo, anchor="nw")
        left, right, bar, bar_w = self._text_geometry(w)
        if self._conversation_lines() and right > left:
            c.create_rectangle(bar, round(6 * s), bar + bar_w, h - round(6 * s),
                               fill="#FFFFFF", outline="")
            self._text_canvas.place(x=left, y=round(self.BALLOON_PAD * s),
                                    width=right - left,
                                    height=h - 2 * round(self.BALLOON_PAD * s))
            self._draw_text(right - left)
        else:
            self._text_canvas.place_forget()

    @staticmethod
    def _line_target(line_width, visible_width):
        return min(0, visible_width - line_width)

    def _slide_offset(self, key, target, now):
        slide = self._slides.get(key)
        if slide is None:
            slide = (0.0, 0.0, now)
        start, end, began = slide
        progress = min(1.0, max(0.0, (now - began) / self.SLIDE_SECONDS))
        current = start + (end - start) * (1 - (1 - progress) ** 3)
        if target != end:
            # Retarget from the current position, even if another sentence
            # arrived before the last animation finished.
            slide = (current, target, now)
        self._slides[key] = slide
        return current

    def _draw_text(self, visible_width):
        canvas = self._text_canvas
        canvas.delete("all")
        font = self._text_font()
        line_h = font.metrics("linespace")
        now = monotonic()
        moving = False
        for row, (key, text, fill) in enumerate(self._conversation_lines()):
            # Newlines in a transcript/reply never create additional rows.
            text = " ".join(text.split())
            item = canvas.create_text(0, row * line_h, text=text, fill=fill,
                                      font=font, anchor="nw")
            bbox = canvas.bbox(item)
            width = bbox[2] - bbox[0]
            target = self._line_target(width, visible_width)
            offset = self._slide_offset(key, target, now)
            # Tk text bboxes include font bearings: normalize their left edge
            # before applying the offset, so fitting text starts at text_left.
            canvas.move(item, offset - bbox[0], 0)
            moving |= abs(offset - target) > 0.01
        if moving and self._slide_id is None:
            self._slide_id = self._root.after(16, self._slide_tick)

    def _slide_tick(self):
        self._slide_id = None
        left, right, _, _ = self._text_geometry(int(self._cur_w))
        if self._conversation_lines() and right > left:
            self._draw_text(right - left)

    def _reset_slides(self):
        if self._slide_id is not None:
            self._root.after_cancel(self._slide_id)
            self._slide_id = None
        self._slides.clear()

    def _update_layout(self):
        self._position()
        self._redraw()

    # --- Drag, resize and click -------------------------------------------

    def _hit_region(self, x):
        _, _, bar, bar_w = self._text_geometry(int(self._cur_w))
        if self._conversation_lines() and self._cur_w > self._mascot_geometry()[3]:
            if bar - 4 * self._scale <= x <= bar + bar_w + 4 * self._scale:
                return "resize"
            if x >= self._mascot_geometry()[2]:
                return "text"
        return "mascot"

    def _on_drag_start(self, event):
        self._press_region = self._hit_region(event.x)
        self._press_x, self._press_y = event.x, event.y
        self._dragging = False
        self._drag_offset_x, self._drag_offset_y = event.x, event.y
        self._resize_start = (self._cur_w - self._mascot_geometry()[3]) / self._scale

    def _on_drag_move(self, event):
        if self._press_region == "text":
            return
        if not self._dragging:
            slop = self._CLICK_SLOP * self._scale
            if (abs(event.x - self._press_x) <= slop
                    and abs(event.y - self._press_y) <= slop):
                return
            self._dragging = True
        if self._press_region == "resize":
            requested = self._resize_start + (event.x - self._press_x) / self._scale
            mascot_w = self._mascot_geometry()[3]
            _, _, right, _ = self._monitor_work_area(self._pos_x, self._pos_y, full=True)
            maximum = max(self.BALLOON_MIN_W,
                          (right - self._window_x - mascot_w) / self._scale)
            self._balloon_width = round(max(self.BALLOON_MIN_W, min(requested, maximum)))
        else:
            self._window_x = self._win.winfo_x() + event.x - self._drag_offset_x
            self._pos_y = self._win.winfo_y() + event.y - self._drag_offset_y
        self._update_layout()

    def _on_drag_end(self, event):
        if self._dragging:
            self._dragging = False
            if self._press_region == "resize":
                if self._on_width_changed:
                    self._on_width_changed(self._balloon_width)
            elif self._on_pos_changed:
                self._on_pos_changed(self._pos_x, self._pos_y)
        elif self._press_region == "text":
            self._dismiss_balloon()
        elif (self._press_region == "mascot" and self._state in self._CLICK_STATES
              and self._on_toggle):
            self._on_toggle()

    def _on_leave(self, event):
        self._canvas.configure(cursor="")

    def _on_mouse_move(self, event):
        region = self._hit_region(event.x)
        cursor = "sb_h_double_arrow" if region == "resize" else (
            "hand2" if region == "text" or self._state in self._CLICK_STATES else "")
        self._canvas.configure(cursor=cursor)

    # --- Mascot -----------------------------------------------------------

    def _mascot_frames(self, height: int, feather: int = 0, radius: int = 0,
                       clip: str = "loop") -> list:
        """A mascot clip's frames as rounded rectangles height px tall in the
        video's 16:9, whose edge fades out over about feather px (a haze into
        the panel), cached per size. The loop falls back to the still image
        (center-cropped to 16:9); a clip that does not load has no frames."""
        cache = self.__dict__.setdefault("_mascot_cache", {})
        key = (clip, height, feather, radius)
        if key in cache:
            return cache[key]
        from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageSequence
        all_sources = self.__dict__.setdefault("_mascot_sources", {})
        sources = all_sources.get(clip)
        if sources is None:
            sources = []
            paths = (MASCOT_LOOP_PATH, MASCOT_PATH) if clip == "loop" else (MASCOT_ZOOM_PATH,)
            for path in paths:
                try:
                    with Image.open(path) as img:
                        sources = [f.convert("RGBA") for f in ImageSequence.Iterator(img)]
                    break
                except Exception:
                    continue
            all_sources[clip] = sources
        frames = []
        if sources and height > 0:
            width = round(height * self.MASCOT_ASPECT)
            ss = 4  # supersampled, for a smooth edge
            inset = feather * ss
            mask = Image.new("L", (width * ss, height * ss), 0)
            ImageDraw.Draw(mask).rounded_rectangle(
                (inset, inset, width * ss - 1 - inset, height * ss - 1 - inset),
                radius=radius * ss, fill=255)
            if feather:
                mask = mask.filter(ImageFilter.GaussianBlur(inset / 2))
            mask = mask.resize((width, height), Image.LANCZOS)
            for src in sources:
                thumb = ImageOps.fit(src, (width, height), Image.LANCZOS)
                thumb.putalpha(mask)
                frames.append(thumb)
        cache[key] = frames
        return frames

    def _sync_mascot_animation(self):
        """Run the mascot's frame timer only in the animated states; the
        other states show the loop's first frame and cost nothing."""
        animate = self._state in self._MASCOT_ANIMATED_STATES
        running = self.__dict__.get("_mascot_anim_id")
        if animate and not running:
            self._mascot_anim_id = self._root.after(
                1000 // self.MASCOT_FPS, self._mascot_tick)
        elif not animate:
            if running:
                self._root.after_cancel(running)
                self._mascot_anim_id = None
            self._mascot_clip = "loop"
            self._mascot_index = 0

    def _mascot_tick(self):
        self._mascot_anim_id = None
        self._mascot_index += 1
        if self._mascot_index >= len(self._mascot_frames(1, clip=self._mascot_clip)):
            # Recording and speaking both repeat the zoom (the user liked it
            # better than the calmer loop).
            self._mascot_index = 0
        self._redraw()  # reschedules through _sync_mascot_animation

    # --- Public state API -------------------------------------------------

    def _cancel_dismiss(self):
        if self._balloon_id is not None:
            self._root.after_cancel(self._balloon_id)
            self._balloon_id = None

    def _finish_turn(self):
        self._turn_finished = True
        if self._balloon_id is None and not self._text_dismissed:
            self._balloon_id = self._root.after(self.BALLOON_DURATION, self._dismiss_balloon)

    def _new_turn(self, voice_chat=False):
        self._cancel_dismiss()
        self._reset_slides()
        self._user_text = self._reply_text = ""
        self._voice_turn = voice_chat
        self._turn_finished = False
        self._text_dismissed = False

    def _set_state(self, state):
        if state != self._state and state in self._MASCOT_ANIMATED_STATES:
            self._mascot_clip, self._mascot_index = "zoom", 0
        self._state = state

    def show_loading(self):
        self._set_state("loading")
        self._update_layout()

    def show_ready(self):
        if self._user_text or self._reply_text:
            self._finish_turn()
        self._set_state("ready")
        self._update_layout()

    def show_recording(self, draft_text="", voice_chat=False):
        # Continuous listening resumes with an empty draft after TTS. Keep
        # that turn through its normal delay, until actual new speech arrives.
        finishing = self._voice_turn and self._state in ("speaking", "processing")
        if finishing and not draft_text:
            self._finish_turn()
        elif draft_text and (self._turn_finished or self._state != "recording"):
            self._new_turn(voice_chat)
        elif self._state != "recording" and (not self._turn_finished or not self._voice_turn):
            self._new_turn(voice_chat)
        self._set_state("recording")
        if draft_text:
            self._user_text = draft_text
        self._update_layout()

    def show_processing(self, text="", voice_chat=False):
        self._set_state("processing")
        if text.strip():
            if self._turn_finished:
                self._new_turn(voice_chat)
            self._voice_turn = voice_chat
            self._user_text = text
            self._cancel_dismiss()
        self._update_layout()

    def show_result(self, text):
        if self._voice_turn and not self._turn_finished:
            self._reply_text = text
        else:
            if self._turn_finished:
                self._new_turn()
            self._user_text = text
        self._set_state("ready")
        self._finish_turn()
        self._update_layout()

    def show_speaking(self, text):
        self._cancel_dismiss()
        self._voice_turn = True
        self._reply_text = text
        self._set_state("speaking")
        self._update_layout()

    def show_notice(self, text):
        """Notices belong to the tray; never overwrite conversation content."""

    def show_error(self):
        if self._state in ("speaking", "processing") and (self._user_text or self._reply_text):
            self._finish_turn()
        self._set_state("error")
        self._update_layout()

    def set_show_balloon(self, enabled):
        """The Settings checkbox controls the in-window text area."""
        self._show_balloon = enabled
        self._update_layout()

    def set_balloon_font_size(self, size):
        self.BALLOON_FONT_SIZE = max(10, min(size, 32))
        self._reset_slides()
        self._update_layout()

    def _dismiss_balloon(self):
        # Retain the turn internally so another streamed sentence cannot
        # reopen text that the user just dismissed.
        self._cancel_dismiss()
        self._text_dismissed = True
        self._reset_slides()
        self._update_layout()

    def hide(self):
        self.show_ready()
