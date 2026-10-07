"""System tray icon manager using pystray."""

import threading
import pystray
from npu_whisper.ui.icons import get_icon, get_volume_icon


class TrayManager:
    """Manages the system tray icon and context menu."""

    def __init__(self, on_toggle, on_quit, on_settings=None, on_history=None, on_hardware_event=None,
                 device="NPU", model="base", hotkey="ctrl+alt+d"):
        self._on_toggle = on_toggle
        self._on_quit = on_quit
        self._on_settings = on_settings
        self._on_history = on_history
        self._on_hardware_event = on_hardware_event
        self._device = device
        self._model = model
        self._hotkey = hotkey
        self._state = "loading"
        self._tooltip = "NPU Dictation — Loading..."
        self._icon: pystray.Icon | None = None
        self._thread: threading.Thread | None = None
        
        # All pystray mutations happen on _render_thread. Setting the icon,
        # title or menu calls Shell_NotifyIcon, which can block on Explorer;
        # done from Tk callbacks it kept the main thread inside a callback,
        # where Tk swallows Ctrl+C's KeyboardInterrupt. Public methods only
        # record what to show and wake the renderer.
        self._level = 0.0
        self._anim_frame = 0
        self._wake = threading.Event()
        self._running = False
        self._render_thread: threading.Thread | None = None

    def _build_menu(self):
        """Build the right-click context menu with dynamic state text."""
        items = [
            pystray.MenuItem(
                lambda _: "Stop Recording" if self._state == "recording" else "Start Recording",
                self._on_toggle_click,
                default=True,
                enabled=lambda _: self._state in ("ready", "recording"),
            ),
            pystray.Menu.SEPARATOR,
        ]

        if self._on_history:
            items.append(pystray.MenuItem("History", lambda: self._on_history()))

        if self._on_settings:
            items.append(pystray.MenuItem("Settings", lambda: self._on_settings()))

        items.extend([
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                lambda _: f"Device: {self._device}",
                None,
                enabled=False,
            ),
            pystray.MenuItem(
                lambda _: f"Model: whisper-{self._model}",
                None,
                enabled=False,
            ),
            pystray.MenuItem(
                lambda _: f"Hotkey: {self._hotkey}",
                None,
                enabled=False,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._on_quit_click),
        ])

        return pystray.Menu(*items)

    def _on_toggle_click(self, icon=None, item=None):
        self._on_toggle()

    def _on_quit_click(self, icon=None, item=None):
        self._on_quit()

    def start(self):
        """Start the tray icon in a daemon thread."""
        initial_icon = get_icon(self._state)
        self._icon = pystray.Icon(
            name="npu-dictation",
            icon=initial_icon,
            title=self._tooltip,
            menu=self._build_menu(),
        )

        import sys
        if sys.platform == "win32" and hasattr(self._icon, "_message_handlers"):
            WM_POWERBROADCAST = 0x021B
            PBT_APMRESUMEAUTOMATIC = 0x0012
            WM_DEVICECHANGE = 0x0219

            def _on_power_broadcast(wparam, lparam):
                if wparam == PBT_APMRESUMEAUTOMATIC and self._on_hardware_event:
                    self._on_hardware_event()
                return 1

            def _on_device_change(wparam, lparam):
                # 0x8000: DEVICEARRIVAL, 0x8004: DEVICEREMOVECOMPLETE, 0x0007: DEVNODES_CHANGED
                if wparam in (0x8000, 0x8004, 0x0007) and self._on_hardware_event:
                    self._on_hardware_event()
                return 1

            self._icon._message_handlers[WM_POWERBROADCAST] = _on_power_broadcast
            self._icon._message_handlers[WM_DEVICECHANGE] = _on_device_change

        self._thread = threading.Thread(target=self._icon.run, daemon=True)
        self._thread.start()

        self._running = True
        self._render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self._render_thread.start()

    def _render_loop(self):
        """Apply state, tooltip and icon frames to pystray. Only Windows calls
        whose result changed are made: a silent mic or a repeated state costs
        nothing."""
        shown_image = shown_title = shown_state = None
        while self._running:
            self._wake.clear()
            state, title = self._state, self._tooltip
            if state == "recording":
                image = get_volume_icon(self._level)
                interval = 0.1
            elif state in ("loading", "processing"):
                self._anim_frame += 1
                image = get_icon(state, self._anim_frame)
                interval = 0.15
            else:
                image = get_icon(state)
                interval = None  # static: sleep until the next update

            icon = self._icon
            if icon is not None and getattr(icon, "visible", True):
                try:
                    if image is not shown_image:
                        icon.icon = image
                        shown_image = image
                    if title != shown_title:
                        icon.title = title
                        shown_title = title
                    if state != shown_state:
                        # Rebuild so the dynamic menu text follows the state.
                        icon.menu = self._build_menu()
                        icon.update_menu()
                        shown_state = state
                except Exception:
                    pass  # icon torn down underneath us during shutdown
            else:
                interval = 0.1  # not shown yet: retry shortly

            self._wake.wait(interval)

    def update_state(self, state_name: str, tooltip: str | None = None):
        """Record a new state/tooltip; the render thread applies it."""
        if state_name != self._state:
            self._anim_frame = 0
        self._state = state_name
        if tooltip:
            self._tooltip = tooltip
        self._wake.set()

    def update_audio_level(self, level: float):
        """Record the mic level; the render thread turns it into icon frames
        while recording."""
        self._level = level

    def update_info(self, device: str, model: str, hotkey: str):
        """Update the device/model/hotkey shown in the menu."""
        self._device = device
        self._model = model
        self._hotkey = hotkey

    def stop(self):
        """Stop the tray icon."""
        self._running = False
        self._wake.set()
        if self._icon:
            try:
                self._icon.stop()
            except Exception:
                pass
