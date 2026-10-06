"""System tray icon manager using pystray."""

import threading
import time
import pystray
from ui.icons import get_icon, get_volume_icon


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
        
        # Animation state
        self._animating = False
        self._anim_frame = 0
        self._anim_thread: threading.Thread | None = None

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
        
        self._check_animation()

    def _animation_loop(self):
        """Background loop to update the icon for animated states."""
        while self._animating and self._icon:
            if not getattr(self._icon, "visible", True):
                time.sleep(0.1)
                continue
                
            self._anim_frame += 1
            try:
                self._icon.icon = get_icon(self._state, self._anim_frame)
            except Exception:
                pass
            time.sleep(0.15)  # 150ms per frame

    def _check_animation(self):
        """Start or stop the animation loop based on the current state."""
        should_animate = self._state in ("loading", "processing")
        
        if should_animate and not self._animating:
            self._animating = True
            self._anim_frame = 0
            self._anim_thread = threading.Thread(target=self._animation_loop, daemon=True)
            self._anim_thread.start()
        elif not should_animate and self._animating:
            self._animating = False
            self._anim_thread = None

    def update_state(self, state_name: str, tooltip: str | None = None):
        """Update tray icon and tooltip for a new state."""
        self._state = state_name
        if tooltip:
            self._tooltip = tooltip

        self._check_animation()

        if self._icon and getattr(self._icon, "visible", True):
            try:
                self._icon.icon = get_icon(self._state, 0)
                self._icon.title = self._tooltip
                # Force menu rebuild so dynamic text updates
                self._icon.menu = self._build_menu()
                self._icon.update_menu()
            except Exception:
                pass

    def update_audio_level(self, level: float):
        """Update the icon dynamically based on audio volume level."""
        if self._state == "recording" and self._icon and getattr(self._icon, "visible", True):
            try:
                self._icon.icon = get_volume_icon(level)
            except Exception:
                pass

    def update_info(self, device: str, model: str, hotkey: str):
        """Update the device/model/hotkey shown in the menu."""
        self._device = device
        self._model = model
        self._hotkey = hotkey

    def stop(self):
        """Stop the tray icon."""
        self._animating = False
        if self._icon:
            try:
                self._icon.stop()
            except Exception:
                pass
