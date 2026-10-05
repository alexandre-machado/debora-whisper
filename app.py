"""GUI orchestrator for NPU Dictation Engine.

Entry point for tray-icon mode. Wires the engine, system tray, overlay,
settings dialog, onboarding wizard, and transcription history together.

Usage:
    python app.py [--device NPU|GPU|CPU] [--model base|small|medium|turbo|parakeet]
"""

import sys
import argparse
import customtkinter as ctk

from dictation_engine import (
    AppState, DictationApp, MODEL_REGISTRY, MODEL_DIR,
    load_config, save_config, validate_config, log, create_model,
    is_model_downloaded, device_failure,
)
from ui.tray import TrayManager
from ui.overlay import OverlayWindow
from ui.settings import SettingsWindow
from ui.history import HistoryWindow
from ui.onboarding import OnboardingWindow


class GUIApp:
    """Main GUI application wiring all components."""

    def __init__(self, config: dict):
        self._config = config

        # Hidden root drives the customtkinter mainloop
        ctk.set_appearance_mode("dark")
        self._root = ctk.CTk()
        self._root.withdraw()  # No visible root window

        # Set app icon on root — inherited by all toplevel windows
        from ui.icons import render_app_icon
        from PIL import ImageTk
        self._icon_photo = ImageTk.PhotoImage(render_app_icon(32))
        self._root.iconphoto(True, self._icon_photo)

        # Engine
        self._engine = DictationApp(config)
        self._engine.add_callback(self._on_state_change)

        # Overlay
        self._overlay = OverlayWindow(
            self._root,
            on_toggle=self._toggle_recording,
            pos_x=config.get("pos_x"),
            pos_y=config.get("pos_y", 10),
            on_pos_changed=self._on_pos_changed
        )
        self._overlay.set_show_balloon(config.get("show_balloon", True))
        self._overlay.set_balloon_font_size(config.get("balloon_font_size", 16))

        # Tray
        self._tray = TrayManager(
            on_toggle=self._toggle_recording,
            on_quit=self._quit,
            on_settings=self._show_settings,
            on_history=self._show_history,
            device=config["device"],
            model=config["model_size"],
            hotkey=config["hotkey"],
        )

        # Lazy-created windows
        self._settings_win: SettingsWindow | None = None
        self._history_win: HistoryWindow | None = None
        self._audio_poll_id = None
        self._torn_down = False

    def run(self):
        """Start the application."""
        # Check if model exists on disk — show onboarding if not
        model_info = MODEL_REGISTRY[self._config["model_size"]]
        model_path = MODEL_DIR / model_info["local_dir"]
        has_files = model_path.exists() and (
            any(model_path.glob("*.xml")) or any(model_path.glob("*.onnx"))
        )
        needs_setup = not has_files

        if needs_setup:
            def _on_onboarding_done():
                self._start_engine()
            onboarding = OnboardingWindow(self._root, self._config, on_done=_on_onboarding_done)
            onboarding.show()
        else:
            self._start_engine()

        # Start tray
        self._tray.start()

        # Run mainloop on main thread
        self._mainloop()

    def _mainloop(self):
        """Run the Tk mainloop, shutting down cleanly on Ctrl+C.

        Ctrl+C surfaces here as a bare KeyboardInterrupt out of Tk. Without
        this, it escapes as an uncaught traceback and `_quit` never runs, so
        the global keyboard hook and the tray icon stay alive until the
        process dies.
        """
        try:
            self._root.mainloop()
        except KeyboardInterrupt:
            log("Interrupted — shutting down.")
        finally:
            self._teardown()
            try:
                self._root.destroy()
            except Exception:
                pass  # already destroyed via _quit

    def _start_engine(self):
        """Start the engine in non-blocking mode."""
        self._engine.start_background()

    def _toggle_recording(self):
        """Overlay/tray toggle. Looks the engine up on every call, because
        Settings can replace it; a bound method would keep the stopped one."""
        self._engine.toggle_recording()

    # -- State callback (fires from bg threads) ----------------------------

    def _on_state_change(self, state: AppState, data: dict):
        """Engine state changed — schedule UI update on main thread."""
        self._root.after(0, self._update_ui, state, data)

    def _update_ui(self, state: AppState, data: dict):
        """Update tray icon and overlay from the main thread."""
        state_name = state.value

        if state == AppState.LOADING:
            self._tray.update_state(state_name, "NPU Dictation — Loading model...")
            self._overlay.show_loading()
            self._settings_status("Loading model...", "#FF9F0A")

        elif state == AppState.READY:
            self._stop_audio_polling()
            text = data.get("text")
            self._tray.update_state(state_name, "NPU Dictation — Ready")
            if text:
                self._overlay.show_result(text)
            else:
                self._overlay.show_ready()
            self._settings_status("Model ready.", "#30D158")
            self._settings_set_apply(True)

        elif state == AppState.RECORDING:
            self._tray.update_state(state_name, "NPU Dictation — Recording...")
            self._overlay.show_recording()
            self._start_audio_polling()

        elif state == AppState.PROCESSING:
            self._stop_audio_polling()
            self._tray.update_state(state_name, "NPU Dictation — Transcribing...")
            self._overlay.show_processing()

        elif state == AppState.ERROR:
            self._stop_audio_polling()
            error_msg = data.get("error", "Unknown error")
            self._overlay.show_error()
            self._settings_set_apply(True)

            if data.get("restart_required") or device_failure():
                # Fail closed: no reload of any kind in this process.
                self._show_restart_required(data)
                return

            self._tray.update_state(state_name, f"NPU Dictation — Error: {error_msg[:60]}")
            self._settings_status(f"Error: {error_msg[:40]}", "#FF453A")

            # An NPU-only device loss may move to the GPU/CUDA, which has not failed.
            if data.get("device_lost") and data.get("device_failure") == "NPU":
                fallback_dev = "GPU"
                from dictation_engine import has_nvidia_gpu
                if has_nvidia_gpu():
                    fallback_dev = "CUDA"
                log(f"NPU DEVICE_LOST detected — falling back to {fallback_dev}")
                self._config["device"] = fallback_dev
                self._tray.update_info(
                    device=fallback_dev,
                    model=self._config["model_size"],
                    hotkey=self._config["hotkey"],
                )
                self._engine.fallback_device(fallback_dev)

    def _show_restart_required(self, data: dict):
        """Tell the user, once, that only an app restart recovers the device."""
        failure = device_failure() or {}
        device = data.get("device_failure") or failure.get("device") or "accelerator"
        message = failure.get("message") or data.get("error", "Restart the app.")
        self._tray.update_state(
            AppState.ERROR.value,
            f"NPU Dictation — {device} failed. Quit and restart the app.",
        )
        self._settings_status(f"{device} failed: restart NPU Dictation.", "#FF453A")
        if getattr(self, "_restart_alert_shown", False):
            return
        self._restart_alert_shown = True
        log(f"Restart required: {message}")
        self._alert_error("NPU Dictation: restart required", message)

    def _alert_error(self, title: str, message: str):
        try:
            from tkinter import messagebox
            messagebox.showerror(title, message, parent=self._root)
        except Exception as exc:
            log(f"Could not show error dialog: {exc}")

    # -- Audio level polling -----------------------------------------------

    def _start_audio_polling(self):
        """Start polling audio levels for waveform display."""
        self._poll_audio()

    def _poll_audio(self):
        if self._engine.is_recording:
            level = self._engine.recorder.audio_level
            self._overlay.update_audio_level(level)
            self._audio_poll_id = self._root.after(50, self._poll_audio)

    def _stop_audio_polling(self):
        if self._audio_poll_id:
            self._root.after_cancel(self._audio_poll_id)
            self._audio_poll_id = None

    def _on_pos_changed(self, pos_x: int, pos_y: int):
        """Save the new window position to config."""
        self._config["pos_x"] = pos_x
        self._config["pos_y"] = pos_y
        save_config(self._config)

    # -- Settings ----------------------------------------------------------

    def _show_settings(self):
        devices = DictationApp.list_input_devices()
        self._settings_win = SettingsWindow(
            self._root, self._config, devices, on_apply=self._on_settings_apply,
        )
        self._root.after(0, self._settings_win.show)

    # Settings that only take effect in a newly built engine.
    _REBUILD_KEYS = ("model_size", "device", "hotkey",
                     "beep_on_start", "sample_rate", "max_record_seconds")

    def _on_settings_apply(self, new_config: dict):
        model_changed = new_config["model_size"] != self._config["model_size"]
        rebuild = any(
            new_config.get(key, self._config.get(key)) != self._config.get(key)
            for key in self._REBUILD_KEYS
        )
        failure = device_failure()

        if rebuild:
            # The running engine shares self._config, so nothing is changed
            # until it is known to be idle. stop_if_idle() checks and stops
            # atomically and never waits on inference that may be hung.
            if failure:
                # Latched: no new work can start, the engine is not rebuilt.
                busy = self._engine.busy_reason()
            else:
                busy = self._engine.stop_if_idle()
            if busy:
                log(f"Settings not applied: {busy} in progress.")
                self._settings_status(
                    f"Not applied: {busy} in progress. Click Apply again "
                    f"when it finishes.", "#FF9F0A")
                return

        self._config.update(new_config)
        save_config(self._config)

        # Update balloon settings immediately (no engine restart needed)
        self._overlay.set_show_balloon(self._config.get("show_balloon", True))
        self._overlay.set_balloon_font_size(self._config.get("balloon_font_size", 16))

        if failure:
            # A rebuilt engine would load models in a process whose device
            # context may hang. Keep the saved settings for the next start,
            # and do not show them in the tray as if they were active now.
            self._tray.update_info(
                device=f"{self._config['device']} (after restart)",
                model=f"{self._config['model_size']} (after restart)",
                hotkey=f"{self._config['hotkey']} (after restart)",
            )
            if rebuild:
                log(f"Settings saved; not reloading after {failure['device']} failure.")
                self._settings_status("Saved. Restart NPU Dictation to apply.", "#FF9F0A")
            else:
                self._settings_status("Settings saved.", "#30D158")
            return

        self._tray.update_info(
            device=self._config["device"],
            model=self._config["model_size"],
            hotkey=self._config["hotkey"],
        )

        if rebuild:
            log("Settings changed — reloading engine...")
            self._settings_status("Loading model...", "#FF9F0A")
            self._settings_set_apply(False)

            # If model needs download, show onboarding first
            if model_changed and not is_model_downloaded(self._config["model_size"]):
                self._settings_status("Downloading model...", "#FF9F0A")
                def _on_download_done():
                    self._settings_status("Loading model...", "#FF9F0A")
                    self._engine = DictationApp(self._config)
                    self._engine.add_callback(self._on_state_change)
                    self._engine.start_background()
                onboarding = OnboardingWindow(self._root, self._config, on_done=_on_download_done)
                onboarding.show()
            else:
                self._engine = DictationApp(self._config)
                self._engine.add_callback(self._on_state_change)
                self._engine.start_background()
        else:
            self._settings_status("Settings saved.", "#30D158")

    def _settings_status(self, text: str, color: str = "gray60"):
        """Push a status message to the settings window if it's open."""
        if self._settings_win and self._settings_win.is_open:
            self._settings_win.update_status(text, color)

    def _settings_set_apply(self, enabled: bool):
        """Enable/disable Apply button in settings window."""
        if self._settings_win and self._settings_win.is_open:
            self._settings_win.set_apply_enabled(enabled)

    # -- History -----------------------------------------------------------

    def _show_history(self):
        self._history_win = HistoryWindow(self._root, self._engine.history)
        self._root.after(0, self._history_win.show)

    # -- Quit --------------------------------------------------------------

    def _quit(self):
        """Quit requested from the tray — mainloop is still running, so the
        window teardown is scheduled back onto the main thread."""
        self._teardown()
        try:
            self._root.after(0, self._root.destroy)
        except Exception:
            pass  # root already gone; the mainloop's finally covers teardown

    def _teardown(self):
        """Stop background workers. Idempotent: both the tray Quit and the
        Ctrl+C path call it, and one may follow the other."""
        if self._torn_down:
            return
        self._torn_down = True
        for name, stop in (("engine", self._engine.stop), ("tray", self._tray.stop)):
            try:
                stop()
            except Exception as exc:
                log(f"Error stopping {name} during shutdown: {exc}")


def main():
    parser = argparse.ArgumentParser(description="NPU Dictation Engine (GUI)")
    parser.add_argument("--device", choices=["NPU", "GPU", "CPU", "CUDA"], help="Override device")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), help="Model size")
    parser.add_argument("--language", type=str, help="Language code")
    parser.add_argument("--auto-enter", action="store_true", help="Press Enter after typing")
    parser.add_argument("--hotkey", type=str, help="Global hotkey")
    args = parser.parse_args()

    config = load_config()

    if args.device:
        config["device"] = args.device
    if args.model:
        config["model_size"] = args.model
    if args.language:
        config["language"] = args.language
    if args.auto_enter:
        config["auto_enter"] = True
    if args.hotkey:
        config["hotkey"] = args.hotkey

    # Auto-select device if not explicitly overridden
    if not args.device:
        model_info = MODEL_REGISTRY[config["model_size"]]
        preferred = model_info["preferred_device"]
        
        if preferred == "GPU":
            from dictation_engine import has_nvidia_gpu
            if has_nvidia_gpu():
                preferred = "CUDA"

        if config["device"] != preferred:
            log(f"Auto-selecting {preferred} for {config['model_size']}")
            config["device"] = preferred

    validate_config(config)

    app = GUIApp(config)
    app.run()


if __name__ == "__main__":
    main()
