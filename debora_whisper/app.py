"""GUI orchestrator for Débora Whisper.

Entry point for tray-icon mode. Wires the engine, system tray, overlay,
settings dialog, onboarding wizard, and transcription history together.

Usage:
    debora [--device NPU|GPU|CPU] [--model base|small|medium|turbo|parakeet]
"""

import sys
import argparse
import customtkinter as ctk

from debora_whisper.dictation_engine import (
    AppState, DictationApp, MODEL_REGISTRY,
    load_config, save_config, validate_config, log, create_model,
    is_model_downloaded, device_failure,
    apply_device_priority, avoid_lost_npu, detect_devices, rotate_logs, log_folder_moves,
    select_device,
)
from debora_whisper.npu_probe import probe_npu
from debora_whisper.ui.tray import TrayManager
from debora_whisper.ui.overlay import OverlayWindow
from debora_whisper.ui.settings import SettingsWindow
from debora_whisper.ui.history import HistoryWindow
from debora_whisper.ui.onboarding import OnboardingWindow


class GUIApp:
    """Main GUI application wiring all components."""

    def __init__(self, config: dict):
        self._config = config

        # Hidden root drives the customtkinter mainloop
        ctk.set_appearance_mode("dark")
        self._root = ctk.CTk()
        self._root.withdraw()  # No visible root window

        # Set app icon on root — inherited by all toplevel windows
        from debora_whisper.ui.icons import render_app_icon
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
            on_hardware_event=self._on_hardware_event,
            device=config["device"],
            model=config["model_size"],
            hotkey=config["hotkey"],
            on_voice_chat=self._toggle_voice_chat,
            voice_chat_on=lambda: bool(self._config.get("voice_chat")),
        )

        # Lazy-created windows
        self._settings_win: SettingsWindow | None = None
        self._history_win: HistoryWindow | None = None
        self._audio_poll_id = None
        self._torn_down = False

    def run(self):
        """Start the application."""
        # Check if model exists on disk — show onboarding if not
        needs_setup = not is_model_downloaded(self._config["model_size"])

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

    def _on_hardware_event(self):
        """Called by the tray manager when the PC wakes up or a device changes."""
        import threading
        from debora_whisper.dictation_engine import log
        
        if getattr(self, "_hw_event_timer", None):
            self._hw_event_timer.cancel()
            
        def recover():
            try:
                # Only warmup if not currently recording
                if not self._engine.is_recording and self._engine._state != AppState.ERROR:
                    log("Hardware change detected. Proactively warming up audio stream...")
                    self._engine.recorder.close()
                    self._engine.recorder.warmup(timeout=3.0)
            except Exception as e:
                log(f"Failed to proactively warmup audio: {e}")
                
        self._hw_event_timer = threading.Timer(1.0, recover)
        self._hw_event_timer.daemon = True
        self._hw_event_timer.start()

    def _on_state_change(self, state: AppState, data: dict):
        """Engine state changed — schedule UI update on main thread."""
        self._root.after(0, self._update_ui, state, data)

    def _update_ui(self, state: AppState, data: dict):
        """Update tray icon and overlay from the main thread."""
        if data.get("notice"):
            self._overlay.show_notice(data["notice"])
            self._settings_status(data["notice"], "gray60")
            return
        state_name = state.value

        if state == AppState.LOADING:
            self._tray.update_state(state_name, "Débora Whisper — Loading model...")
            self._overlay.show_loading()
            self._settings_status("Loading model...", "#FF9F0A")

        elif state == AppState.READY:
            self._stop_audio_polling()
            text = data.get("text")
            self._tray.update_state(state_name, "Débora Whisper — Ready")
            if text:
                self._overlay.show_result(text)
            else:
                self._overlay.show_ready()
            self._settings_status("Model ready.", "#30D158")
            self._settings_set_apply(True)
            
            if getattr(self._engine, "continuous_active", False):
                # Transition back to recording visually
                self._root.after(1500, lambda: self._update_ui(AppState.RECORDING, {}) if getattr(self._engine, "is_recording", False) else None)

        elif state == AppState.RECORDING:
            draft = data.get("draft_text", "")
            self._tray.update_state(state_name, "Débora Whisper — Recording...")
            self._overlay.show_recording(draft)
            self._start_audio_polling()

        elif state == AppState.SPEAKING:
            self._stop_audio_polling()
            self._tray.update_state(state_name, "Débora Whisper — Speaking...")
            self._overlay.show_speaking(data.get("text", ""))

        elif state == AppState.PROCESSING:
            self._stop_audio_polling()
            self._tray.update_state(state_name, "Débora Whisper — Transcribing...")
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

            self._tray.update_state(state_name, f"Débora Whisper — Error: {error_msg[:60]}")
            self._settings_status(f"Error: {error_msg[:40]}", "#FF453A")

            if data.get("device_lost"):
                failed_dev = data.get("device_failure")
                # Next healthy device of config["device_priority"].
                fallback_dev = select_device(self._config, detect_devices(),
                                             exclude={failed_dev})
                if fallback_dev is None:
                    log(f"{failed_dev} DEVICE_LOST detected — no other device in "
                        f"{self._config['device_priority']}; restart the app.")
                    return

                if failed_dev == "NPU":
                    # Schedule an invisible background recovery probe for NPU
                    self._schedule_npu_recovery()
                elif hasattr(self, "_npu_retry_count"):
                    # Reset NPU retry counter since we are switching away
                    self._npu_retry_count = 0

                log(f"{failed_dev} DEVICE_LOST detected — falling back to {fallback_dev} immediately")
                self._config["device"] = fallback_dev
                self._tray.update_info(
                    device=fallback_dev,
                    model=self._config["model_size"],
                    hotkey=self._config["hotkey"],
                )
                self._engine.fallback_device(fallback_dev)

    # Seconds before each NPU recovery probe. A lost NPU failed probes a
    # minute later but worked again (without a reboot) some minutes after.
    NPU_PROBE_DELAYS = (30, 60, 300, 900)

    def _schedule_npu_recovery(self):
        if not hasattr(self, "_npu_retry_count"):
            self._npu_retry_count = 0
        if self._npu_retry_count >= len(self.NPU_PROBE_DELAYS):
            log("NPU recovery retries exhausted. Staying on fallback device.")
            return
        delay = self.NPU_PROBE_DELAYS[self._npu_retry_count]
        self._npu_retry_count += 1
        log(f"Scheduling background NPU recovery probe in {delay}s "
            f"(Attempt {self._npu_retry_count}/{len(self.NPU_PROBE_DELAYS)})...")

        def _probe_thread():
            import time
            time.sleep(delay)
            self._run_npu_recovery_probe()

        import threading
        threading.Thread(target=_probe_thread, daemon=True).start()

    def _run_npu_recovery_probe(self):
        from debora_whisper.dictation_engine import (
            setup_model, create_model, forget_npu_loss, MODEL_REGISTRY, log)
        import numpy as np

        if self._engine.config["device"] == "NPU":
            return

        log("Probing NPU recovery in a separate process...")
        try:
            model_info = MODEL_REGISTRY[self._config["model_size"]]
            probe_config = self._config.copy()
            probe_config["device"] = "NPU"
            model_path = setup_model(probe_config)

            # Loading on a lost NPU can hang in the driver while holding the
            # GIL, which froze the whole app: only load here once a separate
            # process, killed if it hangs, has run the model on the NPU.
            probe_npu(probe_config, model_path)
            test_model = create_model(model_path, device="NPU", backend=model_info["backend"],
                                      model_size=self._config["model_size"])
            # Loaders fall back silently on benign errors (e.g. a locked
            # ov-cache blob), so a successful load is not proof of recovery.
            if test_model.device != "NPU":
                raise RuntimeError(f"probe model loaded on {test_model.device}, not NPU")

            silence = np.zeros(int(self._config["sample_rate"] * 0.5), dtype=np.float32)
            test_model.transcribe(silence, sample_rate=self._config["sample_rate"], language=self._config.get("language", "en"))
            
            log("NPU recovery successful! Swapping active engine back to NPU seamlessly...")
            forget_npu_loss()
            engine = self._engine
            self._root.after(0, lambda: self._swap_to_recovered_npu(engine, test_model))

        except Exception as e:
            log(f"Background NPU recovery probe failed: {e}")
            self._schedule_npu_recovery()

    SWAP_RETRY_MS = 5000

    def _swap_to_recovered_npu(self, engine, test_model):
        if self._engine is not engine:
            # Settings rebuilt the engine meanwhile; it chose its own device.
            return
        if not engine.inject_recovered_model("NPU", test_model):
            self._root.after(self.SWAP_RETRY_MS,
                             lambda: self._swap_to_recovered_npu(engine, test_model))
            return
        self._config["device"] = "NPU"
        self._tray.update_info(
            device="NPU",
            model=self._config["model_size"],
            hotkey=self._config["hotkey"],
        )
        self._npu_retry_count = 0

    def _show_restart_required(self, data: dict):
        """Tell the user, once, that only an app restart recovers the device."""
        failure = device_failure() or {}
        device = data.get("device_failure") or failure.get("device") or "accelerator"
        message = failure.get("message") or data.get("error", "Restart the app.")
        self._tray.update_state(
            AppState.ERROR.value,
            f"Débora Whisper — {device} failed. Quit and restart the app.",
        )
        self._settings_status(f"{device} failed: restart Débora Whisper.", "#FF453A")
        if getattr(self, "_restart_alert_shown", False):
            return
        self._restart_alert_shown = True
        log(f"Restart required: {message}")
        self._alert_error("Débora Whisper: restart required", message)

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
            self._tray.update_audio_level(level)
            self._audio_poll_id = self._root.after(100, self._poll_audio)

    def _stop_audio_polling(self):
        if self._audio_poll_id:
            self._root.after_cancel(self._audio_poll_id)
            self._audio_poll_id = None

    def _on_pos_changed(self, pos_x: int, pos_y: int):
        """Save the new window position to config."""
        self._config["pos_x"] = pos_x
        self._config["pos_y"] = pos_y
        save_config(self._config)

    # -- Voice chat --------------------------------------------------------

    def _toggle_voice_chat(self):
        """Tray item (on the tray's thread)."""
        self._root.after(0, self._set_voice_chat, not self._config.get("voice_chat"))

    def _set_voice_chat(self, enabled: bool):
        """Switch voice chat on the running engine; no restart, no rebuild.
        The LLM loads in its own process, so even a hung load cannot freeze
        the app."""
        self._config["voice_chat"] = enabled
        save_config(self._config)
        self._engine.set_voice_chat(enabled)
        if self._settings_win and self._settings_win.is_open:
            self._settings_win.set_voice_chat(enabled)

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
        voice_chat = new_config.get("voice_chat", self._config.get("voice_chat"))
        voice_chat_changed = bool(voice_chat) != bool(self._config.get("voice_chat"))
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
        if voice_chat_changed and not (rebuild and not failure):
            # A rebuilt engine starts voice chat itself; this one switches now.
            self._engine.set_voice_chat(bool(voice_chat))

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
                self._settings_status("Saved. Restart Débora Whisper to apply.", "#FF9F0A")
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


def _ensure_stdio():
    """pythonw (the Start Menu shortcut) has no console: sys.stdout/stderr
    are None and anything writing to them (download progress bars) would
    raise. app.log still gets every log line."""
    import os
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


INSTANCE_MUTEX = r"Local\debora-whisper-tray"


def _claim_single_instance(name: str = INSTANCE_MUTEX) -> bool:
    """One tray app per user session: a second one (say, a Start Menu click
    while the Startup shortcut already launched one) would hook the same
    hotkey and record twice. The named mutex dies with the process."""
    import ctypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.c_void_p  # a HANDLE, not an int
    kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p)
    global _instance_mutex
    _instance_mutex = kernel32.CreateMutexW(None, False, name)
    ERROR_ALREADY_EXISTS = 183
    return ctypes.get_last_error() != ERROR_ALREADY_EXISTS


_instance_mutex = None


def main():
    _ensure_stdio()
    parser = argparse.ArgumentParser(prog="debora", description="Débora Whisper (GUI)")
    parser.add_argument("--device", choices=["NPU", "GPU", "CPU", "CUDA"], help="Override device")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), help="Model size")
    parser.add_argument("--language", type=str, help="Language code")
    parser.add_argument("--auto-enter", action="store_true", help="Press Enter after typing")
    parser.add_argument("--hotkey", type=str, help="Global hotkey")
    parser.add_argument("--continuous", action="store_true", help="Enable continuous listening")
    parser.add_argument("--voice-chat", action="store_true",
                        help="Talk to a local LLM and hear its reply (voice chat) "
                             "instead of typing")
    shortcut = parser.add_mutually_exclusive_group()
    shortcut.add_argument("--install-shortcut", action="store_true",
                          help="Add Débora Whisper to the Start Menu, then exit")
    parser.add_argument("--autostart", action="store_true",
                        help="With --install-shortcut: also start with Windows")
    shortcut.add_argument("--remove-shortcut", action="store_true",
                          help="Remove the Start Menu and startup shortcuts, then exit")
    args = parser.parse_args()
    log_folder_moves()
    if args.autostart and not args.install_shortcut:
        parser.error("--autostart only works with --install-shortcut")

    if args.install_shortcut or args.remove_shortcut:
        from debora_whisper import shortcuts
        if args.remove_shortcut:
            shortcuts.remove()
        else:
            shortcuts.install(autostart=args.autostart)
        return

    if not _claim_single_instance():
        log("Débora Whisper is already running (see the tray icon); not starting another.")
        return

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
    if args.continuous:
        config["continuous_listening"] = True
    if args.voice_chat:
        config["voice_chat"] = True

    validate_config(config)
    rotate_logs()
    if not args.device:
        apply_device_priority(config)
    check_npu = avoid_lost_npu(config)

    app = GUIApp(config)
    if check_npu:
        app._schedule_npu_recovery()
    app.run()


if __name__ == "__main__":
    main()
