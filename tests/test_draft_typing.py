"""Live drafts typed into the target window and corrected in place."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from npu_whisper import dictation_engine as de
from npu_whisper.dictation_engine import AppState, DictationApp, DEFAULT_CONFIG

AUDIO = np.zeros(16000, dtype=np.float32)


class Screen:
    """The text box keys land in, plus where focus is."""

    def __init__(self):
        self.text = ""
        self.target = ("window", "field")
        self.pasted = []
        self.drafted = []
        self.enters = 0

    def paste(self, text, auto_enter=False):
        self.pasted.append(text)
        self.text += text
        if auto_enter:
            self.enters += 1

    def type_draft(self, text):
        if text:
            self.drafted.append(text)
            self.text += text

    def delete(self, count):
        if count > 0:
            self.text = self.text[:-count]


@pytest.fixture
def screen():
    s = Screen()
    with patch.object(de, "type_text", side_effect=s.paste), \
            patch.object(de, "type_draft_text", side_effect=s.type_draft), \
            patch.object(de, "delete_text", side_effect=s.delete), \
            patch.object(de, "get_input_target", side_effect=lambda: s.target), \
            patch("keyboard.press_and_release") as key:
        key.side_effect = lambda k: setattr(s, "enters", s.enters + 1)
        yield s


def _app(**config):
    app = DictationApp({**DEFAULT_CONFIG, "beep_on_start": False,
                        "continuous_listening": True, **config})
    app.recorder = MagicMock()
    app.whisper = MagicMock()
    app._model_ready.set()
    return app


def _say(app, text, is_final):
    if isinstance(text, Exception):
        app.whisper.transcribe.side_effect = text
    else:
        app.whisper.transcribe.side_effect = None
        app.whisper.transcribe.return_value = text
    app._finish_recording(audio=AUDIO, is_final=is_final)


def test_draft_is_corrected_in_place(screen):
    app = _app()
    _say(app, "Olá tudo", is_final=False)
    _say(app, "Olá tudo bem", is_final=False)
    _say(app, "Olá, tudo bem?", is_final=True)

    assert screen.text == "Olá, tudo bem? "
    assert screen.drafted == ["Olá tudo... ", " bem... "]
    # Only the differing tail is sent as final text.
    assert screen.pasted == [", tudo bem? "]
    assert app._draft_typed_text == ""


def test_final_equal_to_draft_still_presses_auto_enter(screen):
    app = _app(auto_enter=True)
    _say(app, "Pronto.", is_final=False)
    _say(app, "Pronto.", is_final=True)

    assert screen.text == "Pronto. "
    assert screen.pasted == []
    assert screen.enters == 1


@pytest.mark.parametrize("draft", ["Olá, tudo", "Olá, tudo bem?"])
def test_console_logs_complete_final_even_if_only_tail_or_nothing_is_typed(screen, capsys, draft):
    app = _app()
    _say(app, draft, is_final=False)
    assert "Final transcription:" not in capsys.readouterr().out

    _say(app, "Olá, tudo bem?", is_final=True)

    assert screen.text == "Olá, tudo bem? "
    output = capsys.readouterr().out
    assert output.count("Final transcription:") == 1
    assert "Final transcription: Olá, tudo bem?" in output
    assert "Final transcription: Olá, tudo bem?" in de.LOG_FILE.read_text(encoding="utf-8")


def test_final_log_is_not_truncated(screen, capsys):
    text = "Uma frase longa. " * 10
    _say(_app(), text, is_final=True)
    assert f"Final transcription: {text.strip()}" in capsys.readouterr().out


def test_focus_change_leaves_old_draft_and_types_full_text(screen):
    app = _app()
    _say(app, "Primeira parte", is_final=False)
    typed_in_old_window = screen.text
    screen.target = ("other window", "field")
    screen.text = "código do usuário"

    _say(app, "Primeira parte.", is_final=True)

    assert screen.text == "código do usuário" + "Primeira parte. "
    assert typed_in_old_window == "Primeira parte... "


def test_focus_moving_to_another_field_counts_as_change(screen):
    app = _app()
    _say(app, "Rascunho", is_final=False)
    screen.target = ("window", "other field")
    screen.text = "outro campo"

    _say(app, "Rascunho final.", is_final=True)

    assert screen.text == "outro campo" + "Rascunho final. "


def test_empty_draft_keeps_typed_draft(screen):
    app = _app()
    _say(app, "Além disso", is_final=False)
    _say(app, "Obrigado.", is_final=False)  # dropped as a hallucination

    assert screen.text == "Além disso... "
    assert app._draft_typed_text == "Além disso... "


def test_endpoint_receives_raw_draft_before_display_ellipses(screen):
    app = _app()
    app.whisper.transcribe.return_value = "Minha ideia é"
    app._finish_recording(audio=AUDIO, is_final=False, segment_id=7, audio_end=16000)
    app.recorder.endpoint.update.assert_called_once_with(7, 16000, "Minha ideia é")
    assert screen.text == "Minha ideia é... "


def test_hallucination_feedback_is_empty(screen):
    app = _app()
    app.whisper.transcribe.return_value = "Obrigado."
    app._finish_recording(audio=AUDIO, is_final=False, segment_id=7, audio_end=16000)
    app.recorder.endpoint.update.assert_called_once_with(7, 16000, "")


def test_final_does_not_change_endpoint(screen):
    app = _app()
    app.whisper.transcribe.return_value = "Minha ideia é"
    app._finish_recording(audio=AUDIO, is_final=True, segment_id=7, audio_end=16000)
    app.recorder.endpoint.update.assert_not_called()


def test_segment_consumer_preserves_draft_identity_and_position(monkeypatch, screen):
    from types import SimpleNamespace
    from npu_whisper.vad_endpoint import VadSegment

    app = _app()
    app.recorder = de.AudioRecorder(config=app.config)
    segment_id = app.recorder.endpoint.start()
    app.recorder.segment_queue.put(VadSegment(AUDIO, False, segment_id, len(AUDIO)))
    app.whisper.transcribe.return_value = "Minha ideia é"
    finish = app._finish_recording

    def consume(**kwargs):
        finish(**kwargs)
        app._stopping.set()

    app._finish_recording = consume
    # Run the actual consumer synchronously, stopping after this draft.
    monkeypatch.setattr(de.threading, "Thread", lambda target, **kw: SimpleNamespace(start=target))
    app._start_segment_consumer()
    assert app.recorder.endpoint.incomplete
    assert screen.text == "Minha ideia é... "


def test_empty_final_erases_draft(screen):
    app = _app()
    _say(app, "Hum", is_final=False)
    _say(app, "", is_final=True)

    assert screen.text == ""
    assert app._draft_typed_text == ""


def test_failed_final_keeps_draft_but_next_segment_does_not_erase_it(screen):
    app = _app()
    _say(app, "Texto do rascunho", is_final=False)
    _say(app, RuntimeError("inference failed"), is_final=True)
    assert app._state == AppState.ERROR

    _say(app, "Outra frase.", is_final=True)

    assert screen.text == "Texto do rascunho... Outra frase. "


def test_failed_draft_keeps_tracking_so_final_corrects_it(screen):
    # Real log: the NPU was lost during a draft; the final of the same
    # segment ran on CUDA and typed the whole sentence after the stale draft.
    app = _app()
    _say(app, "Oi, ae, tudo bem", is_final=False)
    _say(app, RuntimeError("inference failed"), is_final=False)

    _say(app, "Oi, e aí, tudo bem?", is_final=True)

    assert screen.text == "Oi, e aí, tudo bem? "


def test_missing_focus_report_is_not_a_focus_change(screen):
    app = _app()
    _say(app, "Tudo", is_final=False)
    screen.target = ("window", None)  # GetGUIThreadInfo blinked

    _say(app, "Tudo bem.", is_final=True)

    assert screen.text == "Tudo bem. "


def test_same_input_target():
    assert de.same_input_target((1, 2), (1, 2))
    assert de.same_input_target((1, None), (1, 2))
    assert not de.same_input_target((1, 2), (1, 3))
    assert not de.same_input_target((1, 2), (4, 2))
    assert not de.same_input_target(None, (1, 2))


def test_continuous_final_shows_neither_transcribing_nor_done(screen):
    app = _app()
    app.is_recording = True
    states = []
    app.add_callback(lambda state, data: states.append(state))

    _say(app, "Frase completa.", is_final=True)

    assert AppState.PROCESSING not in states
    # No "Done" balloon either: straight back to listening.
    assert states == [AppState.RECORDING]


def test_push_to_talk_final_still_shows_transcribing_and_done(screen):
    app = _app(continuous_listening=False)
    states = []
    app.add_callback(lambda state, data: states.append(state))

    _say(app, "Frase completa.", is_final=True)

    assert states == [AppState.PROCESSING, AppState.READY]


def test_unicode_typing_sends_utf16_units(monkeypatch):
    import ctypes
    units = []

    def send_input(count, inputs_ref, size):
        key_down = inputs_ref._obj[0]
        units.append(key_down._input.ki.wScan)
        return count

    monkeypatch.setattr(ctypes, "windll", MagicMock())
    ctypes.windll.user32.SendInput.side_effect = send_input
    monkeypatch.setattr(de.time, "sleep", lambda s: None)

    de._type_text_ctypes("é👍")

    assert units == [0xE9, 0xD83D, 0xDC4D]


@pytest.fixture
def clipboard(monkeypatch):
    import pyperclip
    import keyboard

    state = {"text": "user clipboard", "pasted": [], "pending": False}
    monkeypatch.setattr(pyperclip, "paste", lambda: state["text"])
    monkeypatch.setattr(pyperclip, "copy", lambda text: state.update(text=text))

    def key(name):
        if name == "ctrl+v":
            state["pending"] = True

    def wait(seconds):
        # Model a target that reads the clipboard after Ctrl+V returns.
        if state["pending"]:
            state["pasted"].append(state["text"])
            state["pending"] = False

    monkeypatch.setattr(keyboard, "press_and_release", key)
    monkeypatch.setattr(de.time, "sleep", wait)
    return state


def test_drafts_and_final_paste_before_restoring_clipboard(clipboard):
    with patch.object(de, "_type_text_ctypes") as unicode_keys, \
            patch.object(de.threading, "Thread") as background:
        for text, final in [("Olá, tudo... ", False), (" bem? ", False), ("fim", True)]:
            (de.type_text if final else de.type_draft_text)(text)
            assert clipboard["pasted"][-1] == text
            assert clipboard["text"] == "user clipboard"

    unicode_keys.assert_not_called()
    background.assert_not_called()
    assert clipboard["pasted"] == ["Olá, tudo... ", " bem? ", "fim"]


def test_paste_does_not_overwrite_new_user_copy(clipboard, monkeypatch):
    def wait(seconds):
        if seconds == 0.5:
            clipboard["text"] = "new user copy"

    monkeypatch.setattr(de.time, "sleep", wait)
    de.type_draft_text("Olá")
    assert clipboard["text"] == "new user copy"


def test_failed_paste_restores_clipboard_and_releases_lock(clipboard):
    with patch("keyboard.press_and_release", side_effect=RuntimeError("paste failed")):
        with pytest.raises(RuntimeError, match="paste failed"):
            de.type_draft_text("Olá")

    assert clipboard["text"] == "user clipboard"
    assert de._clipboard_lock.acquire(blocking=False)
    de._clipboard_lock.release()
