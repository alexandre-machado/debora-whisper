"""Live drafts typed into the target window and corrected in place."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import dictation_engine as de
from dictation_engine import AppState, DictationApp, DEFAULT_CONFIG

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


def test_draft_is_corrected_in_place_without_clipboard(screen):
    app = _app()
    _say(app, "Olá tudo", is_final=False)
    _say(app, "Olá tudo bem", is_final=False)
    _say(app, "Olá, tudo bem?", is_final=True)

    assert screen.text == "Olá, tudo bem? "
    assert screen.drafted == ["Olá tudo... ", " bem... "]
    # Only the differing tail goes through the clipboard paste.
    assert screen.pasted == [", tudo bem? "]
    assert app._draft_typed_text == ""


def test_final_equal_to_draft_still_presses_auto_enter(screen):
    app = _app(auto_enter=True)
    _say(app, "Pronto.", is_final=False)
    _say(app, "Pronto.", is_final=True)

    assert screen.text == "Pronto. "
    assert screen.pasted == []
    assert screen.enters == 1


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

    de.type_draft_text("é👍")

    assert units == [0xE9, 0xD83D, 0xDC4D]
