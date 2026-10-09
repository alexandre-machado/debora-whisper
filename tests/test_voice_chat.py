"""config["voice_chat"]: the final text goes to a local LLM and its reply is
spoken by a local TTS server instead of being typed."""
import io
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from debora_whisper import dictation_engine as de
from debora_whisper import voice_chat as vc
from debora_whisper.dictation_engine import DictationApp, DEFAULT_CONFIG

AUDIO = np.zeros(16000, dtype=np.float32)


def _wav(seconds=0.1, rate=24000):
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x10" * int(seconds * rate))
    return out.getvalue()


@pytest.fixture
def server():
    """TTS stub (/tts, /health). Set .tts_status or .health_status."""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health" or srv.health_status != 200:
                return self.send_error(503)
            self._send(b'{"ok": true}', "application/json")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            srv.requests.append((self.path, body))
            if self.path != "/tts" or srv.tts_status != 200:
                return self.send_error(srv.tts_status if self.path == "/tts" else 404)
            self._send(_wav(), "audio/wav")

        def _send(self, payload, content_type):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            try:
                self.wfile.write(payload)
            except OSError:
                pass

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.requests, srv.tts_status, srv.health_status = [], 200, 200
    srv.tts_url = f"http://127.0.0.1:{srv.server_port}"
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


class FakeLLM:
    """Stands in for the OpenVINO model: streams .chunks, or raises .error."""

    def __init__(self, chunks=(), error=None):
        self.chunks, self.error, self.calls = list(chunks), error, []

    def __call__(self, messages, on_text, stop):
        self.calls.append(messages)
        if self.error:
            raise self.error
        for chunk in self.chunks:
            if stop.is_set():
                return
            on_text(chunk)


def _cfg(server, **extra):
    return {**DEFAULT_CONFIG, "voice_chat": True, "language": "pt",
            "tts_url": server.tts_url, **extra}


def _chat(server, llm, **extra):
    """A VoiceChat whose playback is recorded, not played."""
    played = []
    chat = vc.VoiceChat(_cfg(server, **extra), log=lambda m: None, llm=llm,
                        play=lambda samples, rate, stop: played.append((len(samples), rate)))
    return chat, played


def _tts_texts(server):
    return [body["text"] for path, body in server.requests if path == "/tts"]


def test_reply_is_streamed_and_spoken_sentence_by_sentence(server):
    llm = FakeLLM(["A capital ", "é Canberra. Fica", " no sul! Mais", " algo?"])
    chat, played = _chat(server, llm)
    assert chat.respond("qual é a capital da austrália") == \
        "A capital é Canberra. Fica no sul! Mais algo?"
    # "Fica no sul!" is too short to send alone: it waits for the next one.
    assert _tts_texts(server) == ["A capital é Canberra.", "Fica no sul! Mais algo?"]
    assert played == [(2400, 24000)] * 2
    assert llm.calls[0] == [
        {"role": "system", "content": vc.VOICE_CHAT_PROMPTS["pt"] +
         f" Agora é {vc._PT_WEEKDAYS[time.localtime().tm_wday]}, " +
         time.strftime("%d/%m/%Y, %H:%M.")},
        {"role": "user", "content": "qual é a capital da austrália"}]
    assert [b["language"] for p, b in server.requests if p == "/tts"] == ["pt"] * 2


def test_a_short_reply_is_still_spoken(server):
    chat, played = _chat(server, FakeLLM(["Opa! ", "Sim."]))
    assert chat.respond("oi") == "Opa! Sim."
    assert _tts_texts(server) == ["Opa! Sim."] and len(played) == 1


def test_emoji_is_neither_shown_nor_spoken(server):
    chat, played = _chat(server, FakeLLM(["Tudo certo por aqui, e você?\n😊"]))
    assert chat.respond("oi") == "Tudo certo por aqui, e você?"
    assert chat._history[-1]["content"] == "Tudo certo por aqui, e você?"
    assert _tts_texts(server) == ["Tudo certo por aqui, e você?"]


def test_emoji_inside_a_sentence_is_not_spoken():
    assert vc.without_emoji("Oi 😊, tudo bem? 👍🏽 Família 👨‍👩‍👧 e ❤️ 1️⃣") == \
        "Oi , tudo bem? Família e 1"
    assert vc.without_emoji("ação ‍ 25° ©") == "ação ‍ 25° ©"


def test_other_languages_get_the_english_prompt_and_their_language(server):
    llm = FakeLLM(["Hola, ¿qué tal?"])
    chat, _ = _chat(server, llm, language="es")
    chat.respond("hola")
    assert llm.calls[0][0]["content"] == vc.DEFAULT_VOICE_CHAT_PROMPT + \
        " The user speaks Spanish: reply in Spanish." + \
        time.strftime(" It is now %A, %Y-%m-%d %H:%M.")


def test_configured_prompt_wins_over_her_portuguese_one(server):
    llm = FakeLLM(["ok"])
    chat, _ = _chat(server, llm, llm_prompt="Seja breve.")
    chat.respond("oi")
    assert llm.calls[0][0]["content"].startswith(
        "Seja breve. The user speaks Portuguese: reply in Portuguese.")


def test_configured_prompt_is_used(server):
    llm = FakeLLM(["ok"])
    chat, _ = _chat(server, llm, llm_prompt="Seja breve.", language="auto")
    chat.respond("oi")
    assert llm.calls[0][0]["content"] == "Seja breve." +         time.strftime(" It is now %A, %Y-%m-%d %H:%M.")


def test_earlier_turns_are_sent_as_history(server):
    llm = FakeLLM(["Canberra."])
    chat, _ = _chat(server, llm)
    chat.respond("capital da austrália")
    llm.chunks = ["Uns 450 mil."]
    chat.respond("e quantos habitantes")
    assert llm.calls[1][1:] == [
        {"role": "user", "content": "capital da austrália"},
        {"role": "assistant", "content": "Canberra."},
        {"role": "user", "content": "e quantos habitantes"}]


def test_a_repeated_reply_is_not_kept_in_the_history(server):
    """A copy in the history made every next reply a copy too."""
    llm = FakeLLM(["Parece que quer brincar."])
    chat, _ = _chat(server, llm)
    chat.respond("ah, mas você")
    assert chat.respond("e...") == "Parece que quer brincar."  # still spoken
    llm.chunks = ["Não sei, não tenho acesso à previsão."]
    chat.respond("vai chover amanhã?")
    assert [m["content"] for m in llm.calls[2][1:]] == [
        "ah, mas você", "Parece que quer brincar.", "e...", "vai chover amanhã?"]


def test_history_resets_after_a_long_pause(server):
    llm = FakeLLM(["Canberra."])
    chat, _ = _chat(server, llm)
    chat.respond("capital da austrália")
    chat._last_turn -= vc.HISTORY_IDLE_RESET_SECONDS + 1
    chat.respond("oi")
    assert len(llm.calls[1]) == 2


def test_markdown_and_inline_reasoning_are_not_spoken(server):
    chat, _ = _chat(server, FakeLLM(["<think>hm</think>**Olá!**\n\n- `um` item"]))
    assert chat.respond("oi") == "Olá! - um item"
    assert _tts_texts(server) == ["Olá! - um item"]


def test_llm_failure_returns_nothing_and_speaks_nothing(server):
    chat, played = _chat(server, FakeLLM(error=RuntimeError("model crashed")))
    assert chat.respond("oi") == ""
    assert played == [] and _tts_texts(server) == []


def test_sentence_the_tts_rejects_is_skipped_and_the_next_still_spoken(server):
    server.tts_status = 500  # what Chatterbox answers for text it cannot speak
    logs = []
    chat, played = _chat(server, FakeLLM(["Primeira frase longa. Segunda frase longa."]))
    chat.log = logs.append
    assert chat.respond("oi") == "Primeira frase longa. Segunda frase longa."
    assert played == []
    assert len(_tts_texts(server)) == 2
    assert any("TTS skipped 'Primeira frase longa.' (HTTP 500" in m for m in logs)


def test_unreachable_tts_still_returns_the_reply(server, monkeypatch):
    monkeypatch.setattr(vc, "synthesize", MagicMock(side_effect=ConnectionResetError()))
    chat, played = _chat(server, FakeLLM(["Primeira frase longa. Segunda frase longa."]))
    assert chat.respond("oi") == "Primeira frase longa. Segunda frase longa."
    assert played == []
    assert vc.synthesize.call_count == 1  # no retry per sentence once the server is gone


def test_interrupt_stops_the_reply(server):
    played = []
    chat = vc.VoiceChat(_cfg(server), log=lambda m: None, llm=FakeLLM(["Um. ", "Dois. ", "Três."]),
                        play=lambda s, r, stop: played.append(1) or chat.interrupt())
    reply = chat.respond("conta até três")
    assert played == [1]
    assert reply.startswith("Um.")


def test_interrupted_turn_stays_stopped_when_the_next_starts(server):
    """The next turn must not revive a generation still winding down."""
    stops = []
    llm = FakeLLM(["Oi."])
    chat, _ = _chat(server, lambda m, on_text, stop: stops.append(stop) or llm(m, on_text, stop))
    chat.respond("um")
    chat.respond("dois")
    assert stops[0] is not stops[1] and stops[0].is_set()


@pytest.mark.parametrize("url", ["file:///C:/Windows/win.ini", "ftp://host/v1"])
def test_non_http_tts_url_is_never_opened(url):
    with patch("urllib.request.OpenerDirector.open") as opened:
        with pytest.raises(ValueError):
            vc.synthesize("oi", {**DEFAULT_CONFIG, "tts_url": url})
    opened.assert_not_called()


def test_environment_proxy_is_not_used(server, monkeypatch):
    # The proxy points nowhere: the request only succeeds if it bypasses it.
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("NO_PROXY", "")
    chat, played = _chat(server, FakeLLM(["Direto."]))
    assert chat.respond("oi") == "Direto."
    assert len(played) == 1


@pytest.mark.parametrize("key, value", [
    ("tts_url", "127.0.0.1:8765"), ("tts_url", None),
    ("tts_timeout_seconds", -1), ("tts_timeout_seconds", "30"), ("tts_timeout_seconds", True),
    ("llm_model", ""), ("llm_model", None), ("llm_device", 3), ("tts_voice", ""), ("tts_voice", 1),
])
def test_invalid_settings_are_rejected(key, value):
    with pytest.raises(ValueError):
        de.validate_config({**DEFAULT_CONFIG, key: value})


@pytest.mark.parametrize("buffer, done, rest", [
    ("Oi. Tudo", ["Oi."], "Tudo"),
    ("Oi", [], "Oi"),
    ("Sim! Não? Talvez… ", ["Sim!", "Não?", "Talvez…"], ""),
    ("Linha um\nLinha", ["Linha um"], "Linha"),
    ("Custa 3.50 reais", [], "Custa 3.50 reais"),
])
def test_split_sentences(buffer, done, rest):
    assert vc.split_sentences(buffer) == (done, rest)


# --- The LLM model files ---------------------------------------------------

def test_hub_model_is_downloaded_once(monkeypatch):
    calls = []

    def snapshot_download(repo, local_files_only=False):
        calls.append(local_files_only)
        if local_files_only and len(calls) == 1:
            raise FileNotFoundError(repo)
        return "/hub/" + repo

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        MagicMock(snapshot_download=snapshot_download))
    assert vc._model_path("OpenVINO/Qwen3-8B-int4-cw-ov", log=lambda m: None) == \
        "/hub/OpenVINO/Qwen3-8B-int4-cw-ov"
    assert vc._model_path("OpenVINO/Qwen3-8B-int4-cw-ov", log=lambda m: None) == \
        "/hub/OpenVINO/Qwen3-8B-int4-cw-ov"
    assert calls == [True, False, True]  # the second load stays offline


# --- In the dictation flow ---------------------------------------------------

def _app(**config):
    app = DictationApp({**DEFAULT_CONFIG, "beep_on_start": False, **config})
    app.recorder = MagicMock()
    app.whisper = MagicMock()
    app._model_ready.set()
    return app


def _wait_reply(app):
    worker = app._voice_worker
    if worker is not None:
        worker.join(5)
        assert not worker.is_alive()


def _say(app, text, is_final=True, wait=True):
    app.whisper.transcribe.return_value = text
    app._finish_recording(audio=AUDIO, is_final=is_final)
    if wait:
        _wait_reply(app)


@pytest.fixture
def typed():
    out = []
    with patch.object(de, "type_text", side_effect=lambda t, auto_enter=False: out.append(t)), \
            patch.object(de, "type_draft_text", side_effect=lambda t: out.append(("draft", t))), \
            patch.object(de, "delete_text", side_effect=lambda n: out.append(("delete", n))), \
            patch.object(de, "get_input_target", return_value=("window", "field")):
        yield out


def test_final_text_goes_to_the_llm_and_nothing_is_typed(typed):
    app = _app(voice_chat=True)
    states = []
    app.add_callback(lambda s, d: states.append((s, d)))
    with patch.object(app.voice_chat, "respond", return_value="Tudo ótimo!") as respond:
        _say(app, " ola tudo bem")
    assert respond.call_args.args[0] == "ola tudo bem"
    assert typed == []
    assert app._history[-1]["text"] == "ola tudo bem\n→ Tudo ótimo!"
    assert states[-1] == (de.AppState.READY, {"text": "Tudo ótimo!"})


def test_the_reply_shows_as_speaking_never_as_ready(typed):
    """While the reply is said the app is not idle: no green Ready until it
    is done, and in continuous listening not even then."""
    app = _app(voice_chat=True, continuous_listening=True)
    app.is_recording = True
    states = []
    app.add_callback(lambda s, d: states.append((s, d)))

    def respond(text, on_reply, stop):
        on_reply("Oi,")
        on_reply("Oi, tudo bem!")
        return "Oi, tudo bem!"
    with patch.object(app.voice_chat, "respond", side_effect=respond):
        _say(app, "oi")
    kinds = [s for s, _ in states]
    assert de.AppState.READY not in kinds
    assert states[-3:] == [(de.AppState.SPEAKING, {"text": "Oi,"}),
                           (de.AppState.SPEAKING, {"text": "Oi, tudo bem!"}),
                           (de.AppState.RECORDING, {"draft_text": ""})]


def test_microphone_keeps_listening_while_the_reply_plays(typed):
    app = _app(voice_chat=True)
    with patch.object(app.voice_chat, "respond", return_value="Oi."):
        _say(app, "oi")
    app.recorder.set_muted.assert_not_called()


def test_voice_chat_off_types_the_text(typed):
    app = _app(voice_chat=False)
    with patch.object(app.voice_chat, "respond") as respond:
        _say(app, "ola tudo bem")
    respond.assert_not_called()
    assert typed == ["ola tudo bem "]


def test_drafts_are_shown_not_typed_or_sent(typed):
    app = _app(voice_chat=True, continuous_listening=True)
    app.is_recording = True
    states = []
    app.add_callback(lambda s, d: states.append((s, d)))
    with patch.object(app.voice_chat, "respond", return_value="Oi!") as respond:
        _say(app, "ola tudo", is_final=False)
        respond.assert_not_called()
        _say(app, "ola tudo bem")
    respond.assert_called_once()
    assert typed == []
    assert (de.AppState.RECORDING, {"draft_text": "ola tudo"}) in states
    # Still listening: back to the microphone after the reply.
    assert states[-1] == (de.AppState.RECORDING, {"draft_text": ""})


def test_short_thanks_is_preserved_in_voice_chat_but_silence_is_not_sent(typed):
    app = _app(voice_chat=True)
    with patch.object(app.voice_chat, "respond", return_value="De nada.") as respond:
        _say(app, "Obrigado.")
        _say(app, "")
    respond.assert_called_once()
    assert app.history[0]["text"].startswith("Obrigado.")
    assert typed == []


def test_failed_reply_warns(typed):
    app = _app(voice_chat=True, beep_on_start=True)
    app.chimes = MagicMock()
    with patch.object(app.voice_chat, "respond", return_value=""):
        _say(app, "oi")
    app.chimes.play.assert_any_call("warning")
    assert app._history[-1]["text"] == "oi"


def test_hotkey_while_speaking_interrupts_the_reply():
    app = _app(voice_chat=True)
    app.voice_chat.speaking = True
    with patch.object(app.voice_chat, "interrupt") as interrupt, \
            patch.object(app, "_watch_key"), patch.object(app, "_start_recording") as start:
        app.toggle_recording()
    interrupt.assert_called_once()
    start.assert_not_called()


def test_stop_interrupts_the_reply():
    app = _app(voice_chat=True)
    with patch.object(app.voice_chat, "interrupt") as interrupt:
        app.stop()
    interrupt.assert_called_once()


@pytest.mark.parametrize("barge_in", [True, False])
@pytest.mark.parametrize("backend", ["local", "claude"])
def test_speech_during_reply_is_transcribed_and_sent_in_order(typed, barge_in, backend):
    app = _app(voice_chat=True, voice_chat_barge_in=barge_in, voice_chat_backend=backend)
    started, release = threading.Event(), threading.Event()
    turns, stops = [], []

    def respond(text, on_reply, stop):
        turns.append(text)
        stops.append(stop)
        if len(turns) == 1:
            started.set()  # still thinking; no audio has been played
            assert release.wait(3)
        return "Resposta."

    with patch.object(app.voice_chat, "respond", side_effect=respond), \
            patch.object(app.voice_chat, "interrupt") as interrupt:
        try:
            _say(app, "sobre aquele projeto...", wait=False)
            assert started.wait(2)
            _say(app, "quero mudar a interface", wait=False)
            assert len(app.history) == 2  # saved before either reply finishes
            assert app.whisper.transcribe.call_count == 2
            assert turns == ["sobre aquele projeto..."]
            assert stops[0].is_set() is barge_in
            assert interrupt.call_count == int(barge_in)
            assert app.history[1]["voice_chat_status"] == ("barge-in" if barge_in else "queued")
        finally:
            release.set()
            _wait_reply(app)
    assert turns == ["sobre aquele projeto...", "quero mudar a interface"]
    assert stops[0] is not stops[1]
    if barge_in:
        assert app.history[0]["voice_chat_status"] == "interrupted"
    app.recorder.set_muted.assert_not_called()
    assert typed == []


def test_echo_is_logged_and_kept_in_history_without_interrupting(typed, monkeypatch):
    app = _app(voice_chat=True)
    logs = []
    monkeypatch.setattr(de, "log", logs.append)
    app.voice_chat._remember_spoken("A próxima etapa é conferir o projeto.", 3)
    with patch.object(app.voice_chat, "respond") as respond, \
            patch.object(app.voice_chat, "interrupt") as interrupt:
        _say(app, "A proxima etapa e conferir o projeto!")
    respond.assert_not_called()
    interrupt.assert_not_called()
    assert app.history[0]["voice_chat_status"] == "echo"
    assert app.history[0]["text"] == "A proxima etapa e conferir o projeto!"
    assert any("Voice chat: ignored echo" in line for line in logs)
    assert any("Final transcription" in line for line in logs)


def test_echo_filter_can_be_disabled(typed):
    app = _app(voice_chat=True, voice_chat_echo_filter=False)
    app.voice_chat._remember_spoken("Um instante.", 1)
    with patch.object(app.voice_chat, "respond", return_value="Oi.") as respond:
        _say(app, "Um instante.")
    respond.assert_called_once()


def test_echo_uses_capture_time_instead_of_transcription_time():
    chat = vc.VoiceChat({})
    chat._spoken = [(100, 104, "O projeto tem quatro arquivos.")]
    assert chat.is_echo("O projeto tem quatro arquivos", 101, 103)
    assert not chat.is_echo("O projeto tem quatro arquivos", 90, 99)
    assert not chat.is_echo("O projeto tem quatro arquivos", 110, 113)
    assert not chat.is_echo("Quero mudar a interface", 101, 103)


def test_interrupted_thinking_keeps_the_local_user_message(server):
    started = threading.Event()
    messages = []

    def llm(turns, on_text, stop):
        messages.append(turns)
        if len(messages) == 1:
            started.set()
            assert stop.wait(3)
        else:
            on_text("Certo.")

    chat, played = _chat(server, llm)
    worker = threading.Thread(target=chat.respond, args=("sobre o projeto...",))
    worker.start()
    try:
        assert started.wait(2)
        chat.interrupt()
        worker.join(2)
        assert not worker.is_alive()
        assert played == []
        chat.respond("quero mudar a interface")
    finally:
        chat.interrupt()
        worker.join(3)
    assert [m["content"] for m in messages[1][1:]] == [
        "sobre o projeto...", "quero mudar a interface"]


def test_played_waiting_phrase_is_an_echo_reference(server):
    chat, _ = _chat(server, FakeLLM(["Um instante."]))
    chat.respond("oi")
    now = time.monotonic()
    assert chat.is_echo("Um instante", now - 1, now)


@pytest.mark.parametrize("module", ["debora_whisper.app", "debora_whisper.dictation_engine"])
def test_voice_chat_flag_turns_the_mode_on(monkeypatch, module):
    import importlib
    import sys
    mod = importlib.import_module(module)
    started = []
    monkeypatch.setattr(sys, "argv", ["debora", "--voice-chat", "--device", "CPU"])
    monkeypatch.setattr(mod, "load_config", lambda: dict(DEFAULT_CONFIG))
    app_class = "GUIApp" if module.endswith(".app") else "DictationApp"
    monkeypatch.setattr(mod, app_class, lambda config: started.append(config) or MagicMock())
    if module.endswith(".app"):
        monkeypatch.setattr(mod, "_claim_single_instance", lambda: True)
    mod.main()
    assert started[0]["voice_chat"] is True


# --- TTS server started by the app ------------------------------------------

@pytest.fixture
def no_tts_process(monkeypatch):
    monkeypatch.setattr(vc, "_tts_process", None)
    yield
    vc.stop_tts_server()


def _closed_port_url():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}"


def test_tts_server_command_is_started_when_nothing_answers(no_tts_process, tmp_path):
    config = {**DEFAULT_CONFIG, "tts_url": _closed_port_url(),
              "tts_server_command": [sys.executable, "-c", "print('carregando'); import time; time.sleep(30)"]}
    log_path = tmp_path / "tts_server.log"
    vc.ensure_tts_server(config, log=lambda m: None, log_path=log_path)
    process = vc._tts_process
    assert process is not None and process.poll() is None
    vc.ensure_tts_server(config, log=lambda m: None, log_path=log_path)
    assert vc._tts_process is process  # one server per app process
    deadline = time.time() + 10
    while "carregando" not in log_path.read_text() and time.time() < deadline:
        time.sleep(0.1)
    vc.stop_tts_server()
    assert process.poll() is not None
    assert "carregando" in log_path.read_text()


def test_running_tts_server_is_reused(no_tts_process, server):
    with patch("subprocess.Popen") as popen:
        vc.ensure_tts_server({**DEFAULT_CONFIG, "tts_url": server.tts_url,
                              "tts_server_command": ["never-run"]})
    popen.assert_not_called()


def test_tts_server_up_needs_a_healthy_answer(server):
    config = {**DEFAULT_CONFIG, "tts_url": server.tts_url}
    assert vc.tts_server_up(config)
    server.health_status = 500
    assert not vc.tts_server_up(config)


def test_wait_gives_up_when_the_started_server_dies(no_tts_process):
    config = {**DEFAULT_CONFIG, "tts_url": _closed_port_url(),
              "tts_server_command": [sys.executable, "-c", "raise SystemExit(1)"]}
    vc.ensure_tts_server(config, log=lambda m: None)
    start = time.time()
    assert not vc.wait_tts_server(config, timeout=30)
    assert time.time() - start < 10


def test_crashed_tts_server_is_not_restarted_every_reply(no_tts_process):
    config = {**DEFAULT_CONFIG, "tts_url": _closed_port_url(),
              "tts_server_command": [sys.executable, "-c", "raise SystemExit(1)"]}
    logged = []
    vc.ensure_tts_server(config, log=logged.append)
    vc._tts_process.wait(10)
    with patch("subprocess.Popen") as popen:
        vc.ensure_tts_server(config, log=logged.append)
        vc.ensure_tts_server(config, log=logged.append)
    popen.assert_not_called()
    assert sum("exited (code 1)" in m for m in logged) == 1


@pytest.mark.parametrize("command", ["python server.py", [], ["python", ""], [1]])
def test_invalid_tts_server_command_is_rejected(command):
    with pytest.raises(ValueError):
        de.validate_config({**DEFAULT_CONFIG, "tts_server_command": command})


# --- Default TTS command (uv) -------------------------------------------------

@pytest.fixture
def voices(monkeypatch, tmp_path):
    monkeypatch.setattr(vc.paths, "VOICES_DIR", tmp_path)
    monkeypatch.setattr(vc.paths, "BUNDLED_VOICES_DIR", tmp_path / "bundled")
    monkeypatch.setattr(vc.shutil, "which", lambda name: "C:/uv/uv.exe")
    (tmp_path / "isabel.wav").write_bytes(_wav())
    return tmp_path


def test_default_command_runs_the_bundled_server_with_uv(voices):
    command = vc.tts_command({**DEFAULT_CONFIG, "tts_voice": "isabel", "language": "pt"})
    assert command == ["C:/uv/uv.exe", "run", "--script", str(vc.TTS_SERVER_SCRIPT),
                       "--port", "8765", "--voice", str(voices / "isabel.wav"),
                       "--language", "pt"]
    assert vc.TTS_SERVER_SCRIPT.is_file()


def test_voice_can_be_a_path(voices):
    path = voices / "outra.wav"
    path.write_bytes(_wav())
    command = vc.tts_command({**DEFAULT_CONFIG, "tts_voice": str(path), "language": "auto"})
    assert command[-2:] == ["--voice", str(path)]


def test_missing_voice_falls_back_to_the_default_voice(voices):
    logged = []
    command = vc.tts_command({**DEFAULT_CONFIG, "tts_voice": "ninguem"}, log=logged.append)
    assert "--voice" not in command
    assert "not found" in logged[0]


def test_voices_are_listed_by_name(voices):
    (voices / "Carol.wav").write_bytes(_wav())
    (voices / "notas.txt").write_text("x")
    (voices / "bundled").mkdir()
    (voices / "bundled" / "isabel.wav").write_bytes(_wav())
    (voices / "bundled" / "mari.wav").write_bytes(_wav())
    assert vc.list_voices() == ["Carol", "isabel", "mari"]


def test_bundled_voice_is_used_unless_the_voices_folder_has_one(voices):
    (voices / "bundled").mkdir()
    (voices / "bundled" / "mari.wav").write_bytes(_wav())
    assert vc.resolve_voice("mari") == voices / "bundled" / "mari.wav"
    (voices / "mari.wav").write_bytes(_wav())
    assert vc.resolve_voice("mari") == voices / "mari.wav"


def test_the_default_voice_ships_with_the_app():
    from debora_whisper import paths
    assert DEFAULT_CONFIG["tts_voice"] == "debora_v2"
    for name in ("carol", "debora", "debora_v2", "isabel", "mari"):
        path = paths.BUNDLED_VOICES_DIR / f"{name}.wav"
        assert "debora_whisper" in path.parts
        with wave.open(str(path)) as w:
            assert w.getnchannels() == 1 and 8 < w.getnframes() / w.getframerate() < 20


@pytest.mark.parametrize("voice, sent", [
    ("isabel", "isabel.wav"),  # a name in the voices folder
    (None, ""),                # Chatterbox's own voice
    ("ninguem", None),         # missing: the server keeps its voice
])
def test_each_request_names_the_voice(voices, server, voice, sent):
    """A voice picked in Settings speaks the next sentence, without
    restarting the TTS server."""
    vc.synthesize("Olá, tudo bem com você?", {**DEFAULT_CONFIG, "tts_url": server.tts_url,
                                               "tts_voice": voice})
    body = server.requests[-1][1]
    if sent is None:
        assert "voice" not in body
    elif sent:
        assert body["voice"] == str((voices / sent).resolve())
    else:
        assert body["voice"] == ""


# --- TTS server: voice per request --------------------------------------------

def test_tts_server_accepts_only_audio_files_as_voice(voices):
    from debora_whisper import tts_server
    assert tts_server.check_voice(None) is None
    assert tts_server.check_voice("") == ""
    assert tts_server.check_voice(str(voices / "isabel.wav")) == str(voices / "isabel.wav")
    (voices / "segredo.txt").write_text("x")
    for bad in (str(voices / "segredo.txt"), str(voices / "falta.wav"), 3,
                r"\\attacker\share\x.wav", "//attacker/share/x.wav"):
        with pytest.raises(ValueError):
            tts_server.check_voice(bad)


@pytest.mark.parametrize("headers, status", [
    ({"Content-Type": "application/json"}, 400),  # gets past the gate: empty text
    ({"Content-Type": "text/plain"}, 403),
    ({"Content-Type": "application/json", "Origin": "https://evil.example"}, 403),
])
def test_tts_server_refuses_requests_from_web_pages(headers, status):
    from debora_whisper import tts_server
    model = type("Model", (), {"sr": 24000, "conds": None})()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), tts_server.make_handler(model, "pt"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{srv.server_port}/tts", data=b'{"text": ""}', headers=headers)
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=5)
        assert error.value.code == status
    finally:
        srv.shutdown()
        srv.server_close()


def test_numbers_too_big_to_spell_are_left_as_digits():
    big = "1" * 40
    assert vc.spoken_numbers(f"São {big} estrelas.", "pt") == f"São {big} estrelas."


def test_tts_server_prepares_each_voice_once(voices):
    from debora_whisper import tts_server
    model = MagicMock(conds="startup", builtin_conds="builtin")
    model.prepare_conditionals.side_effect = lambda path: setattr(model, "conds", f"conds:{path}")
    chosen = tts_server.Voices(model)
    isabel = str(voices / "isabel.wav")
    for voice, conds in [(isabel, f"conds:{isabel}"), ("", "builtin"),
                         (None, "startup"), (isabel, f"conds:{isabel}")]:
        chosen.use(voice)
        assert model.conds == conds
    model.prepare_conditionals.assert_called_once_with(isabel)


def test_configured_command_wins(voices):
    assert vc.tts_command({**DEFAULT_CONFIG, "tts_server_command": ["x"]}) == ["x"]


@pytest.mark.parametrize("url", ["http://192.168.0.9:8765", "not a url"])
def test_remote_tts_url_is_never_started(voices, url):
    assert vc.tts_command({**DEFAULT_CONFIG, "tts_url": url}) is None


def test_without_uv_nothing_is_started(no_tts_process, monkeypatch):
    monkeypatch.setattr(vc.shutil, "which", lambda name: None)
    with patch("subprocess.Popen") as popen:
        vc.ensure_tts_server({**DEFAULT_CONFIG, "tts_url": _closed_port_url()},
                             log=lambda m: None)
    # Only the TTS server counts: telemetry threads left by other tests may
    # run nvidia-smi meanwhile.
    started = [c for c in popen.call_args_list if str(vc.TTS_SERVER_SCRIPT) in map(str, c.args[0])]
    assert started == []
    assert vc._tts_process is None


def test_bundled_server_declares_its_own_environment():
    header = vc.TTS_SERVER_SCRIPT.read_text(encoding="utf-8").split("# ///")[1]
    assert "chatterbox-tts" in header and "download.pytorch.org/whl/cu124" in header


class _Registry:
    """winreg stand-in holding the user's saved environment variables."""
    HKEY_CURRENT_USER = "HKCU"

    def __init__(self, values):
        self.values = values

    def OpenKey(self, root, path):
        from contextlib import nullcontext
        return nullcontext(path)

    def QueryValueEx(self, key, name):
        if name not in self.values:
            raise FileNotFoundError(name)
        return self.values[name], 1


def test_saved_variables_are_used_when_the_terminal_predates_them(monkeypatch, tmp_path):
    """A terminal opened before MODELS_DIR/HF_HOME were set does not pass
    them on; the app must still use the folders they point to."""
    import importlib
    from debora_whisper import paths
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "winreg", _Registry(
        {"MODELS_DIR": str(tmp_path), "HF_HOME": str(tmp_path / "huggingface")}))
    monkeypatch.delenv("MODELS_DIR", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    try:
        importlib.reload(paths)
        assert paths.MODEL_DIR == tmp_path / "debora-whisper" / "models"
        assert paths.VOICES_DIR == tmp_path / "voices"
        import os
        assert os.environ["HF_HOME"] == str(tmp_path / "huggingface")
    finally:
        monkeypatch.undo()
        importlib.reload(paths)


def test_paths_follow_models_dir(monkeypatch, tmp_path):
    import importlib
    from debora_whisper import paths
    monkeypatch.setitem(sys.modules, "winreg", _Registry({}))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    try:
        importlib.reload(paths)
        assert paths.MODEL_DIR == tmp_path / "debora-whisper" / "models"
        assert paths.CACHE_DIR == tmp_path / "debora-whisper" / "ov-cache"
        assert paths.VOICES_DIR == tmp_path / "voices"
        assert paths.CONFIG_FILE == Path.home() / ".debora" / "config.json"
        monkeypatch.delenv("MODELS_DIR")
        importlib.reload(paths)
        assert paths.MODEL_DIR == Path.home() / ".debora" / "models"
    finally:
        monkeypatch.undo()
        importlib.reload(paths)


def test_engine_waits_for_its_speech_model():
    app = DictationApp({**DEFAULT_CONFIG})
    app._model_ready.set()
    assert app._wait_speech_model()
    app._model_ready.clear()
    app._stopping.set()
    assert not app._wait_speech_model()


def test_each_spoken_sentence_is_logged(server):
    logs = []
    chat, _ = _chat(server, FakeLLM(["A capital é Canberra. Fica no sul do país."]))
    chat.log = logs.append
    chat.respond("capital da austrália")
    assert [m for m in logs if m.startswith("Voice chat: reply ")] == [
        "Voice chat: reply 'A capital é Canberra.'",
        "Voice chat: reply 'Fica no sul do país.'"]


def test_tts_timings_follow_each_chunk_through_playback(server):
    logs = []
    chat, played = _chat(server, FakeLLM(["A capital é Canberra. Fica no sul do país."]))
    chat.log = logs.append
    chat.respond("capital")
    chunks = [line for line in logs if " chunk=" in line]
    assert len(chunks) == len(played) == 2
    for line in chunks:
        times = {key: float(value) for key, value in re.findall(r"(\w+)=(\d+\.\d+)s", line)}
        assert times["enqueued"] <= times["synth_start"] <= times["synth_end"] <= times["play"]
        assert times["audio"] == pytest.approx(0.1)
    assert len([line for line in logs if " first_audio=" in line]) == 1
    assert not any("first_audio=-" in line for line in logs)


@pytest.mark.parametrize("boundary", [",", ";", ":", " —"])
def test_only_the_first_reply_chunk_splits_at_a_clause(server, boundary):
    head = "Eu posso explicar isso" + boundary
    tail = " porque temos bastante tempo para conversar. "
    later = "Agora temos outra frase, que deve continuar inteira."
    chat, _ = _chat(server, FakeLLM([head + " ", tail, later]))
    chat.respond("explique")
    assert _tts_texts(server) == [vc.speakable(head), tail.strip(), later]


def test_first_chunk_uses_twelve_complete_words_without_punctuation():
    head = "uma duas três quatro cinco seis sete oito nove dez onze doze"
    assert vc.split_sentences(head + " tre", first=True) == ([head], "tre")
    assert vc.split_sentences(head[:-1], first=True) == ([], head[:-1])
    # A clause later in the same delta must not move the cut past word 12.
    assert vc.split_sentences(head + " treze quatorze, depois", first=True) == (
        [head], "treze quatorze, depois")
    assert vc.split_sentences("Eu vi, mas ainda não terminei", first=True)[0] == []
    assert vc.split_sentences("Eu posso explicar — depois", first=True)[0] == []
    assert vc.split_sentences("a b c d, resto", first=True)[0] == []


def test_warmup_discards_audio_and_caches_waiting_phrase(server):
    chat, played = _chat(server, FakeLLM())
    for _ in range(2):
        worker = chat.warm_tts()
        worker.join(5)
        assert not worker.is_alive()
    assert _tts_texts(server) == ["Olá, estou pronta.", "Um instante."]
    assert not played
    assert chat._waiting_audio(chat.config) is chat._waiting_audio(chat.config)
    assert len(server.requests) == 2


@pytest.mark.parametrize("key,value", [("tts_voice", "another"), ("language", "en"),
                                      ("tts_url", "http://127.0.0.1:9999")])
def test_waiting_cache_is_invalidated_by_settings(key, value, monkeypatch):
    synth = MagicMock(side_effect=[(AUDIO, 16000), (AUDIO.copy(), 16000)])
    monkeypatch.setattr(vc, "synthesize", synth)
    chat = vc.VoiceChat({**DEFAULT_CONFIG, "language": "pt"})
    old = chat._waiting_audio(chat.config)
    assert chat._waiting_audio(chat.config) is old
    chat.config[key] = value
    assert chat._waiting_audio(chat.config) is not old
    assert synth.call_count == 2


def test_warmup_returns_while_server_loads_and_failure_does_not_poison_cache(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def wait(config):
        entered.set()
        assert release.wait(5)
        return True

    monkeypatch.setattr(vc, "wait_tts_server", wait)
    synth = MagicMock(side_effect=[vc.TTSRejected("cold"), (AUDIO, 16000)])
    monkeypatch.setattr(vc, "synthesize", synth)
    chat = vc.VoiceChat(dict(DEFAULT_CONFIG), log=lambda m: None)
    worker = chat.warm_tts()
    try:
        assert entered.wait(5) and worker.is_alive()
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert chat._waiting_audio(chat.config)[1] == 16000
    assert synth.call_count == 2


def test_waiting_phrase_is_reused_across_turns(server, monkeypatch):
    # Trigger feedback deterministically, before the writer's first delta.
    def timer(delay, callback):
        return MagicMock(start=callback)

    monkeypatch.setattr(vc.threading, "Timer", timer)
    chat, played = _chat(server, FakeLLM(["A capital é Canberra."]), voice_chat_backend="claude")
    logs = []
    chat.log = logs.append
    chat.respond("capital")
    chat.respond("capital")
    assert _tts_texts(server) == ["Um instante.", "A capital é Canberra.", "A capital é Canberra."]
    assert len(played) == 4
    assert len([line for line in logs if " cached" in line]) == 1


# --- Switching voice chat while running ---------------------------------------

@pytest.fixture
def loader(monkeypatch):
    """The engine's LLM loading, recorded instead of run."""
    calls = []
    monkeypatch.setattr(de, "ensure_tts_server", lambda *a: calls.append("tts"))
    # Never warm the user's real server from the engine loading tests.
    monkeypatch.setattr(vc.VoiceChat, "warm_tts", MagicMock())
    monkeypatch.setattr(de, "download_llm", lambda *a: calls.append("download"))
    monkeypatch.setattr(de, "llm_loaded", lambda config: False)

    def load_llm(config, log, log_path, stop):
        calls.append(("llm", de._inference_gate.locked()))
        return object()

    monkeypatch.setattr(de, "load_llm", load_llm)
    return calls


def _notices(app):
    notices = []
    app.add_callback(lambda s, d: d.get("notice") and notices.append(d["notice"]))
    return notices


def _wait_for(condition, timeout=5):
    deadline = time.time() + timeout
    while not condition():
        assert time.time() < deadline
        time.sleep(0.02)


def test_switching_on_loads_the_llm_in_the_background(loader):
    app = _app()
    notices = _notices(app)
    app.set_voice_chat(True)
    _wait_for(lambda: "Voice chat ready" in notices)
    assert app.config["voice_chat"] is True
    # Downloaded first, then compiled with every transcription held back.
    assert loader == ["tts", "download", ("llm", True)]
    vc.VoiceChat.warm_tts.assert_called_once()
    assert notices == ["Voice chat: loading the LLM...", "Voice chat ready"]


def test_the_llm_waits_for_the_speech_model(loader):
    """Compiling the LLM on the iGPU during the NPU's warmup lost the NPU."""
    app = _app()
    app._model_ready.clear()
    app.set_voice_chat(True)
    time.sleep(0.3)
    assert loader == ["tts"]
    app._model_ready.set()
    _wait_for(lambda: len(loader) == 3)


def test_a_loaded_llm_is_ready_at_once(loader, monkeypatch):
    monkeypatch.setattr(de, "llm_loaded", lambda config: True)
    app = _app()
    notices = _notices(app)
    app.set_voice_chat(True)
    _wait_for(lambda: notices)
    assert notices == ["Voice chat ready"]
    assert loader == ["tts"]


def test_failed_llm_load_is_told(loader, monkeypatch):
    monkeypatch.setattr(de, "load_llm", MagicMock(side_effect=RuntimeError("no GPU")))
    app = _app()
    notices = _notices(app)
    app.set_voice_chat(True)
    _wait_for(lambda: len(notices) == 2)
    assert notices[-1] == "Voice chat: the LLM did not load (see app.log)"


def test_transcription_waits_while_the_llm_compiles(typed):
    app = _app()
    app.whisper.transcribe.return_value = "ola"
    with de._inference_gate:
        worker = threading.Thread(target=app._finish_recording,
                                  kwargs={"audio": AUDIO, "is_final": True})
        worker.start()
        time.sleep(0.3)
        app.whisper.transcribe.assert_not_called()
    worker.join(5)
    app.whisper.transcribe.assert_called_once()
    assert typed == ["ola "]


def test_switching_off_stops_the_reply_and_forgets_the_conversation():
    app = _app(voice_chat=True)
    app.voice_chat._history = [{"role": "user", "content": "oi"}]
    with patch.object(app.voice_chat, "interrupt") as interrupt:
        app.set_voice_chat(False)
    interrupt.assert_called_once()
    assert app.voice_chat._history == []
    assert app.config["voice_chat"] is False


def test_sentence_typed_as_dictation_goes_to_the_llm_after_the_switch(typed):
    app = _app(voice_chat=False, continuous_listening=True, inline_drafts=True)
    app.is_recording = True
    _say(app, "ola tudo", is_final=False)  # typed as a dictation draft
    app.config["voice_chat"] = True
    with patch.object(app.voice_chat, "respond", return_value="Oi!") as respond:
        _say(app, "ola tudo bem")
    respond.assert_called_once()
    assert typed == [("draft", "ola tudo... "), ("delete", len("ola tudo... "))]


def _gui(**config):
    from debora_whisper.app import GUIApp
    gui = GUIApp.__new__(GUIApp)
    gui._config = {**DEFAULT_CONFIG, **config}
    gui._engine = MagicMock()
    gui._engine.stop_if_idle.return_value = None
    gui._overlay, gui._tray = MagicMock(), MagicMock()
    gui._settings_win = MagicMock(is_open=True)
    gui._settings_status, gui._settings_set_apply = MagicMock(), MagicMock()
    return gui


def test_tray_item_switches_without_a_new_engine():
    gui = _gui()
    engine = gui._engine
    with patch.dict(gui._set_voice_chat.__globals__, {"save_config": MagicMock()}) as g:
        gui._set_voice_chat(True)
        g["save_config"].assert_called_once()
    assert gui._config["voice_chat"] is True
    engine.set_voice_chat.assert_called_once_with(True)
    gui._settings_win.set_voice_chat.assert_called_once_with(True)
    assert gui._engine is engine


def test_settings_switch_voice_chat_without_a_new_engine():
    gui = _gui()
    engine = gui._engine
    factory = MagicMock()
    with patch.dict(gui._on_settings_apply.__globals__,
                    {"save_config": MagicMock(), "DictationApp": factory}):
        gui._on_settings_apply({**gui._config, "voice_chat": True})
    engine.set_voice_chat.assert_called_once_with(True)
    engine.stop_if_idle.assert_not_called()
    factory.assert_not_called()


def test_rebuilt_engine_starts_voice_chat_itself():
    gui = _gui()
    old = gui._engine
    with patch.dict(gui._on_settings_apply.__globals__,
                    {"save_config": MagicMock(), "DictationApp": MagicMock(),
                     "is_model_downloaded": lambda size: True}):
        gui._on_settings_apply({**gui._config, "voice_chat": True, "hotkey": "ctrl+alt+v"})
    old.set_voice_chat.assert_not_called()


def test_notice_shows_only_in_the_tray_without_changing_state():
    gui = _gui()
    gui._update_ui(de.AppState.RECORDING, {"notice": "Voice chat ready"})
    gui._overlay.show_notice.assert_not_called()
    gui._overlay.show_recording.assert_not_called()
    gui._tray.update_state.assert_called_once_with("recording", "Débora Whisper — Voice chat ready")


def test_talking_only_animates_the_mascot():
    gui = _gui()
    gui._update_ui(de.AppState.SPEAKING, {"talking": "debora", "active": True})
    gui._overlay.set_talking.assert_called_once_with("debora", True)
    gui._overlay.show_speaking.assert_not_called()
    gui._tray.update_state.assert_not_called()


def test_engine_wires_both_voices_to_the_ui():
    events = []
    app = de.DictationApp.__new__(de.DictationApp)
    app._stopping = threading.Event()
    app._state = de.AppState.RECORDING
    app._callbacks = [lambda state, data: events.append(data)]
    app._talking("user", True)
    assert events == [{"talking": "user", "active": True}]


# --- Latency and playback -------------------------------------------------------

def test_voice_chat_ends_a_sentence_after_a_shorter_silence():
    config = {"voice_chat": False}
    recorder = de.AudioRecorder(sample_rate=16000, config=config)
    assert recorder.end_silence_frames == int(16000 * 1.5)
    config["voice_chat"] = True  # switched while running
    assert recorder.end_silence_frames == int(16000 * 0.8)
    config["voice_chat_end_silence_seconds"] = 0.6
    assert recorder.end_silence_frames == int(16000 * 0.6)


def test_silence_around_a_clip_is_trimmed():
    rate = 1000
    speech = np.full(500, 0.2, dtype=np.float32)
    clip = np.concatenate([np.zeros(300, np.float32), speech, np.zeros(700, np.float32)])
    trimmed = vc.trim_silence(clip, rate)
    assert len(trimmed) == 500 + 2 * 50  # 50 ms kept on each side
    assert len(vc.trim_silence(np.zeros(100, np.float32), rate)) == 0


def test_hiss_after_the_speech_is_cut():
    rate = 1000
    speech = np.full(500, 0.2, dtype=np.float32)
    hiss = np.full(400, 0.008, dtype=np.float32)  # -42 dB, like Chatterbox's
    clip = np.concatenate([speech, np.zeros(1000, np.float32), hiss])
    trimmed = vc.trim_silence(clip, rate)
    assert len(trimmed) == 500 + 50  # the speech and 50 ms of its pause
    assert trimmed[-1] == 0 and trimmed[0] == np.float32(0.2)


def test_a_pause_before_more_speech_is_kept():
    rate = 1000
    speech = np.full(500, 0.2, dtype=np.float32)
    clip = np.concatenate([speech, np.zeros(1000, np.float32), speech])
    assert len(vc.trim_silence(clip, rate)) == 2000


@pytest.mark.parametrize("language, text, spoken", [
    ("pt", "Hoje é sexta-feira, dia 09 outubro de 2026, às 10h18.",
     "Hoje é sexta-feira, dia nove outubro de dois mil e vinte e seis, às dez e dezoito."),
    ("pt", "Em 09/10/2026, às 21:05, 1h ou 2h.",
     "Em nove de outubro de dois mil e vinte e seis, às vinte e uma e cinco, "
     "uma hora ou duas horas."),
    ("pt", "O 1º lugar, 3,5% e 1.500 reais.",
     "O primeiro lugar, três vírgula cinco por cento e mil e quinhentos reais."),
    ("en", "It is 10/09/2026 at 10:05, 3.5%.",
     "It is October ninth, twenty twenty-six at ten oh five, three point five percent."),
    ("auto", "Oi 2", "Oi 2"),
    ("xx", "Oi 2", "Oi 2"),
])
def test_numbers_are_spoken_as_words(language, text, spoken):
    assert vc.spoken_numbers(text, language) == spoken


@pytest.mark.parametrize("sentence, offer", [
    ("Como posso te ajudar hoje?", True),
    ("Pode dizer o que quer que eu faça?", True),
    ("Vai me dizer o que precisa?", True),
    ("O que deseja fazer agora?", True),
    ("How can I help you today?", True),
    ("O que você quer dizer com isso?", False),
    ("Posso ajudar com o ditado.", False),
    ("Pode repetir?", False),
])
def test_offers_of_help_are_recognized(sentence, offer):
    assert vc.is_help_offer(sentence) is offer


def test_a_closing_offer_of_help_is_neither_said_nor_remembered(server):
    llm = FakeLLM(["Que legal, Alexandre! Pode dizer o que quer que eu faça?"])
    chat, _ = _chat(server, llm)
    assert chat.respond("eu sou o Alexandre") == "Que legal, Alexandre!"
    assert _tts_texts(server) == ["Que legal, Alexandre!"]
    assert chat._history[-1] == {"role": "assistant", "content": "Que legal, Alexandre!"}


def test_a_reply_that_is_only_an_offer_of_help_is_kept(server):
    chat, _ = _chat(server, FakeLLM(["Como posso ajudar você hoje?"]))
    assert chat.respond("oi Débora") == "Como posso ajudar você hoje?"


def test_the_overlay_keeps_digits_but_the_tts_gets_words(server):
    chat, _ = _chat(server, FakeLLM(["Hoje é dia 9 de outubro, às 10h18."]))
    assert chat.respond("que dia é hoje") == "Hoje é dia 9 de outubro, às 10h18."
    assert _tts_texts(server) == ["Hoje é dia nove de outubro, às dez e dezoito."]


class FakeStream:
    instances = []

    def __init__(self, samplerate, channels, dtype):
        self.rate, self.written, self.ended = samplerate, [], None
        FakeStream.instances.append(self)

    def start(self):
        pass

    def write(self, data):
        self.written.append(len(data))

    def stop(self):
        self.ended = "drained"

    def abort(self):
        self.ended = "dropped"

    def close(self):
        pass


@pytest.fixture
def stream(monkeypatch):
    FakeStream.instances = []
    monkeypatch.setitem(sys.modules, "sounddevice", MagicMock(OutputStream=FakeStream))
    return FakeStream.instances


def test_clips_of_a_reply_share_one_stream_with_a_pause_between(stream):
    player, stop = vc.StreamPlayer(), threading.Event()
    player(np.ones(24000, np.float32), 24000, stop)
    player(np.ones(24000, np.float32), 24000, stop)
    player.close(interrupted=False)
    assert len(stream) == 1
    gap = int(24000 * vc.SENTENCE_GAP_SECONDS)
    assert sum(stream[0].written) == 2 * 24000 + gap
    assert stream[0].ended == "drained"  # the last clip plays to its end


def test_interrupt_drops_the_queued_audio(stream):
    player, stop = vc.StreamPlayer(), threading.Event()
    stop.set()
    player(np.ones(24000, np.float32), 24000, stop)
    player.close(interrupted=True)
    assert stream[0].written == [] and stream[0].ended == "dropped"


def test_reply_plays_through_one_stream_player(server, stream, monkeypatch):
    chat = vc.VoiceChat(_cfg(server), log=lambda m: None,
                        llm=FakeLLM(["A capital é Canberra. Fica no sul do país."]))
    assert chat.respond("capital") == "A capital é Canberra. Fica no sul do país."
    assert len(stream) == 1 and stream[0].ended == "drained"


@pytest.mark.shipped_defaults
def test_voice_chat_with_turbo_is_what_users_get():
    assert DEFAULT_CONFIG["voice_chat"] is True
    assert DEFAULT_CONFIG["model_size"] == "turbo"
    de.validate_config(DEFAULT_CONFIG)


def test_debora_prompt_asks_for_speakable_text_only():
    prompt = vc.DEFAULT_VOICE_CHAT_PROMPT
    assert prompt.startswith("You are Débora")
    for rule in ("emoji", "markdown", "parentheses", "URLs", "percent"):
        assert rule in prompt


def test_debora_prompt_fixes_what_the_last_chat_got_wrong():
    prompt = vc.DEFAULT_VOICE_CHAT_PROMPT
    for rule in ("feminine", "ask them to repeat", "do not know the user's name",
                 "offer of help"):
        assert rule in prompt
    assert "most likely said" not in prompt
    pt = vc.VOICE_CHAT_PROMPTS["pt"]
    assert pt.startswith("Você é a Débora")
    for rule in ("feminino", "emoji", "peça para repetir", "não sabe o nome",
                 "qual é meu nome",
                 "oferecendo ajuda", "por cento"):
        assert rule in pt


@pytest.mark.parametrize("text", ["sobre o projeto...", "sobre o projeto…", "Eu queria, mas."])
def test_voice_chat_draft_updates_endpoint_through_asr(typed, text):
    app = _app(voice_chat=True)
    app.recorder = de.AudioRecorder(config=app.config)
    segment = app.recorder.endpoint.start()
    app.whisper.transcribe.return_value = text
    app._finish_recording(audio=AUDIO, is_final=False, segment_id=segment, audio_end=len(AUDIO))
    assert app.recorder.end_silence_frames == 2 * app.config["sample_rate"]


def test_claude_keeps_an_interrupted_reserved_message_and_drains_before_next(tmp_path):
    import queue
    from debora_whisper.harness import HarnessSession

    session = HarnessSession.__new__(HarnessSession)
    session.process = MagicMock()
    session.process.poll.return_value = None
    session.error = None
    session._turn_lock = threading.Lock()
    session._send = MagicMock()
    session._events = queue.Queue()
    session.log = MagicMock()
    session.session_file = tmp_path / "session.json"
    session.session_id = "test-session"
    session.key = (str(tmp_path),)
    first = threading.Event()
    first.set()  # interrupted before send starts, still preserve the user
    session._events.put({"type": "result"})
    session.send("sobre o projeto...", MagicMock(), first)
    assert session._events.empty()
    session._events.put({"type": "result"})
    session.send("quero mudar a interface", MagicMock(), threading.Event())
    sent = [call.args[0] for call in session._send.call_args_list]
    assert [m["type"] for m in sent] == ["user", "control_request", "user"]
    assert [m["message"]["content"] for m in sent if m["type"] == "user"] == [
        "sobre o projeto...", "quero mudar a interface"]


def test_on_audio_marks_when_her_voice_is_heard(server):
    """The mascot zooms from her first clip to the end of the reply."""
    events = []
    chat, played = _chat(server, FakeLLM(["A capital é Canberra. ", "Fica no sul, perto do mar."]))
    chat.on_audio = lambda active: events.append((active, len(played)))
    chat.respond("qual é a capital da austrália")
    assert events[0] == (True, 0)  # before the first clip plays
    assert events[-1] == (False, 2)  # after the last one
    assert all(a != b for (a, _), (b, _) in zip(events, events[1:]))  # once per change


def _harness_session(tmp_path, saved, events, closed=None):
    """closed: set it to close Claude's output after events (None: at once)."""
    import io
    import json as _json
    from debora_whisper.harness import HarnessSession, harness_key
    config = {**DEFAULT_CONFIG, "harness_cwd": str(tmp_path)}
    key = harness_key(config)[0]
    session_file = tmp_path / "harness_session.json"
    session_file.write_text(_json.dumps({key: saved}), encoding="utf-8")

    def output():
        for event in events:
            yield _json.dumps(event) + "\n"
        if closed is not None:
            closed.wait(5)

    process = MagicMock(pid=1, stdin=io.StringIO(), stdout=output(), stderr=iter([]))
    process.poll.return_value = None
    commands = []
    session = HarnessSession(
        config, log=lambda m: None,
        command_factory=lambda config, sid, resume: commands.append((sid, resume)) or ["claude"],
        process_factory=lambda *a, **kw: process, session_file=session_file)
    return session, commands, lambda: _json.loads(
        session_file.read_text(encoding="utf-8")).get(key)


def test_claude_session_that_does_not_resume_is_forgotten(tmp_path):
    saved = "6f1c2d4e-0000-4000-8000-000000000001"
    session, commands, stored = _harness_session(tmp_path, saved, [])  # exits at once
    assert commands == [(saved, True)]
    with pytest.raises(RuntimeError):
        session.send("oi", MagicMock(), threading.Event())
    assert stored() is None


def test_claude_session_that_resumed_is_kept_after_a_later_failure(tmp_path):
    saved = "6f1c2d4e-0000-4000-8000-000000000002"
    closed = threading.Event()
    session, _, stored = _harness_session(
        tmp_path, saved, [{"type": "result", "session_id": saved}], closed)
    session.send("oi", MagicMock(), threading.Event())
    closed.set()
    with pytest.raises(RuntimeError):
        session.send("de novo", MagicMock(), threading.Event())
    assert stored() == saved


def test_invalid_saved_claude_session_is_never_passed_as_a_flag(tmp_path):
    _, commands, _ = _harness_session(tmp_path, "--dangerously-skip-permissions", [])
    sid, resume = commands[0]
    assert resume is False and not sid.startswith("-")


@pytest.mark.parametrize("text, started, echo", [
    ("sim", 10.5, True),     # heard while she was saying it
    ("sim", 12.5, False),    # her sentence ended: the user answered
    ("quê", 13.0, False),
    ("ela disse sim mesmo", 12.5, True),  # longer text keeps the 2 s room tail
])
def test_short_answers_after_her_sentence_are_not_echo(text, started, echo, monkeypatch):
    chat = vc.VoiceChat({"language": "pt"})
    chat._spoken = [(10.0, 12.0, "Sim, ela disse que quê mesmo")]
    assert chat.is_echo(text, started, started + 0.4) is echo


def test_her_text_enters_the_balloon_with_her_voice(server):
    shown = []
    chat, played = _chat(server, FakeLLM(["A capital é Canberra. ", "Fica no sul, perto do mar."]))
    chat.respond("qual é a capital da austrália",
                 on_reply=lambda text: shown.append((text, len(played))))
    # Each sentence appears as its clip starts, never while it is synthesized.
    assert shown == [("A capital é Canberra.", 0),
                     ("A capital é Canberra. Fica no sul, perto do mar.", 1)]


def test_her_text_still_shows_without_a_voice(server, monkeypatch):
    monkeypatch.setattr(vc, "wait_tts_server", lambda config, *a, **kw: False)
    shown = []
    chat, played = _chat(server, FakeLLM(["A capital é Canberra."]))
    chat.respond("capital", on_reply=shown.append)
    assert shown == ["A capital é Canberra."] and played == []
