"""config["llm_cleanup"]: the final text goes through a local LLM server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from npu_whisper import dictation_engine as de
from npu_whisper.dictation_engine import DictationApp, DEFAULT_CONFIG

AUDIO = np.zeros(16000, dtype=np.float32)


@pytest.fixture
def server():
    """An OpenAI-compatible /chat/completions stub; set .reply or .status."""
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            srv.requests.append((self.path, body))
            if srv.status != 200:
                self.send_error(srv.status)
                return
            payload = json.dumps({"choices": [{"message": {"content": srv.reply}}]})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload.encode())

        def log_message(self, *args):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    srv.requests, srv.reply, srv.status = [], "", 200
    srv.url = f"http://127.0.0.1:{srv.server_port}/v1"
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _cfg(url, **extra):
    return {**DEFAULT_CONFIG, "llm_cleanup": True, "llm_url": url, **extra}


def test_cleanup_sends_the_text_and_returns_the_reply(server):
    server.reply = "Abre o arquivo config.json."
    assert de.llm_cleanup("abre o arquivo config ponto json", _cfg(server.url)) == \
        "Abre o arquivo config.json."
    path, body = server.requests[0]
    assert path == "/v1/chat/completions"
    assert body["messages"][0] == {"role": "system", "content": de.DEFAULT_LLM_PROMPT}
    assert body["messages"][1] == {"role": "user", "content": "abre o arquivo config ponto json"}
    assert body["reasoning_effort"] == "low"
    assert "model" not in body  # the server uses whatever model is loaded


def test_configured_model_and_prompt_are_sent(server):
    server.reply = "ok"
    de.llm_cleanup("texto", _cfg(server.url + "/", llm_model="gpt-oss-20b",
                                 llm_prompt="Só pontue.", llm_reasoning_effort=None))
    path, body = server.requests[0]
    assert path == "/v1/chat/completions"
    assert body["model"] == "gpt-oss-20b"
    assert body["messages"][0]["content"] == "Só pontue."
    assert "reasoning_effort" not in body


def test_inline_reasoning_is_dropped(server):
    server.reply = "<think>The user said hello.</think>\nOlá!"
    assert de.llm_cleanup("olá", _cfg(server.url)) == "Olá!"


@pytest.mark.parametrize("status, reply", [(500, ""), (200, ""), (200, "<think>hm</think>")])
def test_server_error_or_empty_reply_keeps_the_raw_text(server, status, reply):
    server.status, server.reply = status, reply
    assert de.llm_cleanup("texto cru", _cfg(server.url)) == "texto cru"


def test_unreachable_server_keeps_the_raw_text():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # closed again: nothing listens there
    assert de.llm_cleanup("texto cru", _cfg(f"http://127.0.0.1:{port}/v1")) == "texto cru"


@pytest.mark.parametrize("key, value", [
    ("llm_url", "localhost:1234"), ("llm_url", None),
    ("llm_timeout_seconds", 0), ("llm_timeout_seconds", "30"),
])
def test_invalid_llm_settings_are_rejected(key, value):
    with pytest.raises(ValueError):
        de.validate_config({**DEFAULT_CONFIG, key: value})


# --- In the dictation flow ---------------------------------------------------

def _app(**config):
    app = DictationApp({**DEFAULT_CONFIG, "beep_on_start": False, **config})
    app.recorder = MagicMock()
    app.whisper = MagicMock()
    app._model_ready.set()
    return app


def _say(app, text, is_final=True):
    app.whisper.transcribe.return_value = text
    app._finish_recording(audio=AUDIO, is_final=is_final)


@pytest.fixture
def typed():
    out = []
    with patch.object(de, "type_text", side_effect=lambda t, auto_enter=False: out.append(t)), \
            patch.object(de, "type_draft_text", side_effect=lambda t: out.append(("draft", t))), \
            patch.object(de, "delete_text", side_effect=lambda n: out.append(("delete", n))), \
            patch.object(de, "get_input_target", return_value=("window", "field")):
        yield out


def test_final_text_is_typed_after_cleanup(typed):
    app = _app(llm_cleanup=True)
    with patch.object(de, "llm_cleanup", return_value="Olá, tudo bem?") as cleanup:
        _say(app, " ola tudo bem")
    cleanup.assert_called_once_with("ola tudo bem", app.config)
    assert typed == ["Olá, tudo bem? "]  # segments keep a trailing space
    assert app._history[-1]["text"] == "Olá, tudo bem? "


def test_cleanup_off_types_the_raw_text(typed):
    app = _app()
    with patch.object(de, "llm_cleanup") as cleanup:
        _say(app, "ola tudo bem")
    cleanup.assert_not_called()
    assert typed == ["ola tudo bem... "]


def test_drafts_stay_raw_and_the_final_replaces_them(typed):
    # Continuous listening: drafts are typed live; the cleaned final rewrites
    # the part of the draft that differs.
    app = _app(llm_cleanup=True, continuous_listening=True)
    with patch.object(de, "llm_cleanup", return_value="Olá, tudo bem?") as cleanup:
        _say(app, "ola tudo", is_final=False)
        cleanup.assert_not_called()
        _say(app, "ola tudo bem")
    cleanup.assert_called_once()
    assert typed == [("draft", "ola tudo... "), ("delete", len("ola tudo... ")), "Olá, tudo bem? "]


def test_hallucinations_and_silence_never_reach_the_llm(typed):
    app = _app(llm_cleanup=True)
    with patch.object(de, "llm_cleanup") as cleanup:
        _say(app, "Obrigado.")
        _say(app, "")
    cleanup.assert_not_called()
    assert typed == []
