"""Voice chat: each final transcription goes to a local LLM (OpenVINO GenAI,
in npu_whisper/llm_server.py's process) and its reply is spoken aloud by the Chatterbox TTS server
(npu_whisper/tts_server.py, in its own uv environment).

The reply is streamed and spoken sentence by sentence: the first sentence
plays while the LLM still writes the rest and the TTS renders the next one.
"""
import atexit
import io
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path

from npu_whisper import paths
from npu_whisper.processes import NO_WINDOW, kill_tree, python_executable

DEFAULT_VOICE_CHAT_PROMPT = (
    "You are a voice assistant. The user talks to you through a speech "
    "recognizer, so their text may contain recognition mistakes: read it for "
    "what they most likely said. Your reply is read aloud, so answer in the "
    "user's language, briefly (one to three short sentences), in plain spoken "
    "language: no markdown, lists, emoji, code or URLs."
)
LLM_MAX_TOKENS = 400
# Earlier turns sent with each request, and how long a pause starts a fresh
# conversation.
HISTORY_TURNS = 8
HISTORY_IDLE_RESET_SECONDS = 600
TTS_MAX_RESPONSE_BYTES = 50_000_000
# How long the first reply waits for a TTS server this process started
# (Chatterbox loads in ~15 s; uv builds its environment on the very first run).
TTS_STARTUP_SECONDS = 180
TTS_SERVER_SCRIPT = Path(__file__).with_name("tts_server.py")
# Shorter sentences wait for the next one: "Opa!" alone sounds clipped, and
# Chatterbox cannot speak a bare "OK" at all.
TTS_MIN_CHARS = 16

# A sentence ends at . ! ? … (or a line break) followed by whitespace.
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")


def is_http_url(url) -> bool:
    return isinstance(url, str) and re.match(r"https?://", url, re.IGNORECASE) is not None


def _opener():
    # No proxies: the reply goes straight to the local TTS server, never
    # through HTTP(S)_PROXY from the environment.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def speakable(text: str) -> str:
    """Text for the TTS: no markdown marks, control characters or runs of
    spaces."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"[*_#`~>|]+", "", text)
    text = "".join(" " if unicodedata.category(c)[0] in "CZ" and c != "‍" else c
                   for c in text)
    return re.sub(r" {2,}", " ", text).strip()


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Complete sentences in buffer, and the unfinished rest."""
    parts = _SENTENCE_END.split(buffer)
    return [p for p in parts[:-1] if p.strip()], parts[-1]


# ---------------------------------------------------------------------------
# LLM (OpenVINO GenAI, in npu_whisper/llm_server.py's process)
# ---------------------------------------------------------------------------
# A first compile on the Arc iGPU, without the cache, can take minutes.
LLM_LOAD_TIMEOUT = 600


def _model_path(model: str, log) -> str:
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(model, local_files_only=True)
    except Exception:
        log(f"Voice chat: downloading {model} (first use, several GB)...")
        return snapshot_download(model)


def download_llm(config: dict, log=print) -> str:
    """The local directory of llm_model, downloaded on first use."""
    return _model_path(config["llm_model"], log)


def llm_command(model_dir: str, device: str) -> list[str]:
    return [python_executable(), "-m", "npu_whisper.llm_server", model_dir, device,
            str(paths.CACHE_DIR)]


class LLMProcess:
    """llm_server.py running one model; replies stream back over its stdout."""

    def __init__(self, key, command: list[str], log=print, log_path=None):
        self.key, self.log = key, log
        self.device: str | None = None
        self.error: str | None = None
        self.loaded = threading.Event()  # ready, failed or exited
        self._replies: dict[int, queue.Queue] = {}
        self._next_id = 0
        self._send_lock = threading.Lock()
        output = open(log_path, "ab") if log_path else subprocess.DEVNULL
        try:
            self.process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=output,
                encoding="utf-8", errors="replace", bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
                creationflags=NO_WINDOW)
        finally:
            if log_path:
                output.close()
        self._start = time.time()
        threading.Thread(target=self._read, daemon=True).start()

    @property
    def running(self) -> bool:
        return self.process.poll() is None and self.error is None

    def _read(self):
        for line in self.process.stdout:
            try:
                message = json.loads(line)
            except ValueError:
                message = None
            if not isinstance(message, dict):
                if line.strip():
                    self.log(f"Voice chat: LLM process: {line.strip()}")
            elif "log" in message:
                self.log(f"Voice chat: {message['log']}")
            elif "ready" in message:
                self.device = message["ready"]
                self.log(f"Voice chat: {self.key[0]} loaded on {self.device} in "
                         f"{time.time() - self._start:.1f}s (process {self.process.pid})")
                self.loaded.set()
            elif "failed" in message:
                self.error = message["failed"]
            elif (replies := self._replies.get(message.get("id"))) is not None:
                replies.put(message)
        code = self.process.wait()
        if self.error is None:
            self.error = f"the LLM process exited (code {code})"
        self.loaded.set()
        for replies in list(self._replies.values()):
            replies.put({"error": self.error})

    def wait_loaded(self, stop: threading.Event | None = None,
                    timeout=LLM_LOAD_TIMEOUT) -> bool:
        """True once loaded, False if stop was set first; raises if it
        failed or took longer than timeout (then it is killed)."""
        deadline = time.time() + timeout
        while not self.loaded.wait(0.2):
            if stop is not None and stop.is_set():
                return False
            if time.time() > deadline:
                self.error = f"the LLM did not load in {timeout}s; process killed"
                kill_tree(self.process)
                raise RuntimeError(self.error)
        if self.error is not None:
            raise RuntimeError(self.error)
        return True

    def _send(self, message: dict):
        try:
            with self._send_lock:
                self.process.stdin.write(json.dumps(message) + "\n")
                self.process.stdin.flush()
        except (OSError, ValueError):
            raise RuntimeError(self.error or "the LLM process is gone") from None

    def generate(self, messages: list[dict], on_text, stop: threading.Event):
        """Stream the reply into on_text(chunk) until it ends or stop is set."""
        with self._send_lock:
            self._next_id += 1
            rid = self._next_id
        replies = self._replies[rid] = queue.Queue()
        try:
            self._send({"id": rid, "messages": messages, "max_new_tokens": LLM_MAX_TOKENS})
            while True:
                try:
                    message = replies.get(timeout=0.1)
                except queue.Empty:
                    message = None
                if stop.is_set():
                    # The process drops the rest at its next token.
                    self._send({"cancel": rid})
                    return
                if message is None:
                    continue
                if "error" in message:
                    raise RuntimeError(message["error"])
                if message.get("done"):
                    return
                on_text(message["text"])
        finally:
            self._replies.pop(rid, None)

    def stop(self):
        """Close its stdin (it exits on its own), else kill it."""
        try:
            self.process.stdin.close()
        except OSError:
            pass
        try:
            self.process.wait(5)
        except subprocess.TimeoutExpired:
            kill_tree(self.process)


_llm_lock = threading.Lock()
_llm: LLMProcess | None = None  # one per app process, kept across engine rebuilds


def start_llm(config: dict, log=print, log_path=None) -> LLMProcess:
    """The process serving llm_model on llm_device, started unless one
    already runs (or loads) it."""
    global _llm
    key = (config["llm_model"], config["llm_device"])
    with _llm_lock:
        if _llm is not None and _llm.key == key and _llm.running:
            return _llm
        if _llm is not None:
            _llm.stop()
        path = download_llm(config, log)
        paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _llm = LLMProcess(key, llm_command(path, config["llm_device"]), log, log_path)
        atexit.register(stop_llm)
        log(f"Voice chat: loading {config['llm_model']} on {config['llm_device']} "
            f"in a separate process (pid {_llm.process.pid})")
        return _llm


def llm_loaded(config: dict) -> bool:
    """The LLM for config is loaded and its process still running."""
    llm = _llm
    return (llm is not None and llm.key == (config["llm_model"], config["llm_device"])
            and llm.running and llm.loaded.is_set())


def load_llm(config: dict, log=print, log_path=None, stop=None) -> "LLMProcess | None":
    """start_llm, then wait until it has loaded (None if stop was set first)."""
    llm = start_llm(config, log, log_path)
    return llm if llm.wait_loaded(stop) else None


def stop_llm():
    global _llm
    with _llm_lock:
        llm, _llm = _llm, None
    if llm is not None:
        llm.stop()


def generate_reply(messages: list[dict], config: dict, on_text, stop: threading.Event,
                   log=print, log_path=None):
    """Stream the reply to messages into on_text(chunk) until it ends or stop
    is set."""
    llm = load_llm(config, log, log_path, stop)
    if llm is not None:
        llm.generate(messages, on_text, stop)


# ---------------------------------------------------------------------------
# TTS server
# ---------------------------------------------------------------------------
class TTSRejected(Exception):
    """The TTS server answered but could not speak this text (Chatterbox
    fails on very short text such as "OK"): skip it, keep the voice on."""


def synthesize(text: str, config: dict):
    """(float32 samples, sample rate) for text from config["tts_url"]."""
    import numpy as np
    url = config["tts_url"]
    if not is_http_url(url):
        raise ValueError(f"tts_url {url!r} is not an http(s) URL")
    body = {"text": text}
    if config.get("language") and config["language"] != "auto":
        body["language"] = config["language"]
    request = urllib.request.Request(
        url.rstrip("/") + "/tts", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with _opener().open(request, timeout=config["tts_timeout_seconds"]) as response:
            raw = response.read(TTS_MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as e:
        try:
            message = json.loads(e.read(4096))["error"]
        except Exception:
            message = e.reason
        raise TTSRejected(f"HTTP {e.code}: {message}") from None
    if len(raw) > TTS_MAX_RESPONSE_BYTES:
        raise ValueError(f"TTS response larger than {TTS_MAX_RESPONSE_BYTES} bytes")
    with wave.open(io.BytesIO(raw)) as w:
        if w.getsampwidth() != 2:
            raise ValueError("TTS server must return 16-bit PCM WAV")
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        samples = samples.reshape(-1, w.getnchannels())[:, 0]
        return samples.astype(np.float32) / 32768.0, w.getframerate()


def resolve_voice(voice) -> Path | None:
    """tts_voice as a file: a path as given, or a bare name looked up as
    <name>.wav in the voices folder."""
    if not voice:
        return None
    path = Path(voice).expanduser()
    if path.suffix or len(path.parts) > 1:
        return path
    return paths.VOICES_DIR / f"{voice}.wav"


def tts_command(config: dict, log=print) -> list[str] | None:
    """tts_server_command, or uv running tts_server.py on tts_url's port
    with tts_voice (None if tts_url is not local or uv is missing)."""
    if config.get("tts_server_command"):
        return config["tts_server_command"]
    url = urllib.parse.urlparse(config.get("tts_url") or "")
    if url.hostname not in ("127.0.0.1", "localhost"):
        return None
    uv = shutil.which("uv")
    if not uv:
        log("Voice chat: uv not found on PATH; start the TTS server yourself "
            f"(uv run --script {TTS_SERVER_SCRIPT}).")
        return None
    command = [uv, "run", "--script", str(TTS_SERVER_SCRIPT), "--port", str(url.port or 80)]
    voice = resolve_voice(config.get("tts_voice"))
    if voice is not None and voice.is_file():
        command += ["--voice", str(voice)]
    elif voice is not None:
        log(f"Voice chat: voice {voice} not found; using Chatterbox's own voice.")
    if config.get("language") and config["language"] != "auto":
        command += ["--language", config["language"]]
    return command


_tts_lock = threading.Lock()
_tts_process: subprocess.Popen | None = None


def tts_server_up(config: dict) -> bool:
    try:
        url = config["tts_url"].rstrip("/") + "/health"
        with _opener().open(url, timeout=1) as response:
            return response.status == 200
    except Exception:
        return False


def ensure_tts_server(config: dict, log=print, log_path=None):
    """Start the TTS server (tts_command) if nothing answers at tts_url.
    One server per process: engine rebuilds (Settings) reuse it."""
    global _tts_process
    if not is_http_url(config.get("tts_url")):
        return
    with _tts_lock:
        if _tts_process is not None:
            if _tts_process.poll() is not None and not _tts_process.reported:
                # Starting it again would only fail again, costing the wait
                # on every reply: the replies go on as text until a restart.
                _tts_process.reported = True
                log(f"Voice chat: the TTS server exited (code {_tts_process.returncode}); "
                    f"see {log_path or 'its output'}. Restart the app after fixing it.")
            return
        if tts_server_up(config):
            return  # started by hand, or left over from an earlier run
        command = tts_command(config, log)
        if not command:
            return
        output = open(log_path, "ab") if log_path else subprocess.DEVNULL
        try:
            _tts_process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
                creationflags=NO_WINDOW)
        except Exception as e:
            log(f"Voice chat: cannot start the TTS server ({e})")
            return
        finally:
            if log_path:
                output.close()
        _tts_process.reported = False
        atexit.register(stop_tts_server)
        log(f"Voice chat: starting the TTS server (pid {_tts_process.pid}); "
            f"output in {log_path or 'nowhere'}")


def wait_tts_server(config: dict, timeout=TTS_STARTUP_SECONDS) -> bool:
    """True once the TTS server answers; waits only while a server this
    process started is still loading."""
    deadline = time.time() + timeout
    while True:
        if tts_server_up(config):
            return True
        process = _tts_process
        if process is None or process.poll() is not None or time.time() > deadline:
            return False
        time.sleep(0.5)


def stop_tts_server():
    """Stop the TTS server this process started, if any."""
    global _tts_process
    with _tts_lock:
        process, _tts_process = _tts_process, None
    if process is None:
        return
    # uv runs the server as a child process: end the whole tree, or the
    # server keeps its GPU memory after uv is gone.
    kill_tree(process)
    try:
        process.wait(5)
    except subprocess.TimeoutExpired:
        process.kill()


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------
class VoiceChat:
    """A conversation with the LLM, spoken through the TTS server.

    respond() blocks until the reply has been spoken (or interrupt() is
    called); it is meant to run on the engine's transcription thread.
    llm(messages, on_text, stop) replaces the OpenVINO model (tests)."""

    def __init__(self, config: dict, log=print, play=None, tts_log_path=None, llm=None,
                 llm_log_path=None):
        self.config = config
        self.log = log
        self.tts_log_path = tts_log_path
        self._play = play or _play_interruptible
        self._llm = llm or (lambda messages, on_text, stop: generate_reply(
            messages, self.config, on_text, stop, self.log, llm_log_path))
        self._history: list[dict] = []
        self._last_turn = 0.0
        self._interrupt = threading.Event()
        self.speaking = False

    def reset(self):
        self._history = []

    def interrupt(self):
        """Stop the reply in progress: no more text, synthesis or audio."""
        self._interrupt.set()

    def _messages(self, text: str) -> list[dict]:
        if time.time() - self._last_turn > HISTORY_IDLE_RESET_SECONDS:
            self._history = []
        prompt = self.config.get("llm_prompt") or DEFAULT_VOICE_CHAT_PROMPT
        language = self.config.get("language")
        if language and language != "auto":
            # A short "Sim." alone does not tell the model the language:
            # it answered in English.
            from npu_whisper.dictation_engine import LANGUAGES
            name = LANGUAGES.get(language, language)
            prompt += f" The user speaks {name}: reply in {name}."
        recent = self._history[-2 * HISTORY_TURNS:]
        return [{"role": "system", "content": prompt}, *recent,
                {"role": "user", "content": text}]

    def respond(self, text: str, on_reply=None) -> str:
        """Ask the LLM, speak its reply and return the text spoken so far
        ("" if the LLM failed). on_reply(text) gets the reply as it grows."""
        # A fresh event per turn: an interrupted turn's threads keep seeing
        # theirs set, even after the next turn starts.
        stop = self._interrupt = threading.Event()
        # Voice chat may have been switched on in Settings since startup.
        ensure_tts_server(self.config, self.log, self.tts_log_path)
        messages = self._messages(text)
        sentences: queue.Queue = queue.Queue()
        # Unbounded: after an interrupt nobody takes clips, and the renderer
        # must still be able to finish.
        clips: queue.Queue = queue.Queue()
        # buffer: the sentence being written; short: finished sentences too
        # short to send alone.
        state = {"reply": "", "llm_error": None, "buffer": "", "short": "", "tokens": False}
        start = time.time()

        def on_text(chunk):
            if not state["tokens"]:
                state["tokens"] = True
                self.log(f"Voice chat: first token after {time.time() - start:.1f}s")
            done, state["buffer"] = split_sentences(state["buffer"] + chunk)
            for sentence in done:
                sentence = f"{state['short']} {sentence}".strip()
                if len(speakable(sentence)) < TTS_MIN_CHARS:
                    state["short"] = sentence
                else:
                    state["short"] = ""
                    sentences.put(sentence)

        def write():
            try:
                self._llm(messages, on_text, stop)
                rest = f"{state['short']} {state['buffer']}".strip()
                if rest and not stop.is_set():
                    sentences.put(rest)
            except Exception as e:
                state["llm_error"] = e
            finally:
                sentences.put(None)

        def render():
            tts_ok = None  # unknown until the first sentence
            try:
                while not stop.is_set():
                    sentence = sentences.get()
                    if sentence is None:
                        return
                    spoken = speakable(sentence)
                    if not spoken:
                        continue
                    state["reply"] = f"{state['reply']} {spoken}".strip()
                    if on_reply:
                        on_reply(state["reply"])
                    self.log(f"Voice chat: reply {spoken!r}")
                    if not any(c.isalnum() for c in spoken):
                        continue  # an emoji or punctuation: nothing to say
                    audio = None
                    if tts_ok is None:
                        tts_ok = wait_tts_server(self.config)
                        if not tts_ok:
                            self.log("Voice chat: TTS server not reachable; showing the reply only.")
                    if tts_ok:
                        try:
                            audio = synthesize(spoken, self.config)
                        except TTSRejected as e:
                            self.log(f"Voice chat: TTS skipped {spoken[:40]!r} ({e})")
                        except Exception as e:
                            # Keep showing the text even without a voice.
                            tts_ok = False
                            self.log(f"Voice chat: TTS failed ({e}); showing the reply only.")
                    if audio is not None:
                        clips.put(audio)
            finally:
                clips.put(None)

        writer = threading.Thread(target=write, daemon=True)
        renderer = threading.Thread(target=render, daemon=True)
        writer.start()
        renderer.start()
        self.speaking = True
        try:
            while True:
                clip = clips.get()
                if clip is None or stop.is_set():
                    break
                self._play(clip[0], clip[1], stop)
        finally:
            self.speaking = False
            stop.set()  # stops the writer and renderer threads

        if state["llm_error"] is not None and not state["reply"]:
            self.log(f"Voice chat: LLM failed ({state['llm_error']})")
            return ""
        reply = state["reply"]
        if reply:
            self._history += [{"role": "user", "content": text},
                              {"role": "assistant", "content": reply}]
            self._last_turn = time.time()
            self.log(f"Voice chat: replied in {time.time() - start:.1f}s")
        return reply


def _play_interruptible(samples, sample_rate, interrupt: threading.Event):
    import sounddevice as sd
    sd.play(samples, sample_rate)
    duration = len(samples) / sample_rate
    deadline = time.time() + duration + 0.5
    while time.time() < deadline:
        if interrupt.wait(0.05):
            sd.stop()
            return
        if not sd.get_stream().active:
            return
    sd.stop()
