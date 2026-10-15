"""Voice chat: each final transcription goes to a local LLM or Claude Code,
and its reply is spoken aloud by the configured TTS server
(debora_whisper/tts_server.py, in its own uv environment).

The reply is streamed and spoken sentence by sentence: the first sentence
plays while the LLM still writes the rest and the TTS renders the next one.
"""
import atexit
import http.client
import io
import json
import math
import os
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.parse
import wave
from dataclasses import dataclass
# Bound at import: tests replace threading.Timer to drive reply feedback.
from threading import Timer as _DeadlineTimer
from pathlib import Path

from debora_whisper import paths
from debora_whisper.processes import NO_WINDOW, kill_tree, python_executable

# Débora's persona. Everything she writes goes to a text-to-speech engine
# that reads characters literally, so the reply must be plain words only.
DEFAULT_VOICE_CHAT_PROMPT = (
    "You are Débora, the voice assistant of Débora Whisper, a dictation app "
    "running entirely on the user's computer. You are a woman: in languages "
    "with grammatical gender, always refer to yourself in the feminine, for "
    "example in Portuguese a Débora and sua assistente. You are warm, "
    "friendly and direct. The user talks to you through a speech recognizer, "
    "so their text may contain recognition mistakes. Fix obvious small "
    "mistakes silently, but when a sentence does not make sense, say you did "
    "not understand and ask them to repeat instead of guessing. You do not "
    "know the user's name unless they tell you. When the user asks about "
    "their own name, as in what is my name, they mean their name, not yours. "
    "Everything you write is converted to speech and nobody sees it, so "
    "write only words meant to be spoken. Answer in the user's language, "
    "briefly, in one to three short sentences. Use only letters, numbers and "
    "basic punctuation: periods, commas, question marks and exclamation "
    "marks. Never use emoji, emoticons, markdown, asterisks, hashes, "
    "bullets, numbered lists, headings, tables, parentheses, brackets, "
    "quotation marks, slashes, dashes as separators, code or URLs. Say "
    "symbols, units and abbreviations as words, for example percent "
    "instead of the percent sign and degrees instead of the degree sign, "
    "and write times, dates and amounts the way a person would say them. "
    "Instead of a list, say the items in one natural sentence. "
    "You have no internet or tools: when asked what you cannot know, such "
    "as the weather or the news, say so instead of guessing. "
    "Most important rule: answer and stop. Never end with an offer of help "
    "or a question to keep the chat going, such as how can I help, what can "
    "I do for you today or what would you like to do next. Every extra "
    "sentence takes time to speak."
)
# Her own words in a language, picked by the configured language; any other
# language uses the English prompt and is told which language to answer in.
VOICE_CHAT_PROMPTS = {
    "pt": (
        "Você é a Débora, a assistente de voz do Débora Whisper, um app de "
        "ditado que roda inteiro no computador do usuário. Você é mulher e "
        "sempre fala de si no feminino. Fale português do Brasil, de um jeito "
        "caloroso, simpático e direto, como numa conversa. O usuário fala com "
        "você por um reconhecedor de voz, então o texto dele pode ter erros de "
        "reconhecimento. Corrija em silêncio os errinhos óbvios, mas quando a "
        "frase não fizer sentido, diga que não entendeu e peça para repetir, "
        "sem adivinhar. Você não sabe o nome do usuário, a não ser que ele "
        "diga. Quando o usuário falar do nome dele, como em qual é meu nome, "
        "ele quer saber o nome dele, não o seu. "
        "Tudo o que você escreve vira fala e ninguém lê, então escreva só "
        "palavras para serem ditas, em uma a três frases curtas. Use só "
        "letras, números e pontuação básica: ponto, vírgula, ponto de "
        "interrogação e de exclamação. Nunca use emoji, emoticons, markdown, "
        "asteriscos, cerquilhas, marcadores, listas numeradas, títulos, "
        "tabelas, parênteses, colchetes, aspas, barras, travessões, código ou "
        "links. Diga símbolos, unidades e abreviações por extenso, por exemplo "
        "por cento no lugar do sinal de porcentagem e graus no lugar do sinal "
        "de grau, e escreva horas, datas e valores do jeito que uma pessoa "
        "fala. Em vez de uma lista, diga os itens numa frase natural. "
        "Você não tem internet nem ferramentas: quando perguntarem algo que "
        "você não tem como saber, como o clima ou as notícias, diga isso em "
        "vez de chutar. "
        "Regra mais importante: responda e pare. Nunca termine oferecendo "
        "ajuda ou puxando assunto, com frases como Como posso ajudar, Em que "
        "posso ajudar hoje, O que deseja fazer agora ou Vai me dizer o que "
        "precisa. Cada frase a mais demora para ser falada."
    ),
}
_PT_WEEKDAYS = ("segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
                "sexta-feira", "sábado", "domingo")
LLM_MAX_TOKENS = 400
# Earlier turns sent with each request, and how long a pause starts a fresh
# conversation.
HISTORY_TURNS = 8
HISTORY_IDLE_RESET_SECONDS = 600
TTS_MAX_RESPONSE_BYTES = 50_000_000
# How long the first reply waits for a TTS server this process started
# (Chatterbox loads in ~15 s; uv builds its environment on the very first run).
TTS_STARTUP_SECONDS = 180
# One health check, start to finish: a peer trickling bytes cannot stretch it.
TTS_HEALTH_SECONDS = 1.0
TTS_HEALTH_MAX_BYTES = 64_000
TTS_SERVER_SCRIPT = Path(__file__).with_name("tts_server.py")
TTS_SERVER_SCRIPTS = {
    "chatterbox": TTS_SERVER_SCRIPT,
    "qwen3": TTS_SERVER_SCRIPT.with_name("tts_server_qwen3.py"),
    "kokoro": TTS_SERVER_SCRIPT.with_name("tts_server_kokoro.py"),
    "xtts": TTS_SERVER_SCRIPT.with_name("tts_server_xtts.py"),
}
# Backends that need their startup reference voice: --voice is passed even
# when the file is missing, so /health explains it, and changing it restarts.
_STARTUP_VOICE_BACKENDS = ("qwen3", "xtts")
# Shorter sentences wait for the next one: "Opa!" alone sounds clipped, and
# Chatterbox cannot speak a bare "OK" at all.
TTS_MIN_CHARS = 16

# A sentence ends at . ! ? … (or a line break) followed by whitespace.
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")
# Emoji with their variation selectors, keycaps, skin tones and the zero
# width joiners between them: Chatterbox would read them out or mumble.
_EMOJI_CHAR = "\U0001F000-\U0001FAFF☀-➿⬀-⯿"
_EMOJI = re.compile(f"[{_EMOJI_CHAR}](?:[️⃣\U000E0020-\U000E007F]"
                    f"|‍?[{_EMOJI_CHAR}])*|️⃣?")


def is_http_url(url) -> bool:
    return isinstance(url, str) and re.match(r"https?://", url, re.IGNORECASE) is not None


def speakable(text: str) -> str:
    """Text for the TTS: no markdown marks, control characters or runs of
    spaces."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"(```|~~~).*?(?:\1|$)", "", text, flags=re.DOTALL)
    text = re.sub(r"(?m)^\s*(?:[-+*]|\d+[.)])\s+", "", text)
    text = re.sub(r"[*_#`~>|]+", "", text)
    text = "".join(" " if unicodedata.category(c)[0] in "CZ" and c != "‍" else c
                   for c in text)
    return re.sub(r" {2,}", " ", text).strip()


# Questions that only offer help or ask for the next task: "Como posso
# ajudar?", "Pode dizer o que quer que eu faça?". Clarifying questions such
# as "O que você quer dizer?" are not among them.
_HELP_OFFER = re.compile(
    r"\b(posso (te |lhe )?ajudar|em que (mais )?posso"
    r"|(pode|vai) (me )?dizer o que (você )?(quer|precisa|deseja)"
    r"|o que (você )?(quer|deseja|gostaria|precisa) que eu"
    r"|o que (você )?(quer|deseja|gostaria) (de )?fazer"
    r"|o que podemos fazer"
    r"|how (can|may) i (help|assist)|what can i do for you|anything else"
    r"|what would you like (me )?to do)\b", re.IGNORECASE)


def is_help_offer(sentence: str) -> bool:
    sentence = sentence.strip()
    return sentence.endswith("?") and _HELP_OFFER.search(sentence) is not None


def without_emoji(text: str) -> str:
    """The reply as said aloud: emoji are shown but never spoken."""
    return re.sub(r" {2,}", " ", _EMOJI.sub(" ", text)).strip()


# Chatterbox reads digits badly: "10h18" and "09/10/2026" came out garbled.
# Only the spoken text is spelled out; the overlay keeps the digits.
_NUM2WORDS_LANG = {"pt": "pt_BR"}
_MONTHS = {
    "pt": ("janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho",
           "agosto", "setembro", "outubro", "novembro", "dezembro"),
    "en": ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"),
}
_PERCENT = {"pt": "por cento", "en": "percent", "es": "por ciento",
            "fr": "pour cent", "it": "per cento", "de": "Prozent"}
# Languages that write 3.5 rather than 3,5.
_DOT_DECIMAL = {"en", "zh", "ja", "ko", "hi", "he", "ms", "sw"}
_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_TIME = re.compile(r"\b([01]?\d|2[0-3])(?::|h)([0-5]\d)\b")
_HOURS = re.compile(r"\b([01]?\d|2[0-3])h\b")


def spoken_numbers(text: str, language) -> str:
    """text with dates, times, percentages and numbers written as words in
    language; unchanged when num2words does not know the language."""
    if not language or language == "auto" or not re.search(r"\d", text):
        return text
    try:
        from num2words import num2words
        lang = _NUM2WORDS_LANG.get(language, language)
        num2words(1, lang=lang)
    except Exception:
        return text

    def words(n, to="cardinal"):
        return num2words(n, lang=lang, to=to)

    def standalone(replace):
        def convert(m):
            # Keep identifiers intact, but allow sentence punctuation and labels.
            before, after = m.string[:m.start()], m.string[m.end():]
            # A path separator before the number, or a colon after a file
            # name (app.py:10), is a reference; "Total:10" is a label.
            if (re.search(r"\w$|[./\\]$|[\d./\\:-]:+$|[./\\]\S*:+$", before)
                    or re.match(r"\w|[./\\]+(?=[\w-])|:+(?=[\d/\\.-])", after)):
                return m[0]
            if before.endswith("-") or after.startswith("-"):
                # A minus or numeric range is spoken; hyphenated filenames and
                # ISO dates stay intact, including later numeric components.
                token = (re.search(r"[\w./\\-]*$", before)[0] + m[0]
                         + re.match(r"[\w./\\-]*", after)[0])
                if (re.search(r"[^\W\d]", token)
                        or re.fullmatch(r"\d{4}-\d{2}-\d{2}", token.rstrip("."))):
                    return m[0]
            return replace(m)
        return convert

    def hour(h):
        spoken = words(int(h))
        if language == "pt":  # uma hora, duas horas, vinte e uma horas
            spoken = re.sub(r"\bum$", "uma", re.sub(r"\bdois$", "duas", spoken))
        return spoken

    def date(m):
        day, month, year = int(m[1]), int(m[2]), int(m[3])
        if language not in _MONTHS or not 1 <= month <= 12:
            return m[0].replace("/", " ")
        if language == "en":  # 10/09/2026 is October 9th
            day, month = month, day
            if not 1 <= month <= 12:
                return m[0].replace("/", " ")
            return f"{_MONTHS['en'][month - 1]} {words(day, 'ordinal')}, {words(year, 'year')}"
        return f"{words(day)} de {_MONTHS['pt'][month - 1]} de {words(year)}"

    def time_of_day(m):
        h, minutes = m[1], int(m[2])
        if language == "pt":
            return f"{hour(h)} horas" if minutes == 0 else f"{hour(h)} e {words(minutes)}"
        if language == "en":
            if minutes == 0:
                return f"{hour(h)} o'clock"
            return f"{hour(h)} {'oh ' if minutes < 10 else ''}{words(minutes)}"
        return f"{hour(h)} {words(minutes)}"

    def number(m):
        digits = m[0]
        thousands, decimal = (",", ".") if language in _DOT_DECIMAL else (".", ",")
        digits = digits.replace(thousands, "").replace(decimal, ".")
        try:
            return words(float(digits) if "." in digits else int(digits))
        except Exception:  # OverflowError past 10**36: read the digits
            return m[0]

    text = _DATE.sub(standalone(date), text)
    text = _TIME.sub(standalone(time_of_day), text)
    if language == "pt":
        text = _HOURS.sub(standalone(
            lambda m: f"{hour(m[1])} hora{'s' if int(m[1]) != 1 else ''}"), text)
        # 1º de outubro, 2ª feira
        text = re.sub(r"\b(\d+)º", standalone(lambda m: words(int(m[1]), "ordinal")), text)
        text = re.sub(r"\b(\d+)ª", standalone(
            lambda m: re.sub(r"o\b", "a", words(int(m[1]), "ordinal"))), text)
    if language in _PERCENT:
        text = re.sub(r"(\d[\d.,]*)\s*%", standalone(
            lambda m: f"{m[1]} {_PERCENT[language]}"), text)
    sep = r"\," if language in _DOT_DECIMAL else r"\."
    dec = r"\." if language in _DOT_DECIMAL else r","
    return re.sub(rf"\d{{1,3}}(?:{sep}\d{{3}})+(?:{dec}\d+)?|\d+(?:{dec}\d+)?",
                  standalone(number), text)


def split_sentences(buffer: str, first=False) -> tuple[list[str], str]:
    """Complete sentences and unfinished rest; optionally shorten the first."""
    if first:
        sentence_end = _SENTENCE_END.search(buffer)
        limit = sentence_end.start() if sentence_end else len(buffer)
        cuts = []
        for boundary in re.finditer(r"[,;:](?=\s)| — ", buffer[:limit]):
            head = buffer[:boundary.end()]
            spoken = speakable(head)
            words = sum(any(c.isalnum() for c in word) for word in spoken.split())
            if words >= 4 and len(spoken) >= TTS_MIN_CHARS:
                cuts.append(boundary.end())
                break
        # Only count complete words: a delta may end halfway through one.
        words = list(re.finditer(r"\S+\s+", buffer[:limit]))
        if len(words) >= 12 and not re.search(r"[,;:—.!?…]", buffer[:words[11].end()]):
            cuts.append(words[11].end())
        if cuts:
            cut = min(cuts)
            head = buffer[:cut].rstrip()
            tail = buffer[cut:].lstrip()
            # A complete short tail already in this delta can stay with
            # the head, instead of creating a new undersized final clip.
            spoken_tail = without_emoji(speakable(tail))
            short_tail = (re.search(r"[.!?…]\s*$", spoken_tail)
                          and len(spoken_tail) < TTS_MIN_CHARS)
            if len(speakable(head)) >= TTS_MIN_CHARS and not short_tail:
                done, rest = split_sentences(tail)
                return [head, *done], rest
    parts = _SENTENCE_END.split(buffer)
    return [p for p in parts[:-1] if p.strip()], parts[-1]


def speech_markdown(buffer: str, fence: str = "") -> tuple[str, str, str]:
    """Remove fenced code before sentence splitting, even across deltas."""
    parts = []
    start = 0
    for match in re.finditer(r"```|~~~", buffer):
        if not fence:
            parts.append(buffer[start:match.start()])
            fence = match.group()
        elif fence == match.group():
            fence = ""
            parts.append("\n")
        start = match.end()
    tail = buffer[start:]
    pending = re.search(r"[`~]{1,2}$", tail)
    rest = pending.group() if pending else ""
    if not fence:
        parts.append(tail[:-len(rest)] if rest else tail)
    return "".join(parts), rest, fence


# ---------------------------------------------------------------------------
# LLM (OpenVINO GenAI, in debora_whisper/llm_server.py's process)
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
    return [python_executable(), "-m", "debora_whisper.llm_server", model_dir, device,
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


class TTSCancellation:
    """Close a turn's in-flight connection, including before headers arrive."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sockets = set()
        self.cancelled = False

    def attach(self, sock):
        with self._lock:
            if self.cancelled:
                raise TTSRejected("cancelled")
            self._sockets.add(sock)

    def detach(self, sock):
        with self._lock:
            self._sockets.discard(sock)

    def cancel(self):
        with self._lock:
            self.cancelled = True
            for sock in self._sockets:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass


def synthesize(text: str, config: dict):
    """(float32 samples, sample rate) for text from config["tts_url"]."""
    import numpy as np
    url = config["tts_url"]
    if not is_http_url(url):
        raise ValueError(f"tts_url {url!r} is not an http(s) URL")
    body = {"text": text}
    if config.get("language") and config["language"] != "auto":
        body["language"] = config["language"]
    # Sent on every request, so a voice picked in Settings speaks the next
    # sentence: "" is Chatterbox's own voice; a missing file leaves the
    # server on the voice it started with.
    voice = resolve_voice(config.get("tts_voice"))
    if voice is None:
        body["voice"] = ""
    elif voice.is_file():
        body["voice"] = str(voice.resolve())
    target = urllib.parse.urlsplit(url.rstrip("/") + "/tts")
    connection_type = (http.client.HTTPSConnection if target.scheme == "https"
                       else http.client.HTTPConnection)
    connection = connection_type(target.hostname, target.port,
                                 timeout=config["tts_timeout_seconds"])
    cancellation = config.get("_tts_cancellation")
    sock = None
    try:
        connection.connect()
        sock = connection.sock
        if cancellation:
            cancellation.attach(sock)
        connection.request("POST", urllib.parse.urlunsplit(("", "", target.path, target.query, "")),
                           body=json.dumps(body).encode("utf-8"),
                           headers={"Content-Type": "application/json"})
        with connection.getresponse() as response:
            if response.status >= 400:
                try:
                    message = json.loads(response.read(4096))["error"]
                except Exception:
                    message = response.reason
                raise TTSRejected(f"HTTP {response.status}: {message}")
            raw = response.read(TTS_MAX_RESPONSE_BYTES + 1)
    finally:
        if cancellation and sock is not None:
            cancellation.detach(sock)
        connection.close()
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
    <name>.wav in the voices folder, then among the bundled voices."""
    if not voice:
        return None
    path = Path(voice).expanduser()
    if path.suffix or len(path.parts) > 1:
        return path
    own = paths.VOICES_DIR / f"{voice}.wav"
    bundled = paths.BUNDLED_VOICES_DIR / f"{voice}.wav"
    return bundled if not own.is_file() and bundled.is_file() else own


def list_voices() -> list[str]:
    """Names of the reference voices (<name>.wav) in the voices folder and
    among the bundled voices."""
    names = set()
    for folder in (paths.VOICES_DIR, paths.BUNDLED_VOICES_DIR):
        try:
            names.update(p.stem for p in folder.glob("*.wav") if p.is_file())
        except OSError:
            pass
    return sorted(names, key=str.casefold)


def tts_command(config: dict, log=print) -> list[str] | None:
    """tts_server_command, or uv running the selected backend on tts_url's port
    with tts_voice (None if tts_url is not local or uv is missing)."""
    if config.get("tts_server_command"):
        return config["tts_server_command"]
    url = urllib.parse.urlparse(config.get("tts_url") or "")
    if url.hostname not in ("127.0.0.1", "localhost"):
        return None
    backend = config.get("tts_backend", "chatterbox")
    script = TTS_SERVER_SCRIPTS[backend]
    uv = shutil.which("uv")
    if not uv:
        log("Voice chat: uv not found on PATH; start the TTS server yourself "
            f"(uv run --script {script}).")
        return None
    command = [uv, "run", "--script", str(script), "--port", str(url.port or 80)]
    voice = resolve_voice(config.get("tts_voice"))
    if voice is not None and backend != "kokoro" and (voice.is_file() or backend in _STARTUP_VOICE_BACKENDS):
        command += ["--voice", str(voice)]
    elif voice is not None and backend == "chatterbox":
        log(f"Voice chat: voice {voice} not found; using Chatterbox's own voice.")
    if config.get("language") and config["language"] != "auto":
        command += ["--language", config["language"]]
    return command


_tts_lock = threading.RLock()
_tts_process: subprocess.Popen | None = None
_tts_server_key = None  # Process settings; also remembers settings seen with an external server.
_tts_retiring_url = None  # A managed child may outlive its uv parent briefly.
_tts_stopping = threading.Event()


class TTSStartupError(RuntimeError):
    """The server is reachable, but its backend cannot finish startup."""


def _shutdown_socket(sock):
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


# Filled by startup checks and background polling, never by telemetry itself.
_tts_health_cache = {}
_tts_health_cache_lock = threading.Lock()
_tts_health_polls = {}
_tts_health_poll_context = threading.local()


def _cache_tts_health(config, health=None):
    with _tts_health_cache_lock:
        cancelled = getattr(_tts_health_poll_context, "cancelled", None)
        if cancelled is not None and cancelled.is_set():
            return  # An in-flight check must not repopulate the cache after stop.
        url = config.get("tts_url", "").rstrip("/")
        if health is None:
            _tts_health_cache.pop(url, None)
        else:
            _tts_health_cache[url] = (time.monotonic(), health)


def _tts_loading_health(config):
    with _tts_health_cache_lock:
        cached = _tts_health_cache.get(config.get("tts_url", "").rstrip("/"))
        if cached and time.monotonic() - cached[0] <= 5 + TTS_HEALTH_SECONDS:
            return cached[1]
    return {}


def tts_download_progress(config):
    """Local snapshot only: no I/O or startup/process lock on telemetry's thread."""
    return _tts_loading_health(config).get("download")


def _start_tts_health_poll(config, process=None):
    """Keep loading telemetry fresh independently of any caller's wait deadline."""
    config = {"tts_url": config["tts_url"]}
    url = config["tts_url"].rstrip("/")
    with _tts_health_cache_lock:
        if url in _tts_health_polls:
            return
        cancelled = threading.Event()

        def poll():
            _tts_health_poll_context.cancelled = cancelled
            try:
                while not cancelled.wait(0.5):
                    if process is not None and process.poll() is not None:
                        break
                    state = _check_tts_health(config)
                    if state == "ready":
                        break
                    if process is None and state != "loading":
                        break
            except TTSStartupError:
                pass  # Individual waiters still receive the terminal error.
            finally:
                with _tts_health_cache_lock:
                    if _tts_health_polls.get(url, (None,))[0] is cancelled:
                        _tts_health_polls.pop(url, None)
                        _tts_health_cache.pop(url, None)
                del _tts_health_poll_context.cancelled

        worker = threading.Thread(target=poll, daemon=True, name="tts-health")
        _tts_health_polls[url] = (cancelled, worker)
        worker.start()


def tts_server_up(config: dict) -> bool:
    """True if /health answers 200; TTSStartupError if the server reports
    a terminal startup error (503 with {"ok": false, "error": ...})."""
    return _check_tts_health(config) == "ready"


def _check_tts_health(config: dict) -> str:
    """Return this request's state and atomically publish loading telemetry.

    Startup callers must use the returned state: another poll may replace the
    shared telemetry as soon as this request publishes it.
    """
    connection = timer = None
    try:
        target = urllib.parse.urlsplit(config["tts_url"].rstrip("/") + "/health")
        connection_type = (http.client.HTTPSConnection if target.scheme == "https"
                           else http.client.HTTPConnection)
        connection = connection_type(target.hostname, target.port, timeout=TTS_HEALTH_SECONDS)
        connection.connect()
        # The socket timeout restarts with every byte, so a peer trickling its
        # answer could hold this check (and _tts_lock) indefinitely. Closing
        # the socket at an absolute deadline ends any read still blocked.
        timer = _DeadlineTimer(TTS_HEALTH_SECONDS, _shutdown_socket, (connection.sock,))
        timer.daemon = True
        timer.start()
        connection.request("GET", urllib.parse.urlunsplit(("", "", target.path, target.query, "")))
        with connection.getresponse() as response:
            if response.status != 503:
                _cache_tts_health(config)
                return "ready" if response.status == 200 else "unavailable"
            raw = response.read(TTS_HEALTH_MAX_BYTES + 1)
        health = json.loads(raw) if len(raw) <= TTS_HEALTH_MAX_BYTES else None
    except Exception:
        _cache_tts_health(config)
        return "unavailable"
    finally:
        if timer is not None:
            timer.cancel()
        if connection is not None:
            connection.close()
    if (isinstance(health, dict) and health.get("ok") is False
            and isinstance(health.get("error"), str) and health["error"]):
        _cache_tts_health(config)
        raise TTSStartupError(health["error"])
    if isinstance(health, dict) and health.get("state") == "loading" and health.get("ok") is False:
        _cache_tts_health(config, health)
        return "loading"
    _cache_tts_health(config)
    return "unavailable"


def ensure_tts_server(config: dict, log=print, log_path=None, *, managed_only=False):
    """Start the TTS server (tts_command) if nothing answers at tts_url.
    Called by the background engine loader; reuse only matching servers.
    With managed_only, reconcile settings only if we already own a server."""
    global _tts_process, _tts_server_key, _tts_retiring_url
    if not is_http_url(config.get("tts_url")):
        return
    with _tts_lock:
        if managed_only and _tts_process is None and _tts_retiring_url is None:
            return
        backend = config.get("tts_backend", "chatterbox")
        custom_command = config.get("tts_server_command")
        # Language and Chatterbox's voice are sent with every request. File
        # additions/removals must not turn them into process restart triggers.
        voice = (resolve_voice(config.get("tts_voice"))
                 if backend in _STARTUP_VOICE_BACKENDS and not custom_command else None)
        key = (backend, tuple(custom_command) if custom_command else None,
               urllib.parse.urlparse(config["tts_url"]).port or 80,
               str(voice) if voice is not None else None)
        if _tts_process is not None and _tts_server_key != key:
            log("Voice chat: TTS settings changed; restarting the managed server.")
            # Keep the outer lock through shutdown and startup so another loader
            # cannot start a replacement while the old process still owns the port.
            _tts_retiring_url = _tts_process.tts_url
            stop_tts_server()
        if _tts_retiring_url is not None:
            deadline = time.monotonic() + 5.0
            while True:
                try:
                    if _check_tts_health({"tts_url": _tts_retiring_url}) == "unavailable":
                        _tts_retiring_url = None
                        break
                except TTSStartupError:
                    pass  # Still answering, even if unhealthy.
                if time.monotonic() >= deadline:
                    log("Voice chat: waiting for the previous TTS server to stop; will retry.")
                    return
                time.sleep(0.1)
        _tts_stopping.clear()
        if _tts_process is not None:
            if _tts_process.poll() is not None and not _tts_process.reported:
                # Starting it again would only fail again, costing the wait
                # on every reply: the replies go on as text until a restart.
                _tts_process.reported = True
                log(f"Voice chat: the TTS server exited (code {_tts_process.returncode}); "
                    f"see {log_path or 'its output'}. Restart the app after fixing it.")
            return
        try:
            state = _check_tts_health(config)
            if state in ("ready", "loading"):
                if state == "loading":
                    _start_tts_health_poll(config)
                if _tts_server_key is not None and _tts_server_key != key:
                    log(f"Voice chat: configured TTS backend/command changed to {key[0]}; "
                        "keeping the external server. Restart it manually to apply the change.")
                if _tts_server_key is None:
                    _tts_server_key = key
                return  # started by hand, or left over from an earlier run
        except TTSStartupError as exc:
            log(f"Voice chat: TTS startup failed ({exc}); restart the server after fixing it.")
            return
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
        _tts_process.tts_url = config["tts_url"]
        _tts_server_key = key
        _start_tts_health_poll(config, _tts_process)
        atexit.register(stop_tts_server)
        log(f"Voice chat: starting the TTS server (pid {_tts_process.pid}); "
            f"output in {log_path or 'nowhere'}")


def wait_tts_server(config: dict, timeout=TTS_STARTUP_SECONDS) -> bool:
    """Wait for managed startup or an explicit loading response from /health.
    Raise TTSStartupError on terminal failure."""
    deadline = time.time() + timeout
    while True:
        state = _check_tts_health(config)
        if state == "ready":
            return True
        process = _tts_process
        if ((process is None and state != "loading")
                or (process is not None and process.poll() is not None)):
            _cache_tts_health(config)
            return False
        _start_tts_health_poll(config, process)
        if time.time() > deadline:
            return False
        time.sleep(0.5)


def stop_tts_server():
    """Stop the TTS server this process started, if any."""
    global _tts_process, _tts_server_key
    with _tts_lock:
        _tts_stopping.set()
        with _tts_health_cache_lock:
            for cancelled, _ in _tts_health_polls.values():
                cancelled.set()
            _tts_health_polls.clear()
            _tts_health_cache.clear()
        process, _tts_process = _tts_process, None
        _tts_server_key = None
    if process is None:
        return
    # uv runs the server as a child process: end the whole tree, or the
    # server keeps its GPU memory after uv is gone.
    kill_tree(process)
    try:
        process.wait(5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(5)


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------
def is_voice_filler(text):
    normalized = unicodedata.normalize("NFKD", text.casefold())
    words = re.findall(r"\w+", "".join(c for c in normalized if not unicodedata.combining(c)))
    return (not words or all(word in {"hum", "ah", "a", "e", "uh", "hmm", "hm", "um", "eh"}
                             or re.fullmatch(r"h*m+", word) for word in words)
            or (len(words) == 1 and len(words[0]) == 1 and not words[0].isdigit()))


# A pause never outlasts this, even if the end of speech is never reported.
PAUSE_MAX_SECONDS = 30.0


class PlaybackStop(threading.Event):
    def __init__(self, stop):
        super().__init__()
        self.stop = stop

    def is_set(self):
        return super().is_set() or self.stop.is_set()


@dataclass
class SynthesisModel:
    """Session estimates, in seconds, learned only from uncached speech."""

    a: float = 0.95
    b: float = 1.15
    sd: float = 0.94
    k: float = 1.0
    chars_per_second: float = 14.0
    raw_chars_per_second: float = 14.0
    alpha: float = 0.2

    def predict(self, spoken):
        raw_duration = len(spoken) / self.raw_chars_per_second
        playback_duration = len(spoken) / self.chars_per_second
        return self.a + self.b * raw_duration + self.k * self.sd, playback_duration

    def update(self, spoken, synth, audio, seconds):
        if not spoken or not all(math.isfinite(t) and t > 0 for t in (synth, audio, seconds)):
            return
        # Bound outliers before an EWMA step. Normalized error updates both
        # coefficients without letting a long clip dominate the fit.
        audio = min(audio, 300.0)
        synth = min(synth, 1800.0)
        # Coverage uses trimmed playback; synthesis cost still uses raw audio.
        speed = min(40.0, max(3.0, len(spoken) / min(seconds, 300.0)))
        self.chars_per_second += self.alpha * (speed - self.chars_per_second)
        raw_speed = min(40.0, max(3.0, len(spoken) / audio))
        self.raw_chars_per_second += self.alpha * (raw_speed - self.raw_chars_per_second)
        error = synth - (self.a + self.b * audio)
        step = self.alpha * error / (1 + audio * audio)
        self.a = min(5.0, max(0.0, self.a + step))
        self.b = min(5.0, max(0.05, self.b + step * audio))


def prebuffer_wait(buffered, pending, *, held=0.0, max_wait=3.0):
    """Required hold for known (remaining synthesis, audio) pairs, capped.

    Unknown sentences add nothing, whether the writer is running or done.
    The caller subtracts elapsed synthesis from the active sentence only.
    """
    finish, play, gap = 0.0, buffered, 0.0
    for synth, audio in pending:
        finish += max(0.0, synth)
        gap = max(gap, finish - play)
        play += max(0.0, audio)
    return min(gap, max(0.0, max_wait - held))


class VoiceChat:
    """A conversation with the LLM, spoken through the TTS server.

    respond() blocks until the reply has been spoken (or interrupt() is
    called); the engine runs it on a separate, serial reply worker.
    llm(messages, on_text, stop) replaces the OpenVINO model (tests)."""

    def __init__(self, config: dict, log=print, play=None, tts_log_path=None, llm=None,
                 llm_log_path=None):
        self.config = config
        self.log = log
        self.tts_log_path = tts_log_path
        # play(samples, rate, stop) per clip; None: a StreamPlayer per reply.
        self._play = play
        self._llm_log_path = llm_log_path
        self._llm = llm or self._generate
        self._history: list[dict] = []
        self._last_turn = 0.0
        self._interrupt = threading.Event()
        self._spoken_lock = threading.Lock()
        self._spoken = []
        self._tts_cache_lock = threading.RLock()
        self._tts_cache_key = None
        self._waiting_clip = None
        self._tts_warmed = False
        self.speaking = False
        # on_audio(bool) when her voice starts or stops coming out.
        self.on_audio = None
        self._audio_shown = False
        self._playback = threading.Condition(threading.RLock())
        self._paused = False
        self._pause_deadline = 0.0
        self._pause_limit = 0.0
        self._clip_stop = None
        self._tts_cancellation = TTSCancellation()
        self._synthesis_model = SynthesisModel()

    def _show_audio(self, active: bool):
        if active == self._audio_shown:
            return
        self._audio_shown = active
        if self.on_audio:
            try:
                self.on_audio(active)
            except Exception:
                pass

    def _select_tts_cache(self, config):
        """Called under the cache lock; settings changes discard the old voice."""
        key = (config.get("tts_backend", "chatterbox"), config.get("tts_voice"),
               config.get("language"), config.get("tts_url"))
        if key != self._tts_cache_key:
            self._tts_cache_key = key
            self._waiting_clip = None
            self._tts_warmed = False

    def _waiting_audio(self, config, timing=None, started=0):
        with self._tts_cache_lock:
            self._select_tts_cache(config)
            if self._waiting_clip is None:
                if timing is not None:
                    timing["synth_start"] = time.perf_counter() - started
                try:
                    self._waiting_clip = synthesize("Um instante.", config)
                finally:
                    if timing is not None:
                        timing["synth_end"] = time.perf_counter() - started
            elif timing is not None:
                timing["cached"] = True
            return self._waiting_clip

    def warm_tts(self):
        """Warm a started server without holding up the UI or LLM loading."""
        config = dict(self.config)

        def warm():
            try:
                with self._tts_cache_lock:
                    self._select_tts_cache(config)
                    if self._tts_warmed or not wait_tts_server(config):
                        return
                    if self.speaking:
                        return  # a first turn arrived while the server loaded
                    started = time.perf_counter()
                    synthesize("Olá, estou pronta.", config)  # discarded, never played
                    if self.speaking:
                        return
                    self._waiting_audio(config)
                    self._tts_warmed = True
                    self.log(f"Voice chat: TTS warmed in {time.perf_counter() - started:.1f}s")
            except TTSStartupError as e:
                self.log(f"Voice chat: TTS startup failed ({e}); restart the server after fixing it.")
            except Exception as e:
                self.log(f"Voice chat: TTS warm-up failed ({e}); will try on the next reply.")

        worker = threading.Thread(target=warm, daemon=True)
        worker.start()
        return worker

    def reset(self, interrupt=True):
        if interrupt:
            self.interrupt()
        self._history = []
        if self.config.get("voice_chat_backend", "local") == "claude":
            from debora_whisper.harness import reset_harness
            reset_harness(self.config, self.log)

    def _generate(self, messages, on_text, stop):
        if self.config.get("voice_chat_backend", "local") == "claude":
            from debora_whisper.harness import start_harness
            start_harness(self.config, self.log).send(messages[-1]["content"], on_text, stop)
        else:
            from debora_whisper.harness import stop_harness
            stop_harness()
            generate_reply(messages, self.config, on_text, stop, self.log, self._llm_log_path)

    def interrupt(self):
        """Stop the reply in progress: no more text, synthesis or audio."""
        with self._playback:
            self._interrupt.set()
            self._paused = False
            if self._clip_stop is not None:
                self._clip_stop.set()
            self._tts_cancellation.cancel()
            self._playback.notify_all()

    def pause(self):
        with self._playback:
            if (not self.config.get("voice_chat_pause_on_speech", True)
                    or not self.config.get("voice_chat_barge_in", True)
                    or not self.speaking or self._interrupt.is_set()):
                return False
            if self._paused:
                # Talking again after a breath: no timeout while they talk.
                self._pause_deadline = None
                return False
            # Also between sentences: the next one waits instead of starting
            # over the user. The timeout counts from when they stop talking.
            self._paused = True
            self._pause_deadline = None
            self._pause_limit = time.monotonic() + PAUSE_MAX_SECONDS
            if self._clip_stop is not None:
                self._clip_stop.set()
            self._show_audio(False)
            with self._spoken_lock:
                now = time.monotonic()
                self._spoken = [(begin, min(end, now), words) for begin, end, words in self._spoken]
            self.log("Voice chat: paused reply on speech")
            self._playback.notify_all()
            return True

    def speech_ended(self):
        """Start the pause timeout once the user stops talking."""
        with self._playback:
            if self._paused:
                self._pause_deadline = (time.monotonic()
                                        + self.config.get("voice_chat_pause_timeout", 3.0))
                self._playback.notify_all()

    def resume(self, reason="invalid transcription"):
        with self._playback:
            if not self._paused:
                return False
            self._paused = False
            self.log(f"Voice chat: resumed reply ({reason})")
            self._playback.notify_all()
            return True

    def _remember_spoken(self, text, duration):
        now = time.monotonic()
        with self._spoken_lock:
            self._spoken.append((now, now + duration, text))
            self._spoken = self._spoken[-256:]

    def is_echo(self, text, started, ended):
        """Compare only audio played during capture (plus a 2 s room tail).

        Capture timestamps keep delayed ASR from matching a later reply.
        Count repeated tokens too, so a single shared word is not enough.
        """
        from collections import Counter

        def tokens(value):
            value = spoken_numbers(value, self.config.get("language"))
            value = unicodedata.normalize("NFKD", value.casefold())
            value = "".join(c for c in value if not unicodedata.combining(c))
            return Counter(re.findall(r"\w+", value))

        heard = tokens(text)
        if not heard:
            return False
        # A short answer ("sim", "quê?") right after her sentence is the user
        # answering; it is only an echo if it was heard while she spoke.
        short = sum(heard.values()) == 1
        with self._spoken_lock:
            recent = " ".join(words for begin, end, words in self._spoken
                              if ((begin <= started and ended <= end + 0.3) if short
                                  else (begin <= ended and end + 2 >= started)))
        overlap = sum((heard & tokens(recent)).values())
        # Her echo comes back as a piece of her sentence. In a few words, one
        # of the user's own ("Isso do tempo" after "...previsão do tempo?")
        # means it is them.
        count = sum(heard.values())
        return overlap == count if count <= 4 else overlap / count >= 0.6

    def _messages(self, text: str) -> list[dict]:
        if self.config.get("voice_chat_backend", "local") == "claude":
            self._history = []
            return [{"role": "user", "content": text}]
        if time.time() - self._last_turn > HISTORY_IDLE_RESET_SECONDS:
            self._history = []
        language = self.config.get("language")
        own = self.config.get("llm_prompt")
        if not own and language in VOICE_CHAT_PROMPTS:
            # Her own words already say the language; the date goes in it too.
            now = time.localtime()
            return self._with_history(
                VOICE_CHAT_PROMPTS[language]
                + f" Agora é {_PT_WEEKDAYS[now.tm_wday]}, "
                + time.strftime("%d/%m/%Y, %H:%M.", now), text)
        prompt = own or DEFAULT_VOICE_CHAT_PROMPT
        if language and language != "auto":
            # A short "Sim." alone does not tell the model the language:
            # it answered in English.
            from debora_whisper.dictation_engine import LANGUAGES
            name = LANGUAGES.get(language, language)
            prompt += f" The user speaks {name}: reply in {name}."
        # Without it, asked the time at 00:25, it said 10 in the morning.
        prompt += f" It is now {time.strftime('%A, %Y-%m-%d %H:%M')}."
        return self._with_history(prompt, text)

    def _with_history(self, prompt: str, text: str) -> list[dict]:
        recent = self._history[-2 * HISTORY_TURNS:]
        return [{"role": "system", "content": prompt}, *recent,
                {"role": "user", "content": text}]

    def respond(self, text: str, on_reply=None, stop=None) -> str:
        """Ask the LLM, speak its reply and return the text spoken so far
        ("" if the LLM failed). on_reply(text) gets the reply as it grows,
        each sentence when its audio starts (at once if it has no audio);
        on_reply(text, seconds) also says how long that audio plays."""
        started = time.perf_counter()
        turn = time.monotonic_ns()
        timings = []
        timing_lock = threading.Lock()
        first_audio = None
        pending = {}
        buffered = 0.0
        prebuffer = self.config.get("voice_chat_prebuffer", True)

        def enqueue(sentence):
            is_feedback = sentence is waiting
            shown = without_emoji(speakable("Um instante." if is_feedback else sentence))
            spoken = spoken_numbers(shown, self.config.get("language"))
            with timing_lock:
                timing = {"chunk": len(timings) + 1, "enqueued": time.perf_counter() - started,
                          "spoken": spoken, "feedback": is_feedback}
                timings.append(timing)
            with self._playback:
                if any(c.isalnum() for c in spoken):
                    pending[timing["chunk"]] = timing
                sentences.put((sentence, timing))
                self._playback.notify_all()

        def hold(timing):
            if not prebuffer or timing.get("feedback") or timing.get("cached"):
                return
            began = None
            snapshot = None
            with self._playback:
                while not stop.is_set() and not self._paused:
                    now = time.perf_counter()
                    plan = []
                    for item in pending.values():
                        synth, audio = self._synthesis_model.predict(item["spoken"])
                        if "synth_start" in item:
                            synth = max(0.0, synth - (now - started - item["synth_start"]))
                        plan.append((synth, audio))
                    gap = prebuffer_wait(buffered, plan, max_wait=math.inf)
                    elapsed = now - began if began is not None else 0.0
                    delay = min(gap, max(0.0, self.config.get("voice_chat_prebuffer_max", 3.0)
                                         - elapsed))
                    if delay <= 0:
                        break
                    if began is None:
                        began = now
                        snapshot = gap, buffered, len(plan)
                        self._show_audio(False)
                    self._playback.wait(min(0.1, delay))
            if began is not None:
                gap, ready, count = snapshot
                self.log(f"Voice chat: prebuffer held {time.perf_counter() - began:.2f}s "
                         f"(predicted gap {gap:.2f}s, buffered {ready:.2f}s, pending {count})")

        def wake_on_stop():
            # An externally supplied Event cannot notify our condition.
            stop.wait()
            with self._playback:
                self._playback.notify_all()

        def log_timing(timing):
            with timing_lock:
                if timing.get("logged"):
                    return
                timing["logged"] = True
                fields = " ".join(f"{key}={timing[key]:.3f}s" if key in timing else f"{key}=-"
                                  for key in ("enqueued", "synth_start", "synth_end", "audio", "play"))
                self.log(f"Voice chat: TTS turn={turn} chunk={timing['chunk']} {fields}"
                         + (" cached" if timing.get("cached") else ""))

        shown = []

        def show(timing, audible=False):
            # Her words enter the balloon together with her voice, and with
            # how long that voice lasts.
            if timing.get("shown") and on_reply:
                if not timing.get("displayed"):
                    shown.append(timing["shown"])
                    timing["displayed"] = True
                seconds = timing.get("seconds") if audible else None
                if seconds:
                    on_reply(" ".join(shown), seconds)
                else:
                    on_reply(" ".join(shown))

        def playback_started(timing):
            nonlocal first_audio
            with self._playback:
                if self._clip_stop.is_set():
                    return
                timing["play"] = time.perf_counter() - started
                self._show_audio(True)
                show(timing, audible=True)
                self._remember_spoken(timing["spoken"], timing["audio"])
                if first_audio is None:
                    first_audio = timing["play"]
                log_timing(timing)

        if (self.config.get("voice_chat_backend", "local") == "claude"
                and text.strip().rstrip(".!?").casefold() in ("nova conversa", "new conversation")):
            self.reset()
        # A fresh event per turn: an interrupted turn's threads keep seeing
        # theirs set, even after the next turn starts.
        stop = self._interrupt = stop if stop is not None else threading.Event()
        with self._playback:
            cancellation = self._tts_cancellation = TTSCancellation()
            self._paused = False
        # Voice chat may have been switched on in Settings since startup.
        ensure_tts_server(self.config, self.log, self.tts_log_path)
        messages = self._messages(text)
        claude = self.config.get("voice_chat_backend", "local") == "claude"
        conversation = self._history
        if not claude:
            # Keep the user even if interruption precedes the first token.
            conversation.append({"role": "user", "content": text})
            self._last_turn = time.time()
        sentences: queue.Queue = queue.Queue()
        # Unbounded: after an interrupt nobody takes clips, and the renderer
        # must still be able to finish.
        clips: queue.Queue = queue.Queue()
        # buffer: the sentence being written; short: finished sentences too
        # short to send alone.
        # said: whether a sentence was kept, so an offer of help after it can go.
        state = {"reply": "", "llm_error": None, "buffer": "", "short": "", "tokens": False,
                 "said": False, "markdown_pending": "", "fence": "", "first": True}
        start = time.time()
        waiting = object()
        feedback_lock = threading.Lock()
        finished = threading.Event()

        def feedback():
            with feedback_lock:
                if not state["tokens"] and not stop.is_set() and not finished.is_set():
                    self.log("Voice chat: waiting for Claude. Um instante.")
                    enqueue(waiting)

        timer = threading.Timer(1.5, feedback) if claude else None

        def unwanted(sentence):
            # The prompt alone did not stop them, and each one kept in the
            # history made the next reply end the same way.
            if not claude and state["said"] and is_help_offer(sentence):
                self.log(f"Voice chat: dropped the offer of help {sentence!r}")
                return True
            state["said"] = True
            return False

        def on_text(chunk):
            if stop.is_set():
                return
            with feedback_lock:
                if not state["tokens"]:
                    state["tokens"] = True
                    self.log(f"Voice chat: first token after {time.time() - start:.1f}s")
            if claude:
                chunk, state["markdown_pending"], state["fence"] = speech_markdown(
                    state["markdown_pending"] + chunk, state["fence"])
            done, state["buffer"] = split_sentences(state["buffer"] + chunk, first=state["first"])
            for sentence in done:
                if unwanted(sentence):
                    continue
                sentence = f"{state['short']} {sentence}".strip()
                if len(speakable(sentence)) < TTS_MIN_CHARS:
                    state["short"] = sentence
                else:
                    state["short"] = ""
                    state["first"] = False
                    enqueue(sentence)

        def write():
            try:
                self._llm(messages, on_text, stop)
                if state["buffer"].strip() and unwanted(state["buffer"].strip()):
                    state["buffer"] = ""
                rest = f"{state['short']} {state['buffer']}".strip()
                if rest and not stop.is_set():
                    enqueue(rest)
            except Exception as e:
                state["llm_error"] = e
                self.log(f"Voice chat: LLM failed ({e})")
                if claude and not stop.is_set():
                    enqueue("Não consegui falar com o Claude. Confira a pasta e o log da Débora.")
            finally:
                with feedback_lock:
                    finished.set()
                    if timer:
                        timer.cancel()
                    sentences.put(None)
                with self._playback:
                    self._playback.notify_all()

        def render():
            nonlocal buffered
            tts_ok = None  # unknown until the first sentence
            try:
                while not stop.is_set():
                    item = sentences.get()
                    if item is None:
                        return
                    sentence, timing = item
                    if stop.is_set():
                        return
                    is_feedback = sentence is waiting
                    if is_feedback:
                        sentence = "Um instante."
                    # Her text is for speech: emoji the prompt did not stop
                    # are neither said nor shown.
                    spoken = without_emoji(speakable(sentence))
                    if not spoken:
                        continue
                    if not is_feedback:
                        state["reply"] = f"{state['reply']} {spoken}".strip()
                        timing["shown"] = spoken
                    self.log(f"Voice chat: reply {spoken!r}")
                    spoken = timing["spoken"]
                    if not any(c.isalnum() for c in spoken):
                        continue  # an emoji or punctuation: nothing to say
                    audio = None
                    if tts_ok is None:
                        try:
                            tts_ok = wait_tts_server(self.config)
                        except TTSStartupError as e:
                            tts_ok = False
                            self.log(f"Voice chat: TTS startup failed ({e}); showing the reply only.")
                        else:
                            if not tts_ok:
                                self.log("Voice chat: TTS server not reachable; showing the reply only.")
                    if tts_ok:
                        try:
                            config = dict(self.config)
                            config["_tts_cancellation"] = cancellation
                            if is_feedback:
                                audio = self._waiting_audio(config, timing, started)
                            else:
                                with self._playback:
                                    timing["synth_start"] = time.perf_counter() - started
                                    self._playback.notify_all()
                                try:
                                    audio = synthesize(spoken, config)
                                finally:
                                    timing["synth_end"] = time.perf_counter() - started
                        except TTSRejected as e:
                            self.log(f"Voice chat: TTS skipped {spoken[:40]!r} ({e})")
                        except Exception as e:
                            # Keep showing the text even without a voice.
                            tts_ok = False
                            if not stop.is_set() and not _tts_stopping.is_set():
                                self.log(f"Voice chat: TTS failed ({e}); showing the reply only.")
                    if stop.is_set():
                        return
                    if audio is not None:
                        timing["audio"] = len(audio[0]) / audio[1]
                        timing["spoken"] = spoken
                        samples = trim_silence(*audio)
                        if len(samples):
                            # What is heard; the balloon slides at this pace.
                            timing["seconds"] = len(samples) / audio[1]
                        with self._playback:
                            if not is_feedback and not timing.get("cached"):
                                self._synthesis_model.update(
                                    spoken, timing["synth_end"] - timing["synth_start"],
                                    timing["audio"], timing.get("seconds", 0.0))
                            buffered += timing.get("seconds", 0.0)
                            pending.pop(timing["chunk"], None)
                            clips.put((samples, audio[1], timing))
                            self._playback.notify_all()
                    else:
                        log_timing(timing)
                        with self._playback:
                            pending.pop(timing["chunk"], None)
                            clips.put(((), None, timing))  # text only, in order
                            self._playback.notify_all()
            finally:
                with self._playback:
                    pending.clear()
                    clips.put(None)
                    self._playback.notify_all()

        writer = threading.Thread(target=write, daemon=True)
        renderer = threading.Thread(target=render, daemon=True)
        if prebuffer:
            threading.Thread(target=wake_on_stop, daemon=True).start()
        if timer:
            timer.daemon = True
            timer.start()
        writer.start()
        renderer.start()
        self.speaking = True
        play = self._play or StreamPlayer()
        idle = True
        try:
            while not stop.is_set():
                try:
                    try:
                        clip = clips.get_nowait()
                    except queue.Empty:
                        idle = True
                        clip = clips.get(timeout=0.1)
                except queue.Empty:
                    # The last clip has played out; the next is still being
                    # synthesized.
                    self._show_audio(False)
                    continue
                if clip is None or stop.is_set():
                    break
                if not len(clip[0]):
                    show(clip[2])  # nothing to hear (no TTS, or only silence)
                else:
                    if idle:
                        hold(clip[2])
                    with self._playback:
                        buffered = max(0.0, buffered - clip[2].get("seconds", 0.0))
                    idle = False
                    while not stop.is_set():
                        with self._playback:
                            while self._paused and not stop.is_set():
                                deadline = min(self._pause_limit, self._pause_deadline
                                               or self._pause_limit)
                                remaining = deadline - time.monotonic()
                                if remaining <= 0:
                                    self.resume("timeout")
                                else:
                                    self._playback.wait(remaining)
                            clip_stop = self._clip_stop = PlaybackStop(stop)
                        if stop.is_set():
                            break
                        if self._play is None:
                            play(clip[0], clip[1], clip_stop,
                                 on_start=lambda timing=clip[2]: playback_started(timing),
                                 wait_for_tail=(self.config.get("voice_chat_pause_on_speech", True)
                                                and self.config.get("voice_chat_barge_in", True)))
                        else:
                            playback_started(clip[2])
                            play(clip[0], clip[1], clip_stop)
                        with self._playback:
                            cut = clip_stop.is_set()
                            self._clip_stop = None
                        if cut and hasattr(play, "close"):
                            play.close(interrupted=True)
                        if not cut or stop.is_set():
                            break
        finally:
            try:
                if hasattr(play, "close"):
                    play.close(interrupted=stop.is_set())  # waits for the last clip
            finally:
                self._show_audio(False)
                self.speaking = False
                stop.set()  # stops the writer and renderer threads
                with self._playback:
                    self._paused = False
                    self._clip_stop = None
                    cancellation.cancel()
                    self._playback.notify_all()
                # Drain Claude's result / local cancellation before the next
                # turn. In particular, never clear its stop event early.
                writer.join()
                with self._spoken_lock:
                    now = time.monotonic()
                    self._spoken = [(begin, min(end, now), words)
                                    for begin, end, words in self._spoken]
                for timing in list(timings):
                    log_timing(timing)
                elapsed = f"{first_audio:.3f}s" if first_audio is not None else "-"
                self.log(f"Voice chat: TTS turn={turn} first_audio={elapsed}")

        if state["llm_error"] is not None and not state["reply"]:
            self.log(f"Voice chat: LLM failed ({state['llm_error']})")
            return ""
        reply = state["reply"]
        if claude:
            self.log(f"Voice chat: replied in {time.time() - start:.1f}s")
            return reply
        if reply and any(turn == {"role": "assistant", "content": reply}
                         for turn in self._history):
            # A copy of an earlier reply. Kept in the history, it made the
            # next replies copies too, whatever the user said.
            self.log("Voice chat: the LLM repeated an earlier reply; "
                     "not keeping it in the conversation.")
            self._last_turn = time.time()
        elif reply and self._history is conversation:
            conversation.append({"role": "assistant", "content": reply})
            self._last_turn = time.time()
            self.log(f"Voice chat: replied in {time.time() - start:.1f}s")
        return reply


# Pause between two sentences, after their own silence is trimmed.
SENTENCE_GAP_SECONDS = 0.15
_SILENCE_LEVEL = 10 ** (-45 / 20)  # -45 dBFS


def trim_silence(samples, sample_rate: int, keep_seconds=0.05):
    """samples without the silence the TTS leaves before and after speech."""
    import numpy as np
    loud = np.flatnonzero(np.abs(samples) > _SILENCE_LEVEL)
    if len(loud) == 0:
        return samples[:0]
    keep = int(sample_rate * keep_seconds)
    samples = samples[max(loud[0] - keep, 0):loud[-1] + keep + 1]
    return _without_trailing_hiss(samples, sample_rate, keep)


# Chatterbox sometimes adds a faint hiss after a pause once the speech is
# over ("Prazer, Alexandre.": 1 s of silence, then 0.4 s of hiss at -42 dB).
_FRAME_SECONDS = 0.02
_TAIL_GAP_SECONDS = 0.3
_GAP_LEVEL_DB = -50
_SPEECH_LEVEL_DB = -30


def _without_trailing_hiss(samples, sample_rate: int, keep: int):
    """samples cut at the first long pause after which nothing reaches
    speech level, with a short fade so the cut does not click."""
    import numpy as np
    n = max(int(sample_rate * _FRAME_SECONDS), 1)
    count = len(samples) // n
    if count == 0:
        return samples
    frames = samples[:count * n].reshape(count, n).astype(np.float64)
    db = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-12)
    gap = max(int(_TAIL_GAP_SECONDS / _FRAME_SECONDS), 1)
    quiet = 0
    for i, level in enumerate(db):
        quiet = quiet + 1 if level < _GAP_LEVEL_DB else 0
        if quiet >= gap and db[i + 1:].max(initial=-np.inf) < _SPEECH_LEVEL_DB:
            samples = samples[:(i - quiet + 1) * n + keep].copy()
            fade = min(len(samples), keep)
            samples[len(samples) - fade:] *= np.linspace(1, 0, fade, dtype=samples.dtype)
            return samples
    return samples


class StreamPlayer:
    """Plays a reply's clips back to back on one output stream.

    sd.play() opened a stream per clip: the next one started while the last
    one's buffered tail was still playing, and the two overlapped."""

    BLOCK_SECONDS = 0.05  # how soon an interrupt silences it

    def __init__(self):
        self._stream = None
        self._rate = None

    def __call__(self, samples, sample_rate: int, stop: threading.Event, on_start=None,
                 wait_for_tail=False):
        import numpy as np
        import sounddevice as sd
        gap = 0
        if self._stream is not None and self._rate != sample_rate:
            self.close(interrupted=False)
        if self._stream is None:
            self._stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32")
            self._stream.start()
            self._rate = sample_rate
        else:
            gap = int(sample_rate * SENTENCE_GAP_SECONDS)
            samples = np.concatenate([np.zeros(gap, dtype=np.float32), samples])
        data = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1, 1)
        block = max(int(sample_rate * self.BLOCK_SECONDS), 1)
        for start in range(0, len(data), block):
            if stop.is_set():
                return
            if on_start is not None and start + block > gap:
                # First speech block submitted to the output device. Its
                # hardware buffer adds latency we cannot measure here.
                on_start()
                on_start = None
            # Blocks while the stream's buffer is full, so this keeps pace
            # with playback.
            self._stream.write(data[start:start + block])
        if wait_for_tail and not stop.is_set():
            # Keep the cut sentence available while its buffered tail plays,
            # including short clips that fit entirely in the output buffer.
            stop.wait(self._stream.latency)

    def close(self, interrupted: bool):
        """Let the queued audio finish (or drop it if interrupted)."""
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.abort() if interrupted else stream.stop()
        finally:
            stream.close()
