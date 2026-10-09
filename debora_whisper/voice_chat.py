"""Voice chat: each final transcription goes to a local LLM (OpenVINO GenAI,
in debora_whisper/llm_server.py's process) and its reply is spoken aloud by the Chatterbox TTS server
(debora_whisper/tts_server.py, in its own uv environment).

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
TTS_SERVER_SCRIPT = Path(__file__).with_name("tts_server.py")
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

    text = _DATE.sub(date, text)
    text = _TIME.sub(time_of_day, text)
    if language == "pt":
        text = _HOURS.sub(lambda m: f"{hour(m[1])} hora{'s' if int(m[1]) != 1 else ''}", text)
        # 1º de outubro, 2ª feira
        text = re.sub(r"\b(\d+)º", lambda m: words(int(m[1]), "ordinal"), text)
        text = re.sub(r"\b(\d+)ª", lambda m: re.sub(r"o\b", "a", words(int(m[1]), "ordinal")), text)
    if language in _PERCENT:
        text = re.sub(r"(\d)\s*%", rf"\1 {_PERCENT[language]}", text)
    sep = r"\," if language in _DOT_DECIMAL else r"\."
    dec = r"\." if language in _DOT_DECIMAL else r","
    return re.sub(rf"\d{{1,3}}(?:{sep}\d{{3}})+(?:{dec}\d+)?|\d+(?:{dec}\d+)?",
                  number, text)


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Complete sentences in buffer, and the unfinished rest."""
    parts = _SENTENCE_END.split(buffer)
    return [p for p in parts[:-1] if p.strip()], parts[-1]


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
        # play(samples, rate, stop) per clip; None: a StreamPlayer per reply.
        self._play = play
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
        # said: whether a sentence was kept, so an offer of help after it can go.
        state = {"reply": "", "llm_error": None, "buffer": "", "short": "", "tokens": False,
                 "said": False}
        start = time.time()

        def unwanted(sentence):
            # The prompt alone did not stop them, and each one kept in the
            # history made the next reply end the same way.
            if state["said"] and is_help_offer(sentence):
                self.log(f"Voice chat: dropped the offer of help {sentence!r}")
                return True
            state["said"] = True
            return False

        def on_text(chunk):
            if not state["tokens"]:
                state["tokens"] = True
                self.log(f"Voice chat: first token after {time.time() - start:.1f}s")
            done, state["buffer"] = split_sentences(state["buffer"] + chunk)
            for sentence in done:
                if unwanted(sentence):
                    continue
                sentence = f"{state['short']} {sentence}".strip()
                if len(speakable(sentence)) < TTS_MIN_CHARS:
                    state["short"] = sentence
                else:
                    state["short"] = ""
                    sentences.put(sentence)

        def write():
            try:
                self._llm(messages, on_text, stop)
                if state["buffer"].strip() and unwanted(state["buffer"].strip()):
                    state["buffer"] = ""
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
                    # Her text is for speech: emoji the prompt did not stop
                    # are neither said nor shown.
                    spoken = without_emoji(speakable(sentence))
                    if not spoken:
                        continue
                    state["reply"] = f"{state['reply']} {spoken}".strip()
                    if on_reply:
                        on_reply(state["reply"])
                    self.log(f"Voice chat: reply {spoken!r}")
                    spoken = spoken_numbers(spoken, self.config.get("language"))
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
                        clips.put((trim_silence(*audio), audio[1]))
            finally:
                clips.put(None)

        writer = threading.Thread(target=write, daemon=True)
        renderer = threading.Thread(target=render, daemon=True)
        writer.start()
        renderer.start()
        self.speaking = True
        play = self._play or StreamPlayer()
        try:
            while True:
                clip = clips.get()
                if clip is None or stop.is_set():
                    break
                play(clip[0], clip[1], stop)
        finally:
            try:
                if hasattr(play, "close"):
                    play.close(interrupted=stop.is_set())  # waits for the last clip
            finally:
                self.speaking = False
                stop.set()  # stops the writer and renderer threads

        if state["llm_error"] is not None and not state["reply"]:
            self.log(f"Voice chat: LLM failed ({state['llm_error']})")
            return ""
        reply = state["reply"]
        if reply and any(turn == {"role": "assistant", "content": reply}
                         for turn in self._history):
            # A copy of an earlier reply. Kept in the history, it made the
            # next replies copies too, whatever the user said.
            self.log("Voice chat: the LLM repeated an earlier reply; "
                     "not keeping it in the conversation.")
            self._last_turn = time.time()
        elif reply:
            self._history += [{"role": "user", "content": text},
                              {"role": "assistant", "content": reply}]
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

    def __call__(self, samples, sample_rate: int, stop: threading.Event):
        import numpy as np
        import sounddevice as sd
        if self._stream is not None and self._rate != sample_rate:
            self.close(interrupted=False)
        if self._stream is None:
            self._stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32")
            self._stream.start()
            self._rate = sample_rate
        else:
            samples = np.concatenate([np.zeros(int(sample_rate * SENTENCE_GAP_SECONDS),
                                               dtype=np.float32), samples])
        data = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1, 1)
        block = max(int(sample_rate * self.BLOCK_SECONDS), 1)
        for start in range(0, len(data), block):
            if stop.is_set():
                return
            # Blocks while the stream's buffer is full, so this keeps pace
            # with playback.
            self._stream.write(data[start:start + block])

    def close(self, interrupted: bool):
        """Let the queued audio finish (or drop it if interrupted)."""
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.abort() if interrupted else stream.stop()
        finally:
            stream.close()
