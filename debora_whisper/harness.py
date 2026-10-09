"""One persistent Claude Code conversation over newline-delimited JSON."""
import atexit
import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from debora_whisper import paths
from debora_whisper.processes import NO_WINDOW, kill_tree

HARNESS_PROMPT = Path(__file__).with_name("harness_prompt.md")
PERMISSION_MODES = ("acceptEdits", "auto", "bypassPermissions", "manual", "dontAsk", "plan")
MEMORY_MAX_LINES = 100
MEMORY_MAX_BYTES = 8192
# Exact forms deliberately avoid writable flags (git branch -D, --output,
# nvidia-smi -pl) and indefinite reads (Get-Content -Wait).
DEFAULT_ALLOWED_TOOLS = tuple(
    f"{tool}({command})"
    for tool in ("Bash", "PowerShell")
    for command in (
        "git status", "git status --short", "git status --porcelain",
        "git log", "git log --oneline", "git log -5 --oneline",
        "git diff", "git diff --stat", "git diff --cached", "git show",
        "git branch", "git branch --all", "git branch --show-current",
        "docker ps", "docker ps --all", "docker ps -a", "docker info",
        "docker images", "docker version", "docker compose ps",
        "docker compose ps --all", "wsl -l -v", "wsl --list",
        "wsl --list --verbose", "zellij list-sessions", "nvidia-smi",
        "Get-ChildItem", "Get-ChildItem -Force", "Get-ChildItem -Path .",
        "Get-Content README.md", "Get-Content -Path README.md",
        "Get-Process", "Get-Service", "Test-Path .", "Test-Path -Path .",
    )
)


def harness_allowed_tools(config: dict) -> tuple[str, ...]:
    rules = config.get("harness_allowed_tools")
    return DEFAULT_ALLOWED_TOOLS if rules is None else tuple(rules)


def harness_command(config: dict, session_id: str, resume: bool) -> list[str]:
    """Build argv; executable lookup and the dated prompt happen at launch."""
    command = ["claude", "-p", "--input-format", "stream-json",
               "--output-format", "stream-json", "--verbose", "--include-partial-messages",
               "--resume" if resume else "--session-id", session_id,
               "--permission-mode", config.get("harness_permission_mode", "acceptEdits"),
               "--permission-prompts", "host", "--permission-prompt-tool", "stdio",
               "--add-dir", str(harness_memory_file(config).parent),
               "--append-system-prompt-file", str(config.get("harness_prompt_file") or HARNESS_PROMPT),
               "--system-prompt-snapshot", "off"]
    config_dir = paths.CONFIG_DIR.resolve()
    if not config_dir.is_relative_to(harness_cwd(config)):
        command += ["--add-dir", str(config_dir)]
    if config.get("harness_model"):
        command += ["--model", config["harness_model"]]
    rules = harness_allowed_tools(config)
    if rules:
        command += ["--allowedTools", *rules]
    return command


def harness_cwd(config: dict) -> Path:
    return Path(config.get("harness_cwd") or Path.home()).expanduser().resolve()


def harness_memory_file(config: dict) -> Path:
    path = Path(config.get("harness_memory_file") or
                paths.CONFIG_DIR / "harness" / "voice_memory.md").expanduser()
    if not path.is_absolute():
        path = harness_cwd(config) / path
    return path.resolve()


def harness_key(config: dict) -> tuple:
    return (os.path.normcase(str(harness_cwd(config))), config.get("harness_model"),
            config.get("harness_permission_mode", "acceptEdits"),
            config.get("harness_prompt_file"), config.get("language"),
            config.get("harness_permission_response", "deny"),
            os.path.normcase(str(harness_memory_file(config))), harness_allowed_tools(config))


def voice_prompt(prompt: str, language: str | None, now: str) -> str:
    return (prompt.rstrip() + f"\nIdioma configurado: {language or 'auto'} "
            "(auto: acompanhe o idioma do usuário)."
            + f"\nData e hora local ao iniciar este processo: {now}.\n")


def _self_prompt(config: dict) -> str:
    logs = paths.LOG_DIR.resolve()
    return ("\n## Sobre a própria Débora\n"
            "Você pode ler e editar estes arquivos para diagnosticar e corrigir a Débora:\n"
            f"Configuração: {paths.CONFIG_FILE.resolve()}\n"
            f"Inicialização, tempos de transcrição, hardware e turnos de voz: {logs / 'app.log'}\n"
            f"CPU/RAM/VRAM e buffer de áudio: {logs / 'telemetry.log'}\n"
            f"Servidor de voz: {logs / 'tts_server.log'}\n"
            f"Memória de voz: {harness_memory_file(config)}\n")


def memory_terms(text: str) -> list[str]:
    """Extract unique correction terms, in order, without optional context."""
    terms = []
    for line in text.splitlines():
        match = re.fullmatch(r'- "(?:[^"\\]|\\.)+" → (.+?)(?: \([^()]*\))?', line)
        if match:
            term = match[1].strip()
            if term and term not in terms:
                terms.append(term)
    return terms


class MemoryHotwords:
    """Cache a small, newest-first recognition vocabulary by file mtime."""

    def __init__(self):
        self._key = None
        self._terms = ()

    def terms(self, config: dict, log=print) -> tuple[str, ...]:
        path = harness_memory_file(config)
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            mtime = None
        key = (path, mtime)
        if key == self._key:
            return self._terms
        self._key, self._terms = key, ()
        if mtime is None:
            return self._terms
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as e:
            log(f"Voice chat: cannot read hotword memory ({type(e).__name__})")
            return self._terms
        # Reverse lines before extraction so repeated terms retain their latest entry.
        terms, seen, tokens = [], set(), 0
        for term in memory_terms("\n".join(reversed(text.splitlines()))):
            folded = term.casefold()
            if folded in seen:
                continue
            seen.add(folded)
            # UTF-8 bytes / 3 plus a separator is a cheap, conservative estimate
            # for project names; leave room below Whisper's ~224-token budget.
            cost = (len(term.encode("utf-8")) + 2) // 3 + 1
            if tokens + cost > 150:
                continue
            terms.append(term)
            tokens += cost
            if len(terms) == 40:
                break
        self._terms = tuple(terms)
        return self._terms


def _memory_prompt(path: Path, log) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as file:
            file.write("# Memória de reconhecimento de voz\n")
    except FileExistsError:
        pass
    lines = path.read_text(encoding="utf-8").splitlines()
    kept, size = [], 0
    for line in reversed(lines[-MEMORY_MAX_LINES:]):
        length = len((line + "\n").encode("utf-8"))
        if size + length > MEMORY_MAX_BYTES:
            break
        kept.append(line)
        size += length
    if len(kept) < len(lines):
        log(f"Voice chat: memory truncated for prompt: {path} "
            f"({len(kept)}/{len(lines)} lines, {size} bytes)")
    content = "\n".join(reversed(kept))
    return (f"\n## Erros de reconhecimento já conhecidos\n"
            f"Arquivo de memória de voz: {path}\n\n{content}\n")


def permission_response(request_id: str, request: dict, allow: bool) -> dict:
    response = ({"behavior": "allow", "updatedInput": request["input"]} if allow else
                {"behavior": "deny", "message": "Permissão negada pela Débora."})
    return {"type": "control_response", "response": {
        "subtype": "success", "request_id": request_id, "response": response}}


def _sessions(path: Path, log) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        log(f"Voice chat: cannot read harness sessions ({e})")
        return {}


_sessions_lock = threading.Lock()


def _save_session(path: Path, cwd: str, session_id: str | None, log):
    with _sessions_lock:
        data = _sessions(path, log)
        if session_id is None:
            data.pop(cwd, None)
        else:
            data[cwd] = session_id
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(".tmp")
        pending.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        pending.replace(path)


class HarnessSession:
    """Claude owns history; only the latest transcript goes to send().

    on_event(kind, message) reports tool_use and permission events.
    permission_handler(request) can replace the configured decision later.
    """

    def __init__(self, config: dict, log=print, command_factory=harness_command,
                 process_factory=subprocess.Popen, session_file=None, on_event=None,
                 permission_handler=None):
        self.config, self.log = dict(config), log
        self.key = harness_key(config)
        self.cwd = harness_cwd(config)
        if not self.cwd.is_dir():
            log(f"Voice chat: harness folder does not exist: {self.cwd}")
            raise ValueError(f"Harness folder does not exist: {self.cwd}")
        self.on_event, self.permission_handler = on_event, permission_handler
        self.session_file = Path(session_file) if session_file else paths.CONFIG_DIR / "harness_session.json"
        saved = _sessions(self.session_file, log).get(self.key[0])
        self.session_id = saved or str(uuid.uuid4())
        self.error: str | None = None
        self._events: queue.Queue = queue.Queue()
        self._send_lock = threading.Lock()
        self._turn_lock = threading.Lock()
        self._prompt_file: Path | None = None
        try:
            source = Path(config.get("harness_prompt_file") or HARNESS_PROMPT).expanduser()
            if not source.is_absolute():
                source = self.cwd / source
            prompt = voice_prompt(source.read_text(encoding="utf-8"), config.get("language"),
                                  time.strftime("%A, %Y-%m-%d %H:%M"))
            prompt += _self_prompt(config)
            prompt += _memory_prompt(harness_memory_file(config), log)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".md",
                                             prefix="debora-voice-", delete=False) as file:
                self._prompt_file = Path(file.name)
                file.write(prompt)
            command = command_factory({**config, "harness_prompt_file": str(self._prompt_file)},
                                      self.session_id, bool(saved))
            command[0] = shutil.which(command[0]) or command[0]
            self.process = process_factory(
                command, cwd=str(self.cwd), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=NO_WINDOW)
        except Exception:
            self._remove_prompt()
            raise
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        log(f"Voice chat: Claude session {self.session_id} in {self.cwd} "
            f"({'resume' if saved else 'new'}, pid {self.process.pid})")

    @property
    def running(self) -> bool:
        return self.process.poll() is None and self.error is None

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    message = json.loads(line)
                except ValueError:
                    self.log(f"Voice chat: Claude stdout: {line.strip()}")
                    continue
                if isinstance(message, dict):
                    self._events.put(message)
        finally:
            self.error = self.error or "Claude closed its output"
            self._events.put(None)

    def _read_stderr(self):
        for line in self.process.stderr:
            if line.strip():
                self.log(f"Voice chat: Claude stderr: {line.strip()}")

    def _send(self, message: dict):
        try:
            with self._send_lock:
                self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
                self.process.stdin.flush()
        except (OSError, ValueError) as e:
            raise RuntimeError(self.error or f"Claude input closed: {e}") from e

    def _notice(self, kind: str, message: str, on_event):
        self.log(f"Voice chat: {message}")
        callback = on_event or self.on_event
        if callback:
            callback(kind, message)

    def _permission_denied(self, denial: dict, denials_seen: set, on_event):
        tool_id = denial.get("tool_use_id")
        if tool_id and tool_id in denials_seen:
            return
        if tool_id:
            denials_seen.add(tool_id)
        tool_input = denial.get("tool_input", denial.get("input", {}))
        detail = tool_input.get("command", tool_input) if isinstance(tool_input, dict) else tool_input
        detail = f"{denial.get('tool_name', 'unknown')} {detail}"
        if len(detail) > 200:
            detail = detail[:197] + "..."
        self._notice("permission", f"Claude permission denied: {detail}", on_event)

    def interrupt(self):
        self._send({"type": "control_request", "request_id": str(uuid.uuid4()),
                    "request": {"subtype": "interrupt"}})

    def send(self, text: str, on_text, stop: threading.Event, on_event=None):
        """Stream text and drain an interrupted turn before accepting another."""
        with self._turn_lock:
            if stop.is_set():
                return
            if not self.running:
                raise RuntimeError(self.error or "Claude is not running")
            self._send({"type": "user", "message": {"role": "user", "content": text}})
            interrupted = None
            last_event = time.monotonic()
            tools_seen = set()
            denials_seen = set()
            try:
                while True:
                    now = time.monotonic()
                    if stop.is_set() and interrupted is None:
                        self.interrupt()
                        interrupted = now
                        self.log("Voice chat: interrupting Claude; draining the turn")
                    if ((interrupted is not None and now - interrupted > 15)
                            or now - last_event > 180):
                        raise TimeoutError("Claude did not finish the turn; see app.log")
                    try:
                        message = self._events.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if message is None:
                        raise RuntimeError(self.error)
                    last_event = now
                    kind = message.get("type")
                    if kind == "control_request":
                        request = message["request"]
                        if request.get("subtype") == "can_use_tool":
                            self._notice("permission", f"Claude pede permissão: {request['tool_name']} "
                                         f"{request.get('input', {})}", on_event)
                            allow = self.config.get("harness_permission_response", "deny") == "allow"
                            if self.permission_handler and not stop.is_set():
                                allow = bool(self.permission_handler(request))
                            allow = allow and not stop.is_set()
                            self._send(permission_response(message["request_id"], request, allow))
                            if allow:
                                self.log(f"Voice chat: Claude permission allowed: {request['tool_name']}")
                            else:
                                self._permission_denied(request, denials_seen, on_event)
                        else:
                            self._send({"type": "control_response", "response": {
                                "subtype": "error", "request_id": message["request_id"],
                                "error": "Unsupported control request"}})
                    elif kind == "stream_event" and not message.get("parent_tool_use_id"):
                        event = message.get("event", {})
                        block = event.get("content_block", {})
                        if block.get("type") == "tool_use":
                            tools_seen.add(block.get("id"))
                            self._notice("tool_use", f"Claude está usando {block['name']}…", on_event)
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta" and not stop.is_set():
                            on_text(delta["text"])
                    elif kind == "assistant":
                        # The full message repeats deltas, but may reveal a tool without partials.
                        for block in message.get("message", {}).get("content", []):
                            if block.get("type") == "tool_use" and block.get("id") not in tools_seen:
                                tools_seen.add(block.get("id"))
                                self._notice("tool_use", f"Claude está usando {block['name']}…", on_event)
                    elif kind == "result":
                        self.session_id = message.get("session_id", self.session_id)
                        try:
                            _save_session(self.session_file, self.key[0], self.session_id, self.log)
                        except OSError as e:
                            self.log(f"Voice chat: cannot save Claude session {self.session_id} ({e})")
                        for denial in message.get("permission_denials", []):
                            self._permission_denied(denial, denials_seen, on_event)
                        if message.get("is_error") and not stop.is_set():
                            raise RuntimeError("; ".join(message.get("errors", [])) or "Claude turn failed")
                        return
            except Exception as e:
                self.error = str(e)
                self.log(f"Voice chat: Claude failed ({e})")
                self.stop()
                raise

    def _remove_prompt(self):
        if self._prompt_file:
            self._prompt_file.unlink(missing_ok=True)
            self._prompt_file = None

    def stop(self):
        try:
            self.process.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            self.process.wait(5)
        except subprocess.TimeoutExpired:
            kill_tree(self.process)
            self.process.wait(5)
        finally:
            self._remove_prompt()


_harness_lock = threading.Lock()
_harness: HarnessSession | None = None


def start_harness(config: dict, log=print) -> HarnessSession:
    global _harness
    key = harness_key(config)
    with _harness_lock:
        if _harness is not None and _harness.key == key and _harness.running:
            return _harness
        if _harness is not None:
            _harness.stop()
            _harness = None
        try:
            _harness = HarnessSession(config, log)
        except Exception as e:
            log(f"Voice chat: cannot start Claude ({e})")
            raise
        return _harness


def stop_harness():
    global _harness
    with _harness_lock:
        harness, _harness = _harness, None
        if harness is not None:
            harness.stop()


def reset_harness(config: dict, log=print):
    """Forget this folder's conversation; the next turn gets a fresh UUID."""
    global _harness
    key = harness_key(config)
    with _harness_lock:
        if _harness is not None and _harness.key[0] == key[0]:
            _harness.stop()
            # An interrupted writer may still be saving its last result.
            with _harness._turn_lock:
                pass
            _harness = None
        _save_session(paths.CONFIG_DIR / "harness_session.json", key[0], None, log)
    log(f"Voice chat: new Claude conversation requested in {key[0]}")


atexit.register(stop_harness)
