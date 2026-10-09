# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "chatterbox-tts==0.1.7", "torch==2.6.0", "torchaudio==2.6.0",
#     # resemble-perth (Chatterbox's watermark) imports pkg_resources,
#     # which setuptools 81 dropped.
#     "setuptools<81",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cu124"
# url = "https://download.pytorch.org/whl/cu124"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu124" }
# torchaudio = { index = "pytorch-cu124" }
# ///
"""Local TTS server for debora-whisper's voice chat, backed by Chatterbox
Multilingual on an NVIDIA GPU.

It never runs in the app's environment: chatterbox-tts pins torch 2.6 (CUDA)
and numpy 1.26. The block above lets uv build its own environment on the first
run (a few GB, cached afterwards):

    uv run --script debora_whisper/tts_server.py --voice ref.wav

The app starts it this way by itself in voice chat mode, with the voice
from "tts_voice" in config.json, and stops it on exit.

Endpoints (127.0.0.1 only):
    GET  /health -> {"ok": true, "sample_rate": ...}
    POST /tts    {"text": "...", "language": "pt", "voice": "C:/voices/ana.wav"}
                 -> audio/wav (16-bit mono). "voice" is optional: "" is
                 Chatterbox's own voice, no "voice" the one from --voice.

Log lines look like app.log's: "[YYYY-MM-DD HH:MM:SS] message".
"""
import argparse
import io
import json
import logging
import os
import sys
import threading
import time
import types
import warnings
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_TEXT_CHARS = 1000
MAX_BODY_BYTES = 64_000
VOICE_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg")
# Prepared voices kept in memory (a few MB each on the GPU).
MAX_CACHED_VOICES = 8

log = logging.getLogger("tts_server").info


def setup_logging():
    """app.log's line format, without the libraries' noise: deprecation
    warnings, the HF token hint, per-sentence progress bars and Chatterbox's
    "forcing EOS token" notes."""
    # Libraries' INFO lines (each HTTP request to the hub, frame rates) stay
    # out; their warnings and errors stay in.
    logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.WARNING, stream=sys.stdout, force=True)
    logging.getLogger("tts_server").setLevel(logging.INFO)
    for name in ("huggingface_hub", "transformers", "diffusers", "chatterbox"):
        logging.getLogger(name).setLevel(logging.ERROR)
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.captureWarnings(True)
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


# Smart App Control blocks one of scikit-learn's DLLs, and nothing here calls
# it: while torch/transformers load it must look absent (they probe its spec),
# and librosa's later imports get empty stand-ins without DLLs.
def _unavailable(*a, **k):
    raise RuntimeError("scikit-learn is stubbed out in the TTS server")


def _stub_attr(attr):
    if attr.startswith("__"):
        raise AttributeError(attr)
    return _unavailable


def _stub_sklearn():
    for name in ("sklearn", "sklearn.decomposition", "sklearn.cluster",
                 "sklearn.feature_extraction", "sklearn.neighbors", "sklearn.metrics"):
        mod = types.ModuleType(name)
        mod.__path__ = []
        mod.__getattr__ = _stub_attr
        sys.modules[name] = mod
        if "." in name:
            setattr(sys.modules["sklearn"], name.split(".", 1)[1], mod)


# Chatterbox Multilingual reads the attention of these layers to keep speech
# aligned with the text (AlignmentStreamAnalyzer's LLAMA_ALIGNED_HEADS).
SPIED_LAYERS = (9, 12, 13)


def drop_attention_spies(model):
    """Each generate() adds 3 forward hooks that are never removed, each
    copying its layer's attention to the CPU on every token: remove the old
    ones before the next generate() adds its own. Unmeasurable after 20
    sentences (93 hooks), but they pile up for as long as the server runs."""
    for idx in SPIED_LAYERS:
        model.t3.tfmr.layers[idx].self_attn._forward_hooks.clear()


def load_model(device: str, voice: str | None):
    sys.modules["sklearn"] = None
    import torch
    import torch._dynamo  # noqa: F401
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    if device == "cuda" and not torch.cuda.is_available():
        log("CUDA not available; using the CPU (much slower).")
        device = "cpu"
    start = time.time()
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    _stub_sklearn()
    model.builtin_conds = model.conds
    if voice:
        model.prepare_conditionals(voice)
    log(f"Chatterbox loaded on {device} in {time.time() - start:.1f}s")
    return model


def to_wav(samples, sample_rate: int) -> bytes:
    import numpy as np
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return out.getvalue()


class Voices:
    """Chatterbox's conditionals per reference file, so each request can pick
    its voice without reloading the model. Call under the generation lock."""

    def __init__(self, model):
        self.model = model
        self.startup = model.conds
        self.cache = {}  # (path, mtime) -> conditionals

    def use(self, voice: str | None):
        """Point the model at voice: None keeps the startup voice, "" is
        Chatterbox's own, anything else a reference file."""
        if voice is None:
            conds = self.startup
        elif voice == "":
            conds = self.model.builtin_conds
        else:
            path = os.path.abspath(voice)
            key = (path, os.path.getmtime(path))
            conds = self.cache.get(key)
            if conds is None:
                start = time.time()
                self.model.prepare_conditionals(path)
                conds = self.model.conds
                if len(self.cache) >= MAX_CACHED_VOICES:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[key] = conds
                log(f"Voice {path} prepared in {time.time() - start:.1f}s")
        if conds is None:
            raise ValueError("no voice: pass a reference file")
        self.model.conds = conds


def check_voice(voice) -> str | None:
    """The request's "voice", or ValueError: a reference audio file, never
    anything else on disk."""
    if voice is None or voice == "":
        return voice
    if not isinstance(voice, str) or not voice.lower().endswith(VOICE_SUFFIXES):
        raise ValueError(f"voice must be one of {', '.join(VOICE_SUFFIXES)} files")
    if not os.path.isfile(voice):
        raise ValueError(f"voice {voice!r} not found")
    return voice


def make_handler(model, default_language: str):
    # One generation at a time: the GPU is shared and Chatterbox is not
    # thread-safe.
    lock = threading.Lock()
    voices = Voices(model)

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code, body: bytes, content_type="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, code, message):
            self._reply(code, json.dumps({"error": message}).encode("utf-8"))

        def do_GET(self):
            if self.path != "/health":
                return self._error(404, "not found")
            self._reply(200, json.dumps({"ok": True, "sample_rate": model.sr}).encode())

        def do_POST(self):
            if self.path != "/tts":
                return self._error(404, "not found")
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY_BYTES:
                return self._error(413, "body missing or too large")
            try:
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                text = str(request["text"]).strip()[:MAX_TEXT_CHARS]
                language = str(request.get("language") or default_language)
                voice = check_voice(request.get("voice"))
            except Exception as e:
                return self._error(400, f"bad request: {e}")
            if not text:
                return self._error(400, "empty text")
            import torch
            with lock:
                start = time.time()
                try:
                    voices.use(voice)
                except Exception as e:
                    log(f"Voice {voice!r} unusable: {type(e).__name__}: {e}")
                    return self._error(400, f"voice unusable: {e}")
                try:
                    drop_attention_spies(model)
                    with torch.inference_mode():
                        wav = model.generate(text, language_id=language)
                except Exception as e:
                    # Chatterbox fails on text too short to speak ("OK").
                    log(f"Failed after {time.time() - start:.1f}s on {text[:300]!r}: "
                        f"{type(e).__name__}: {e}")
                    return self._error(500, f"generation failed: {e}")
            samples = wav.squeeze(0).float().cpu().numpy()
            log(f"{len(samples) / model.sr:.1f}s of audio in {time.time() - start:.1f}s: "
                f"{text[:300]!r}")
            self._reply(200, to_wav(samples, model.sr), "audio/wav")

        def log_message(self, fmt, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--voice", help="Reference audio to clone (wav/flac, ~10 s)")
    parser.add_argument("--language", default="pt", help="Default language id")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    setup_logging()
    log(f"Starting (pid {os.getpid()}, voice {args.voice or 'default'}, "
        f"language {args.language})")
    model = load_model(args.device, args.voice)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(model, args.language))
    log(f"Listening on http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("Stopped.")


if __name__ == "__main__":
    main()
