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
"""Local TTS server for npu-whisper's voice chat, backed by Chatterbox
Multilingual on an NVIDIA GPU.

It never runs in the app's environment: chatterbox-tts pins torch 2.6 (CUDA)
and numpy 1.26. The block above lets uv build its own environment on the first
run (a few GB, cached afterwards):

    uv run --script npu_whisper/tts_server.py --voice ref.wav

npu-whisper starts it this way by itself in voice chat mode, with the voice
from "tts_voice" in config.json, and stops it on exit.

Endpoints (127.0.0.1 only):
    GET  /health -> {"ok": true, "sample_rate": ...}
    POST /tts    {"text": "...", "language": "pt"} -> audio/wav (16-bit mono)
"""
import argparse
import io
import json
import sys
import threading
import time
import types
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_TEXT_CHARS = 1000
MAX_BODY_BYTES = 64_000


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


def load_model(device: str, voice: str | None):
    sys.modules["sklearn"] = None
    import torch
    import torch._dynamo  # noqa: F401
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using the CPU (much slower).")
        device = "cpu"
    start = time.time()
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    _stub_sklearn()
    if voice:
        model.prepare_conditionals(voice)
    print(f"Chatterbox loaded on {device} in {time.time() - start:.1f}s")
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


def make_handler(model, default_language: str):
    # One generation at a time: the GPU is shared and Chatterbox is not
    # thread-safe.
    lock = threading.Lock()

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
            except Exception as e:
                return self._error(400, f"bad request: {e}")
            if not text:
                return self._error(400, "empty text")
            import torch
            with lock:
                start = time.time()
                try:
                    with torch.inference_mode():
                        wav = model.generate(text, language_id=language)
                except Exception as e:
                    return self._error(500, f"generation failed: {e}")
            samples = wav.squeeze(0).float().cpu().numpy()
            print(f"{len(samples) / model.sr:.1f}s of audio in {time.time() - start:.1f}s: "
                  f"{text[:60]!r}")
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
    model = load_model(args.device, args.voice)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(model, args.language))
    print(f"TTS server on http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
