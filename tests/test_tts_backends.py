"""Backend contracts: no model packages or weights are downloaded."""
import importlib.util
import io
import json
from pathlib import Path
import runpy
import socket
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import wave
from contextlib import nullcontext

import numpy as np
import pytest

from debora_whisper import tts_server as server
from debora_whisper import tts_server_kokoro as kokoro
from debora_whisper import tts_server_qwen3 as qwen
from debora_whisper import voice_chat as vc
from debora_whisper.dictation_engine import DEFAULT_CONFIG, validate_config

ROOT = Path(__file__).resolve().parents[1]


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    @property
    def shape(self):
        return self.values.shape

    def reshape(self, *shape):
        return FakeTensor(self.values.reshape(*shape))

    def squeeze(self, axis):
        return FakeTensor(self.values.squeeze(axis))

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeModule:
    def __init__(self):
        self.hooks = []

    def modules(self):
        return [self]

    def children(self):
        return []

    def register_forward_pre_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))

    def __call__(self):
        for hook in self.hooks:
            hook(self, ())


@pytest.fixture
def fake_torch(monkeypatch):
    # The app/test environment intentionally has no torch dependency.
    fake = SimpleNamespace(
        float32="torch.float32", float16="torch.float16", bfloat16="torch.bfloat16",
        cuda=SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False),
        inference_mode=nullcontext, as_tensor=FakeTensor,
        cat=lambda chunks: FakeTensor(np.concatenate([chunk.values for chunk in chunks])))
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


@pytest.mark.parametrize("backend,filename", [
    ("chatterbox", "tts_server.py"), ("qwen3", "tts_server_qwen3.py"),
    ("kokoro", "tts_server_kokoro.py"),
])
def test_backend_selection(monkeypatch, backend, filename):
    monkeypatch.setattr(vc.shutil, "which", lambda _: "uv.exe")
    config = {**DEFAULT_CONFIG, "tts_backend": backend}
    validate_config(config)
    command = vc.tts_command(config)
    assert command[:3] == ["uv.exe", "run", "--script"]
    assert Path(command[3]).name == filename
    assert Path(command[3]).is_file()
    assert ("--voice" in command) == (backend != "kokoro")


def test_default_and_legacy_configs_keep_chatterbox(monkeypatch):
    assert DEFAULT_CONFIG["tts_backend"] == "chatterbox"
    monkeypatch.setattr(vc.shutil, "which", lambda _: "uv.exe")
    config = {k: v for k, v in DEFAULT_CONFIG.items() if k != "tts_backend"}
    validate_config(config)
    assert Path(vc.tts_command(config)[3]).name == "tts_server.py"


@pytest.mark.parametrize("backend", [None, "qwen3-tts", "", "unknown", 1, [], {}])
def test_invalid_backend_rejected(backend):
    with pytest.raises(ValueError, match="tts_backend"):
        validate_config({**DEFAULT_CONFIG, "tts_backend": backend})


def test_qwen_missing_voice_is_not_silently_replaced(monkeypatch, tmp_path):
    monkeypatch.setattr(vc.shutil, "which", lambda _: "uv.exe")
    missing = tmp_path / "missing.wav"
    command = vc.tts_command({**DEFAULT_CONFIG, "tts_backend": "qwen3", "tts_voice": str(missing)})
    assert command[command.index("--voice") + 1] == str(missing)


def response(handler_class, path="/health", body=None, headers=None):
    """Exercise HTTP handlers without binding a socket or starting a server."""
    handler = handler_class.__new__(handler_class)
    handler.path = path
    handler._reply = Mock()
    handler.connection = object()
    if body is None:
        handler.do_GET()
    else:
        payload = json.dumps(body).encode()
        handler.headers = {"Content-Type": "application/json", "Content-Length": str(len(payload)),
                           **(headers or {})}
        handler.rfile = io.BytesIO(payload)
        handler.do_POST()
    return handler._reply.call_args.args


@pytest.mark.parametrize("transcript", [None, "", "  \n"])
def test_missing_or_empty_transcript_fails_at_health_before_model_load(tmp_path, monkeypatch, transcript):
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"reference")
    if transcript is not None:
        voice.with_suffix(".txt").write_text(transcript, encoding="utf-8")
    load = Mock(side_effect=AssertionError("must not load a model"))
    monkeypatch.setitem(sys.modules, "qwen_tts", SimpleNamespace(Qwen3TTSModel=SimpleNamespace(from_pretrained=load)))
    model, error = server.load_backend(qwen.QwenBackend, "cpu", str(voice), "qwen-tts", qwen.MODEL_ID)
    code, body = response(server.make_handler(model, "pt", adapter=True, error=error))
    health = json.loads(body)
    assert code == 503 and not health["ok"]
    assert "voice.txt" in health["error"] and "exact words" in health["error"]
    assert not health["inventory"]["loaded"]
    load.assert_not_called()


def test_missing_espeak_fails_at_health_before_model_load(monkeypatch):
    monkeypatch.setitem(sys.modules, "misaki.espeak", SimpleNamespace(
        EspeakG2P=Mock(side_effect=RuntimeError("espeak not installed"))))
    load = Mock(side_effect=AssertionError("must not load a model"))
    monkeypatch.setitem(sys.modules, "kokoro", SimpleNamespace(KPipeline=load))
    model, error = server.load_backend(kokoro.KokoroBackend, "cpu", None, "kokoro", kokoro.MODEL_ID)
    handler = server.make_handler(model, "pt", adapter=True, error=error)
    code, body = response(handler)
    assert code == 503
    assert "Install espeak-ng" in json.loads(body)["error"]
    assert "PHONEMIZER_ESPEAK_LIBRARY" in json.loads(body)["error"]
    assert response(handler, "/tts", {"text": "Olá"})[0] == 503
    load.assert_not_called()


@pytest.fixture
def qwen_model(monkeypatch, tmp_path, fake_torch):
    inner = FakeModule()
    inner.speech_tokenizer = SimpleNamespace(model=FakeModule())
    model = Mock(model=inner)
    model.generate_voice_clone.return_value = ([np.zeros(2400, dtype=np.float32)], 24000)
    load = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "qwen_tts", SimpleNamespace(Qwen3TTSModel=SimpleNamespace(from_pretrained=load)))
    voice = tmp_path / "voice.wav"
    voice.write_bytes(b"reference")
    voice.with_suffix(".txt").write_text("Olá, referência confirmada.", encoding="utf-8")
    backend = qwen.QwenBackend("cpu", str(voice))
    return backend, model, load, voice


def test_qwen_uses_exact_transcript_and_caches_prompt(qwen_model):
    backend, model, load, voice = qwen_model
    assert load.call_args.args == (qwen.MODEL_ID,)
    assert load.call_args.kwargs["attn_implementation"] == "sdpa"
    model.create_voice_clone_prompt.assert_called_once_with(
        ref_audio=str(voice.resolve()), ref_text="Olá, referência confirmada.", x_vector_only_mode=False)
    backend.use(None)
    backend.prepare_voice()
    assert model.create_voice_clone_prompt.call_count == 1
    assert backend.generate("Olá, mundo.", "pt-BR").shape == (1, 2400)
    assert model.generate_voice_clone.call_args.kwargs["language"] == "Portuguese"
    assert model.generate_voice_clone.call_args.kwargs["non_streaming_mode"] is True
    voice.with_suffix(".txt").write_text("Texto corrigido.", encoding="utf-8")
    backend.use(str(voice))
    backend.prepare_voice()
    assert model.create_voice_clone_prompt.call_args.kwargs["ref_text"] == "Texto corrigido."
    with pytest.raises(ValueError, match="requires tts_voice"):
        backend.use("")


def test_kokoro_fixed_voice_and_complete_audio(monkeypatch, fake_torch):
    phonemizer = Mock(return_value=Mock(return_value=("ola", None)))
    monkeypatch.setitem(sys.modules, "misaki.espeak", SimpleNamespace(EspeakG2P=phonemizer))
    pipeline = Mock(model=FakeModule())
    pipeline.return_value = [("Olá", "ola", FakeTensor(np.zeros(1200))),
                             ("mundo", "mundo", FakeTensor(np.zeros(2400)))]
    load = Mock(return_value=pipeline)
    monkeypatch.setitem(sys.modules, "kokoro", SimpleNamespace(KPipeline=load))
    backend = kokoro.KokoroBackend("cpu", "ignored.wav")
    phonemizer.assert_called_once_with(language="pt-br")
    load.assert_called_once_with(lang_code="p", repo_id=kokoro.MODEL_ID, device="cpu")
    backend.use("another.wav")
    backend.prepare_voice()
    assert backend.generate("Olá, mundo.", "pt").shape == (1, 3600)
    pipeline.assert_called_once_with("Olá, mundo.", voice="pf_dora")


def test_adapter_http_contract_and_origin_checks(qwen_model, monkeypatch):
    backend, model, _, _ = qwen_model
    monkeypatch.setattr(server, "check_connection", lambda _: None)
    logs = []
    monkeypatch.setattr(server, "log", logs.append)
    handler = server.make_handler(backend, "pt", adapter=True)
    code, body = response(handler)
    assert code == 200 and json.loads(body)["sample_rate"] == 24000
    assert response(handler, "/tts", {"text": "Olá"}, {"Origin": "null"})[0] == 403
    assert response(handler, "/tts", {"text": "Olá"}, {"Content-Type": "text/plain"})[0] == 403
    model.generate_voice_clone.assert_not_called()
    code, wav, content_type = response(handler, "/tts", {"text": "Olá, mundo."})
    assert code == 200 and content_type == "audio/wav"
    with wave.open(io.BytesIO(wav), "rb") as audio:
        assert audio.getnframes() == 2400 and audio.getframerate() == 24000
    assert any("s of audio in " in line for line in logs)


def test_adapter_disconnect_cancels_and_removes_hooks(qwen_model, monkeypatch):
    backend, _, _, _ = qwen_model
    client, connection = socket.socketpair()
    # Advance monotonically beyond the 20 ms poll throttle at every call.
    ticks = iter(range(100))
    monkeypatch.setattr(server.time, "monotonic", lambda: next(ticks, 100))
    try:
        with pytest.raises(server.SynthesisCancelled):
            with server.cancellable_synthesis(backend, connection, backend.cancel_modules):
                client.close()
                backend.cancel_modules[0]()
        assert all(not module.hooks for module in backend.cancel_modules)
    finally:
        client.close()
        connection.close()


@pytest.mark.parametrize("filename", ["debora_whisper/tts_server.py",
    "debora_whisper/tts_server_qwen3.py", "debora_whisper/tts_server_kokoro.py",
    "docs/benchmarks/tts-bench/bench_tts.py"])
def test_pep723_pins_and_scoped_indexes(filename):
    import tomllib
    from packaging.requirements import Requirement
    text = (ROOT / filename).read_text(encoding="utf-8")
    block = text.split("# /// script\n", 1)[1].split("# ///", 1)[0]
    metadata = tomllib.loads("\n".join(line[2:] if line.startswith("# ") else "" for line in block.splitlines()))
    for spec in metadata["dependencies"]:
        requirement = Requirement(spec)
        pins = list(requirement.specifier)
        assert len(pins) == 1 and pins[0].operator == "==" and "*" not in pins[0].version
    if "tts_server" in filename:
        uv = metadata["tool"]["uv"]
        assert uv["index"] == [{"name": "pytorch-cu124", "url": "https://download.pytorch.org/whl/cu124", "explicit": True}]
        assert uv["sources"] == {name: {"index": "pytorch-cu124"} for name in ("torch", "torchaudio")}
    else:
        assert metadata["dependencies"] == []  # stdlib-only benchmark needs no index


def test_default_transcript_is_exact_and_packaged():
    import tomllib
    expected = "Sistemas iniciados com sucesso. Estou processando o áudio localmente, garantindo a sua privacidade. A captação está limpa. Pode começar a ditar quando quiser. Eu estarei aqui, ouvindo."
    assert (ROOT / "debora_whisper/voices/debora_v2.txt").read_text(encoding="utf-8").splitlines() == [expected]
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    targets = project["tool"]["hatch"]["build"]["targets"]
    assert targets["wheel"]["force-include"]["debora_whisper/voices"] == "debora_whisper/voices"
    assert "debora_whisper" in targets["sdist"]["include"]


@pytest.mark.parametrize("filename", ["tts_server_qwen3.py", "tts_server_kokoro.py"])
def test_script_imports_sibling_http_code_without_app_install(monkeypatch, filename):
    monkeypatch.syspath_prepend(str(ROOT / "debora_whisper"))
    namespace = runpy.run_path(str(ROOT / "debora_whisper" / filename), run_name="isolated_backend")
    assert Path(namespace["server"].__file__).resolve() == ROOT / "debora_whisper/tts_server.py"


def test_benchmark_response_metrics_and_repo_output_guard(monkeypatch):
    spec = importlib.util.spec_from_file_location("bench_tts", ROOT / "docs/benchmarks/tts-bench/bench_tts.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    wav = server.to_wav(np.zeros(24000), 24000)
    opener = Mock()
    opener.open.return_value = io.BytesIO(wav)
    ticks = iter([10.0, 10.25])
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(ticks, 10.25))
    result, metrics = bench.synthesize(opener, "http://127.0.0.1:8765", "Olá.", None, 60)
    assert result == wav
    assert metrics == {"response_seconds": 0.25, "audio_seconds": 1.0, "rtf": 0.25}
    assert bench.percentile([1, 2, 3, 4], 0.5) == 2.5
    assert bench.percentile([1, 2, 3, 4], 0.9) == pytest.approx(3.7)
    with pytest.raises(ValueError, match="outside a repository"):
        bench.output_dir(ROOT / "generated-audio")
    assert len((ROOT / "docs/benchmarks/tts-bench/frases_ptbr.txt").read_text(encoding="utf-8").splitlines()) == 20
