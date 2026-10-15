"""Download telemetry without fetching weights or importing model backends."""
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from debora_whisper.download_progress import DownloadProgress, format_download, track_hf_downloads
from debora_whisper import tts_server, voice_chat as vc


def test_aggregate_concurrent_files_and_resume_even_with_disabled_bars():
    from huggingface_hub.utils import tqdm
    tracker = DownloadProgress()
    callback = Mock()
    with track_hf_downloads(tracker, callback):
        # The snapshot's file-count bar must not be counted as bytes.
        with tqdm(total=2, disable=True) as files:
            with tqdm(total=400_000_000, initial=100_000_000, unit="B", disable=True) as first:
                with tqdm(total=340_000_000, unit="B", disable=True) as second:
                    with ThreadPoolExecutor(2) as pool:
                        list(pool.map(lambda bar: bar.update(105_000_000), [first, second]))
                    files.update(1)
                    assert tracker.snapshot() == {"done": 310_000_000, "total": 740_000_000}
                    assert format_download("STT", tracker.snapshot()) == " DL STT 42% (310/740 MB)"
                    assert callback.call_args.args == (310_000_000, 740_000_000)
                # A completed/closed file's bytes remain in the aggregate.
                assert tracker.snapshot()["total"] == 740_000_000
            assert tracker.snapshot() is None
    assert tracker.snapshot() is None


def test_progress_cleared_and_hf_hooks_restored_after_failure():
    from huggingface_hub.utils import tqdm
    tracker = DownloadProgress()
    original = (tqdm.__init__, tqdm.update, tqdm.close)
    with pytest.raises(RuntimeError):
        with track_hf_downloads(tracker):
            bar = tqdm(total=100, unit="B", disable=True)
            bar.update(17)
            assert format_download("TTS", tracker.snapshot()) == " DL TTS 17% (0/0 MB)"
            raise RuntimeError("download interrupted")
    assert tracker.snapshot() is None
    assert (tqdm.__init__, tqdm.update, tqdm.close) == original
    bar.close()


def test_stt_setup_preserves_callback_and_clears_progress(monkeypatch, tmp_path):
    import huggingface_hub
    from huggingface_hub.utils import tqdm
    from debora_whisper import dictation_engine as de
    monkeypatch.setattr(de, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(de, "model_files_complete", Mock(side_effect=[False, True]))
    callback = Mock()

    def download(*args, **kwargs):
        with tqdm(total=100_000_000, unit="B", disable=True) as bar:
            bar.update(42_000_000)
            assert format_download("STT", de.STT_DOWNLOAD.snapshot()) == " DL STT 42% (42/100 MB)"
            bar.update(58_000_000)
            assert de.STT_DOWNLOAD.snapshot() is None

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert de.setup_model({**de.DEFAULT_CONFIG, "device": "CPU"}, callback).parent == tmp_path
    assert callback.call_args.args == (100_000_000, 100_000_000)
    assert de.STT_DOWNLOAD.snapshot() is None


@pytest.mark.parametrize("progress", [None, {}, {"done": 1, "total": 0},
                                      {"done": 10, "total": 10}, {"done": "bad", "total": 10}])
def test_no_segment_when_idle_or_invalid(progress):
    assert format_download("STT", progress) == ""


@pytest.fixture
def loading_tts(monkeypatch):
    from huggingface_hub.utils import tqdm
    started, release = threading.Event(), threading.Event()
    model = SimpleNamespace(package="missing-test-package", model_id="test", sr=24000,
                            _debora_device="CPU")
    failure = []

    def loader():
        with tqdm(total=100_000_000, unit="B", disable=True) as bar:
            bar.update(17_000_000)
            started.set()
            assert release.wait(10)
        if failure:
            raise RuntimeError(failure[0])
        return model, None

    server = tts_server.loading_server(("127.0.0.1", 0), loader, "pt")
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    assert started.wait(5)
    config = {"tts_url": f"http://127.0.0.1:{server.server_port}"}
    monkeypatch.setattr(vc, "_tts_process", None)
    monkeypatch.setattr(vc, "_tts_health_cache", {})
    try:
        yield config, release, failure
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        worker.join(2)


def test_loading_health_segment_no_extra_network_and_wait_until_ready(loading_tts, monkeypatch):
    config, release, _ = loading_tts
    assert not vc.tts_server_up(config)
    assert format_download("TTS", vc.tts_download_progress(config)) == " DL TTS 17% (17/100 MB)"
    connection = http.client.HTTPConnection(config["tts_url"].removeprefix("http://"), timeout=1)
    try:
        connection.request("POST", "/tts", b'{}')
        response = connection.getresponse()
        assert response.status == 503
        assert json.loads(response.read()) == {"error": "loading"}
    finally:
        connection.close()
    with monkeypatch.context() as patch:
        patch.setattr(vc.http.client, "HTTPConnection", Mock(side_effect=AssertionError("extra poll")))
        for _ in range(10):
            assert vc.tts_download_progress(config)["done"] == 17_000_000
        assert vc.tts_download_progress({"tts_url": "http://unused"}) is None
    # Existing startup checks must wait through loading even for external servers.
    timer = threading.Timer(0.6, release.set)
    timer.start()
    try:
        assert vc.wait_tts_server(config, timeout=5)
    finally:
        timer.cancel()
    assert vc.tts_download_progress(config) is None


def test_loading_failure_remains_terminal_and_clears_segment(loading_tts):
    config, release, failure = loading_tts
    assert not vc.tts_server_up(config)
    failure.append("missing dependency")
    release.set()
    with pytest.raises(vc.TTSStartupError, match="missing dependency"):
        vc.wait_tts_server(config, timeout=5)
    assert vc.tts_download_progress(config) is None


def test_external_loading_server_is_reused(loading_tts, monkeypatch):
    config, _, _ = loading_tts
    monkeypatch.setattr(vc, "_tts_retiring_url", None)
    monkeypatch.setattr(vc, "_tts_server_key", None)
    spawn = Mock(side_effect=AssertionError("duplicate server"))
    monkeypatch.setattr(vc.subprocess, "Popen", spawn)
    vc.ensure_tts_server({**config, "tts_server_command": ["never-run"]})
    spawn.assert_not_called()


def test_cached_progress_expires_without_polling(monkeypatch):
    monkeypatch.setattr(vc, "_tts_health_cache", {})
    config = {"tts_url": "http://unused"}
    vc._cache_tts_health(config, {"state": "loading", "download": {"done": 1, "total": 2}})
    assert vc.tts_download_progress(config)
    now = vc.time.monotonic()
    monkeypatch.setattr(vc.time, "monotonic", lambda: now + 7)
    assert vc.tts_download_progress(config) is None


def test_telemetry_appends_downloads_and_drops_them_when_idle(monkeypatch):
    from debora_whisper import dictation_engine as de
    app = de.DictationApp.__new__(de.DictationApp)
    app.config = {}
    app._accelerator_memory = lambda: ""
    app._stopping = Mock()
    app._stopping.wait.side_effect = [False, False, True]
    monkeypatch.setattr(de.STT_DOWNLOAD, "snapshot", Mock(side_effect=[{"done": 42_000_000, "total": 100_000_000}, None]))
    monkeypatch.setattr(de, "tts_download_progress", Mock(side_effect=[{"done": 17_000_000, "total": 100_000_000}, None]))
    logs = []
    monkeypatch.setattr(de, "log", logs.append)
    app._monitor_resources()
    assert len(logs) == 2
    assert logs[0].endswith(" DL STT 42% (42/100 MB) DL TTS 17% (17/100 MB)")
    assert " DL " not in logs[1]
    assert all(call.args == (5.0,) for call in app._stopping.wait.call_args_list)
