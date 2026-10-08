"""Shared fixtures for dictation engine tests."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Importing the package must never move the user's pre-rename folders, nor
# create model or cache folders in the user's shared MODELS_DIR.
os.environ["DEBORA_WHISPER_NO_MIGRATION"] = "1"
os.environ["MODELS_DIR"] = tempfile.mkdtemp(prefix="debora-models-")

# Add project root to path so we can import debora_whisper
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True, scope="session")
def _private_logs(tmp_path_factory):
    """Tests log to a temporary folder, never to the user's app.log, and
    never record a lost NPU for the real app. Not restored afterwards: daemon
    threads of finished tests may still log."""
    from debora_whisper import dictation_engine as engine
    home = tmp_path_factory.mktemp("debora")
    engine.LOG_DIR = home / "logs"
    engine.LOG_FILE = engine.LOG_DIR / "app.log"
    engine.TELEMETRY_LOG = engine.LOG_DIR / "telemetry.log"
    engine.TTS_SERVER_LOG = engine.LOG_DIR / "tts_server.log"
    engine.LLM_SERVER_LOG = engine.LOG_DIR / "llm_server.log"
    engine.NPU_LOST_FILE = home / "npu_lost.json"


@pytest.fixture(autouse=True)
def _no_npu_loss_record():
    from debora_whisper import dictation_engine as engine
    engine.NPU_LOST_FILE.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def _reset_device_failure_latch():
    """The fatal device latch is process-wide; isolate it per test."""
    from debora_whisper import dictation_engine
    dictation_engine._reset_device_failure_for_tests()
    yield
    dictation_engine._reset_device_failure_for_tests()
