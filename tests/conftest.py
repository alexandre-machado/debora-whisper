"""Shared fixtures for dictation engine tests."""
import sys
from pathlib import Path

import pytest

# Add project root to path so we can import dictation_engine
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def _reset_device_failure_latch():
    """The fatal device latch is process-wide; isolate it per test."""
    import dictation_engine
    dictation_engine._reset_device_failure_for_tests()
    yield
    dictation_engine._reset_device_failure_for_tests()
