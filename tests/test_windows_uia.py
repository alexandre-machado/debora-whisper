"""Caret reads use mocked COM objects, never the desktop."""
import builtins
import ctypes
import gc
import runpy
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from debora_whisper.platforms.base import Platform
from debora_whisper.platforms.windows import WindowsPlatform
from debora_whisper.platforms import windows_uia as uia


@pytest.fixture
def isolated_worker(monkeypatch):
    monkeypatch.setattr(uia, "_worker", None)
    yield
    if uia._worker is not None:
        requests, thread = uia._worker
        # Only mocked COM workers may be stopped; production lives until exit.
        requests.put(None)
        thread.join(timeout=3)
        assert not thread.is_alive()


@pytest.fixture
def com(monkeypatch, isolated_worker):
    constants = SimpleNamespace(
        CUIAutomation=object(), IUIAutomation=object(), IUIAutomationTextPattern2=object(),
        UIA_TextPattern2Id=10024, TextPatternRangeEndpoint_Start=0,
        TextPatternRangeEndpoint_End=1, TextUnit_Character=0)
    api = Mock(COINIT_MULTITHREADED=0)
    api.client.GetModule.return_value = constants
    element = api.client.CreateObject.return_value.GetFocusedElement.return_value
    pattern = element.GetCurrentPattern.return_value.QueryInterface.return_value
    caret = Mock()
    pattern.GetCaretRange.return_value = (True, caret)
    selection = pattern.GetSelection.return_value
    selection.Length = 1
    selection.GetElement.return_value.CompareEndpoints.return_value = 0
    caret.Clone.return_value.GetText.return_value = "draft"
    monkeypatch.setattr(uia, "comtypes", api)
    monkeypatch.setattr(uia, "_release", Mock())
    return SimpleNamespace(api=api, constants=constants, element=element,
                           pattern=pattern, caret=caret, selection=selection)


def test_generic_caret_is_unknown():
    assert Platform().text_before_caret(5) is None


def test_new_windows_provider_cannot_reach_real_uia():
    assert uia.comtypes is None
    assert WindowsPlatform().text_before_caret(5) is None
    assert uia._worker is None


@pytest.mark.parametrize("observed", ["draft", "other", "", "é👍\r\n"])
def test_windows_reads_actual_text(com, observed):
    com.caret.Clone.return_value.GetText.return_value = observed
    threads = []
    com.api.CoInitializeEx.side_effect = lambda flags: threads.append(threading.get_ident())
    assert WindowsPlatform().text_before_caret(5) == observed
    assert threads != [threading.get_ident()]
    com.api.CoInitializeEx.assert_called_once_with(0)
    com.api.CoUninitialize.assert_not_called()
    com.api.client.GetModule.assert_called_once_with("UIAutomationCore.dll")
    com.element.GetCurrentPattern.assert_called_once_with(10024)
    com.element.GetCurrentPattern.return_value.QueryInterface.assert_called_once_with(
        com.constants.IUIAutomationTextPattern2)
    selected = com.selection.GetElement.return_value
    selected.CompareEndpoints.assert_called_once_with(0, selected, 1)
    com.caret.Clone.return_value.MoveEndpointByUnit.assert_called_once_with(0, 0, -5)
    com.caret.Clone.return_value.GetText.assert_called_once_with(-1)
    com.caret.MoveEndpointByUnit.assert_not_called()


@pytest.mark.parametrize("condition", ["inactive", "selected", "multiple", "no_selection",
                                        "missing_pattern", "exception", "init_failure"])
def test_windows_unknown(com, condition):
    if condition == "inactive":
        com.pattern.GetCaretRange.return_value = (False, com.caret)
    elif condition == "selected":
        com.selection.GetElement.return_value.CompareEndpoints.return_value = -1
    elif condition in ("multiple", "no_selection"):
        com.selection.Length = 2 if condition == "multiple" else 0
    elif condition == "missing_pattern":
        com.element.GetCurrentPattern.return_value = None
    elif condition == "exception":
        com.pattern.GetSelection.side_effect = OSError("COM failure")
    else:
        com.api.CoInitializeEx.side_effect = OSError("COM failure")
    assert WindowsPlatform().text_before_caret(5) is None
    com.api.CoUninitialize.assert_not_called()


def test_timeout_discards_late_result_and_bounds_workers(com):
    release = threading.Event()
    entered = threading.Event()

    def blocked():
        entered.set()
        release.wait(2)
        return True, com.caret

    com.pattern.GetCaretRange.side_effect = blocked
    try:
        start = time.monotonic()
        assert WindowsPlatform().text_before_caret(5) is None
        assert time.monotonic() - start < 0.75
        assert entered.is_set()
        assert WindowsPlatform().text_before_caret(5) is None
        com.api.CoInitializeEx.assert_called_once()
    finally:
        release.set()
        assert uia._query_lock.acquire(timeout=2)
        uia._query_lock.release()
    com.pattern.GetCaretRange.side_effect = None
    com.caret.Clone.return_value.GetText.return_value = "fresh"
    assert WindowsPlatform().text_before_caret(5) == "fresh"
    com.api.CoInitializeEx.assert_called_once()
    com.api.CoUninitialize.assert_not_called()


def test_queries_reuse_initialized_worker_after_provider_failure(com):
    threads = []
    com.api.client.GetModule.side_effect = lambda name: (
        threads.append(threading.get_ident()) or com.constants)
    assert uia.text_before_caret(5) == "draft"
    worker = uia._worker[1]
    com.pattern.GetSelection.side_effect = OSError("provider unavailable")
    assert uia.text_before_caret(5) is None
    com.pattern.GetSelection.side_effect = None
    assert uia.text_before_caret(5) == "draft"
    assert threads == [worker.ident] * 3
    assert worker.is_alive()
    com.api.CoInitializeEx.assert_called_once_with(0)
    com.api.CoUninitialize.assert_not_called()


def test_initialization_failure_can_retry(com):
    com.api.CoInitializeEx.side_effect = [OSError("COM unavailable"), None]
    assert uia.text_before_caret(5) is None
    assert uia.text_before_caret(5) == "draft"
    assert com.api.CoInitializeEx.call_count == 2
    com.api.CoUninitialize.assert_not_called()


def test_worker_start_failure_releases_query_slot(com, monkeypatch):
    start = threading.Thread.start
    monkeypatch.setattr(threading.Thread, "start", Mock(side_effect=RuntimeError("no thread")))
    assert uia.text_before_caret(5) is None
    assert uia._worker is None
    monkeypatch.setattr(threading.Thread, "start", start)
    assert uia.text_before_caret(5) == "draft"


@pytest.mark.parametrize("failure", [None, "GetFocusedElement", "GetCurrentPattern",
                                    "QueryInterface", "GetCaretRange", "GetSelection",
                                    "GetElement", "CompareEndpoints", "Clone",
                                    "MoveEndpointByUnit", "GetText", "inactive",
                                    "selected", "multiple", "release"])
def test_cyclic_pointers_released_on_worker_even_after_error(monkeypatch, isolated_worker,
                                                           failure):
    pointers = []
    created = []
    released = []
    initialized = []

    class Pointer(ctypes.c_void_p):
        def __init__(self, name):
            super().__init__(1)
            self.name = name
            self.cycle = self  # Reference-dropping alone cannot release this.
            pointers.append(self)
            created.append((name, threading.get_ident()))

        def Release(self):
            released.append((self.name, threading.get_ident()))
            if failure == "release" and self.name == "Clone":
                raise OSError("provider release failure")

        def __del__(self):
            if self:
                self.Release()

        @property
        def Length(self):
            return 2 if failure == "multiple" else 1

        def __getattr__(self, method):
            def call(*args):
                assert initialized == [threading.get_ident()]
                if method == failure:
                    raise OSError("provider failure retaining this frame")
                if method == "GetCaretRange":
                    return failure != "inactive", Pointer("caret")
                if method == "CompareEndpoints":
                    return -1 if failure == "selected" else 0
                if method == "MoveEndpointByUnit":
                    return -5
                if method == "GetText":
                    return "draft"
                return Pointer(method)
            return call

    constants = SimpleNamespace(
        CUIAutomation=object(), IUIAutomation=object(), IUIAutomationTextPattern2=object(),
        UIA_TextPattern2Id=10024, TextPatternRangeEndpoint_Start=0,
        TextPatternRangeEndpoint_End=1, TextUnit_Character=0)
    api = Mock(COINIT_MULTITHREADED=0)
    api.CoInitializeEx.side_effect = lambda flags: initialized.append(threading.get_ident())
    api.client.GetModule.return_value = constants
    api.client.CreateObject.side_effect = lambda *a, **kw: Pointer("automation")
    monkeypatch.setattr(uia, "comtypes", api)

    assert uia.text_before_caret(5) == ("draft" if failure in (None, "release") else None)
    assert initialized != [threading.get_ident()]
    assert released == list(reversed(created))
    assert all(not pointer for pointer in pointers)
    pointers.clear()
    gc.collect()  # A different thread collects the deliberately retained cycles.
    assert released == list(reversed(created))  # No second Release from __del__.
    api.CoUninitialize.assert_not_called()


@pytest.mark.skipif(sys.platform != "win32", reason="comtypes is Windows-only")
def test_release_clears_real_comtypes_pointer_without_calling_native_com():
    comtypes = pytest.importorskip("comtypes")
    pointer = ctypes.POINTER(comtypes.IUnknown)()
    release = Mock()
    pointer.Release = release
    pointer.cycle = pointer
    # Synthetic address, with Release replaced before making it non-null.
    ctypes.c_void_p.value.__set__(pointer, 1)
    uia._release(pointer)
    assert not pointer
    release.assert_called_once_with()
    del pointer
    gc.collect()
    release.assert_called_once_with()


@pytest.mark.parametrize("failure", [ImportError, OSError])
def test_comtypes_import_failure_is_optional(monkeypatch, failure):
    real_import = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "comtypes.client":
            raise failure("unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    module = runpy.run_path(uia.__file__)
    assert module["text_before_caret"](5) is None
