"""Pure policy examples and resource admission, without hardware or user files."""

import copy
import threading

import pytest

from debora_whisper import device_inventory, device_planner
from debora_whisper.device_planner import plan


def _config(**changes):
    return {"model_size": "turbo", "device": "auto", "llm_device": "auto", "tts_device": "auto",
            "device_priority": ["CUDA", "NPU", "GPU", "CPU"], "llm_model": "Qwen-8B-int4", **changes}


def _inventory(kinds=("CUDA", "NPU", "GPU"), *, cuda_free=14336, ram_free=49152, parakeet=False):
    def device(kind):
        return {"id": "CUDA:0" if kind == "CUDA" else "CPU" if kind == "CPU" else "OV:" + kind,
                "kind": kind, "index": 0 if kind == "CUDA" else kind, "physical_id": kind + "-uuid",
                "name": kind, "driver": "1", "memory_shared": kind != "CUDA",
                "memory_total": (16384 if kind == "CUDA" else 65536) * 2**20,
                "memory_free": None if kind == "CUDA" and cuda_free is None else (
                    cuda_free if kind == "CUDA" else ram_free) * 2**20}

    observed = {
        "system": {"devices": [device("CPU")]},
        "nvidia": {"devices": [device("CUDA")] if "CUDA" in kinds else []},
        "openvino": {"devices": [{**device("CPU"), "id": "OV:CPU"}] +
                     [device(k) for k in kinds if k != "CUDA"]},
        "environment": {"versions": dict.fromkeys(("openvino", "openvino-genai", "onnxruntime",
                                                    "faster-whisper", "ctranslate2"), "1"),
                        "loss_checked": True, "artifacts": {
                            "openvino": {"model": "parakeet" if parakeet else "whisper-turbo", "complete": True},
                            "cuda": {"model": "whisper-turbo", "complete": True},
                            "llm": {"model": "Qwen-8B-int4", "complete": True}}},
        "tts": {"status": "ok", "device": "CUDA" if "CUDA" in kinds else "CPU",
                "physical_id": "CUDA-uuid" if "CUDA" in kinds else None,
                "model": "ChatterboxMultilingualTTS", "versions": {"torch": "1", "chatterbox-tts": "1"}},
    }
    inventory = device_inventory.make_inventory(observed, now=100)
    for row in inventory["capabilities"]:
        if row["prerequisites"]:
            row.update(validation="verified", ready=True)
        # Synthetic evidence of CPU TTS readiness as well as CUDA readiness.
        # Today's live probe reports only the server's current device.
        if row["device"] == "TTS:CPU":
            row.update(prerequisites=True, artifacts_complete=True, ready=True, validation="verified")
    return inventory


def _chain(entry):
    return ([entry["kind"]] if entry["kind"] else []) + [d["kind"] for d in entry["fallback_details"]]


@pytest.mark.parametrize("kinds,mode,stt,llm,tts", [
    (("CUDA", "NPU", "GPU"), "dictation", ["CUDA", "NPU", "GPU", "CPU"], [], []),
    (("CUDA", "NPU", "GPU"), "chat_local", ["NPU", "GPU", "CUDA", "CPU"], ["GPU", "CPU"], ["CUDA", "CPU"]),
    (("CUDA", "NPU", "GPU"), "chat_claude", ["NPU", "GPU", "CUDA", "CPU"], [], ["CUDA", "CPU"]),
    (("NPU", "GPU"), "dictation", ["NPU", "GPU", "CPU"], [], []),
    (("NPU", "GPU"), "chat_local", ["NPU", "GPU", "CPU"], ["GPU", "CPU"], ["CPU"]),
    (("NPU", "GPU"), "chat_claude", ["NPU", "GPU", "CPU"], [], ["CPU"]),
    ((), "dictation", ["CPU"], [], []),
    ((), "chat_local", ["CPU"], ["CPU"], ["CPU"]),
    ((), "chat_claude", ["CPU"], [], ["CPU"]),
])
def test_documented_examples(kinds, mode, stt, llm, tts):
    result = plan(_inventory(kinds), mode=mode, config=_config())["components"]
    assert _chain(result["stt"]) == stt
    assert _chain(result["llm"]) == llm
    assert _chain(result["tts"]) == tts
    if mode == "dictation":
        assert result["llm"]["state"] == result["tts"]["state"] == "inactive"
    if mode == "chat_claude":
        assert result["llm"]["state"] == "disabled"
        assert result["llm"]["reservation_mib"] is None


@pytest.mark.parametrize("free", [6500, None])
def test_insufficient_or_unknown_vram_blocks_cuda_sharing(free):
    # 6500 - 15% of 16384 fits TTS, but not TTS + Whisper.
    result = plan(_inventory(("CUDA",), cuda_free=free), mode="chat_local", config=_config())["components"]
    assert result["tts"]["kind"] == "CUDA"
    assert _chain(result["stt"]) == ["CPU"]


def test_unknown_memory_allows_one_accelerator_load():
    result = plan(_inventory(("CUDA",), cuda_free=None), mode="dictation", config=_config())["components"]["stt"]
    assert result["kind"] == "CUDA"
    assert result["confidence"] == "unknown"


@pytest.mark.parametrize("kinds,expected", [(("CUDA", "NPU", "GPU"), "CUDA"), (("NPU", "GPU"), None)])
def test_shared_ram_is_one_budget(kinds, expected):
    inventory = _inventory(kinds, ram_free=14000 if "CUDA" not in kinds else 11000)
    for d in inventory["devices"]:
        if d["memory_shared"]:
            d["memory_total"] = 16384 * 2**20
    result = plan(inventory, mode="chat_local", config=_config())["components"]
    assert result["llm"]["kind"] == "GPU"
    assert result["stt"]["kind"] == expected
    assert "NPU" not in _chain(result["stt"])
    assert "GPU" not in _chain(result["stt"])
    if expected is None:
        assert result["stt"]["state"] == "unavailable"


@pytest.mark.parametrize("changes,component,expected", [
    ({"device": "gpu"}, "stt", "GPU"),
    ({"device": "CPU"}, "stt", "CPU"),
    ({"llm_device": "cpu"}, "llm", "CPU"),
    ({"tts_device": "CPU"}, "tts", "CPU"),
])
def test_explicit_override_wins(changes, component, expected):
    result = plan(_inventory(), mode="chat_local", config=_config(**changes))["components"][component]
    assert result["kind"] == expected
    assert "explicit" in result["reason"]


def test_explicit_stt_reserves_before_automatic_tts():
    result = plan(_inventory(("CUDA",), cuda_free=6500), mode="chat_local",
                  config=_config(device="CUDA"))["components"]
    assert result["stt"]["kind"] == "CUDA"
    assert result["tts"]["kind"] == "CPU"


def test_auto_is_case_insensitive_and_priority_breaks_ties():
    result = plan(_inventory(), mode="dictation", config=_config(device="AuTo", device_priority=["GPU", "NPU", "CUDA"]))
    assert _chain(result["components"]["stt"]) == ["GPU", "NPU", "CUDA", "CPU"]


@pytest.mark.parametrize("mode", ["dictation", "chat_local", "chat_claude"])
@pytest.mark.parametrize("device", ["auto", "CPU", "CUDA"])
def test_parakeet_has_encoder_decoder_and_never_cuda(mode, device):
    result = plan(_inventory(parakeet=True), mode=mode, config=_config(model_size="parakeet", device=device))["components"]
    assert "stt" not in result
    encoder, decoder = result["stt.parakeet.encoder"], result["stt.parakeet.decoder"]
    assert encoder["kind"] == ("CPU" if device == "CPU" else "NPU")
    assert decoder["kind"] == ("CPU" if device == "CPU" else "GPU")
    assert "CUDA" not in _chain(encoder) + _chain(decoder)
    assert "NPU" not in _chain(decoder)
    assert encoder["model"] == decoder["model"] == "parakeet"
    assert encoder["backend"] == decoder["backend"] == "openvino"


def test_parakeet_cpu_only():
    result = plan(_inventory((), parakeet=True), mode="dictation", config=_config(model_size="parakeet"))["components"]
    assert result["stt.parakeet.encoder"]["kind"] == result["stt.parakeet.decoder"]["kind"] == "CPU"


@pytest.mark.parametrize("state,expected", [("unverified", ["GPU", "CPU", "NPU"]),
                                            ("quarantined", ["GPU", "CPU"])])
def test_npu_requires_validation_even_with_override(state, expected):
    inventory = _inventory(("NPU", "GPU"))
    for row in inventory["capabilities"]:
        if row["device"] == "OV:NPU":
            row.update(validation=state, ready=False)
    result = plan(inventory, mode="chat_local", config=_config(device="NPU"))["components"]["stt"]
    assert _chain(result) == expected
    if state == "unverified":
        assert result["fallback_details"][-1]["requires_validation"]


@pytest.mark.parametrize("field,value", [("supported", False), ("dependencies_installed", False),
                                        ("artifacts_complete", False), ("prerequisites", False),
                                        ("validation", "quarantined")])
def test_capability_filters_apply_to_preferred_and_fallbacks(field, value):
    inventory = _inventory()
    for row in inventory["capabilities"]:
        if row["device"] == "CUDA:0":
            row[field] = value
    result = plan(inventory, mode="dictation", config=_config())["components"]["stt"]
    assert _chain(result) == ["NPU", "GPU", "CPU"]


def test_absent_device_is_not_a_candidate():
    inventory = _inventory()
    inventory["devices"] = [d for d in inventory["devices"] if d["kind"] != "NPU"]
    assert "NPU" not in _chain(plan(inventory, mode="dictation", config=_config())["components"]["stt"])


@pytest.mark.parametrize("inventory", [None, device_inventory.make_inventory({}, now=0)])
def test_missing_inventory_is_conservative(inventory):
    result = plan(inventory, mode="chat_local", config=_config(device="NPU"), generation=7)
    assert result["id"] == result["generation"] == 7
    for entry in result["components"].values():
        assert entry["kind"] == "CPU"
        assert entry["confidence"] == "unknown"


def test_timeout_does_not_authorize_known_missing_artifacts():
    inventory = _inventory(())
    inventory.update(timed_out=True)
    for row in inventory["capabilities"]:
        row["artifacts_complete"] = False
    result = plan(inventory, mode="dictation", config=_config())["components"]["stt"]
    assert result["state"] == "unavailable"


def test_partial_inventory_keeps_observed_facts_with_low_confidence():
    inventory = _inventory(("GPU",))
    inventory.update(timed_out=True, pending=["nvidia"])
    result = plan(inventory, mode="dictation", config=_config())["components"]["stt"]
    assert _chain(result) == ["GPU", "CPU"]
    assert result["confidence"] == "unknown"


def test_measured_latency_beats_priority_and_measured_peak_is_used():
    inventory = _inventory()
    for row in inventory["capabilities"]:
        if row["component"] == "stt.whisper":
            inventory["measurements"][row["id"]] = {
                "peak_bytes": 1200 * 2**20, "latency_ms": 10 if row["device"] == "OV:GPU" else 20}
    result = plan(inventory, mode="dictation", config=_config())["components"]["stt"]
    assert result["kind"] == "GPU"
    assert result["reservation_mib"] == 1200
    assert result["confidence"] == "measured"


def test_resident_model_is_not_reserved_again_and_breaks_ties():
    inventory = _inventory(cuda_free=2800)
    loaded = {"stt": {"device": "CUDA:0", "model": "turbo"}}
    result = plan(inventory, mode="dictation", config=_config(device_priority=["GPU", "CUDA", "NPU"]), loaded=loaded)
    assert result["components"]["stt"]["kind"] == "CUDA"
    assert "already resident" in result["components"]["stt"]["reason"]
    loaded["stt"]["model"] = "small"
    assert plan(inventory, mode="dictation", config=_config(), loaded=loaded)["components"]["stt"]["kind"] == "NPU"


def test_inactive_resident_tts_blocks_unknown_cuda_sharing():
    result = plan(_inventory(cuda_free=None), mode="dictation", config=_config(),
                  loaded={"tts": {"device": "CUDA:0", "model": "ChatterboxMultilingualTTS"}})["components"]
    assert result["stt"]["kind"] == "NPU"
    assert "CUDA" not in _chain(result["stt"])
    assert result["tts"]["state"] == "inactive"


def test_resident_tts_free_memory_is_not_charged_twice():
    result = plan(_inventory(("CUDA",), cuda_free=4500), mode="chat_claude", config=_config(),
                  loaded={"tts": {"device": "CUDA:0", "model": "ChatterboxMultilingualTTS"}})["components"]
    assert result["tts"]["kind"] == result["stt"]["kind"] == "CUDA"


def test_cpu_also_requires_memory_headroom():
    inventory = _inventory((), ram_free=1800)
    inventory["devices"][0]["memory_total"] = 2048 * 2**20
    result = plan(inventory, mode="dictation", config=_config())["components"]["stt"]
    assert result["state"] == "unavailable"  # 1800 - minimum margin 512 < 1500.


def test_tts_unknown_uuid_never_assumes_first_cuda_card():
    inventory = _inventory(("CUDA",))
    for row in inventory["capabilities"]:
        if row["component"] == "tts":
            row["physical_id"] = None
    result = plan(inventory, mode="chat_claude", config=_config())["components"]
    assert result["tts"]["device"] == "TTS:CUDA"
    assert result["tts"]["physical_id"] is None
    assert result["stt"]["kind"] == "CPU"


@pytest.mark.parametrize("second_free,expected", [(None, "CUDA:1"), (4500, "CUDA:1"), (3500, "CPU")])
@pytest.mark.parametrize("verified_npu", [False, True])
def test_multiple_cuda_cards_keep_identity_and_separate_reservations(second_free, expected, verified_npu):
    inventory = _inventory(("CUDA", "NPU") if verified_npu else ("CUDA",), cuda_free=None)
    first = next(d for d in inventory["devices"] if d["kind"] == "CUDA")
    # 4500 MiB minus the margin fits STT alone, not STT + TTS; 3500 cannot fit STT.
    inventory["devices"].append({**first, "id": "CUDA:1", "index": 1, "physical_id": "second-uuid",
                                 "memory_free": None if second_free is None else second_free * 2**20})
    # The matrix also contains an unsupported OpenVINO row for CUDA:0.
    row = next(r for r in inventory["capabilities"] if r["component"] == "stt.whisper"
               and r["device"] == "CUDA:0" and r["backend"] == "faster-whisper")
    assert row["supported"] and row["prerequisites"]
    inventory["capabilities"].append({**row, "id": row["id"] + "-second", "device": "CUDA:1"})
    result = plan(inventory, mode="chat_claude", config=_config())["components"]
    assert result["tts"]["device"] == "CUDA:0"
    assert result["stt"]["device"] == ("OV:NPU" if verified_npu else expected)
    assert "CUDA:0" not in result["stt"]["fallbacks"]
    if expected == "CUDA:1":
        cuda = next(d for d in [result["stt"], *result["stt"]["fallback_details"]] if d["device"] == "CUDA:1")
        assert cuda["physical_id"] == "second-uuid"
        assert cuda["index"] == 1
        assert cuda["backend"] == "faster-whisper"
    else:
        assert "CUDA:1" not in result["stt"]["fallbacks"]


def test_incompatible_llm_override_falls_back_without_changing_model():
    config = _config(llm_device="CUDA")
    result = plan(_inventory(), mode="chat_local", config=config)["components"]["llm"]
    assert _chain(result) == ["GPU", "CPU"]
    assert result["model"] == config["llm_model"]
    assert "requested device unavailable" in result["reason"]


@pytest.mark.parametrize("changes", [{"tts_server_command": ["custom-server"]}, {"tts_url": "https://voice.example/tts"}])
def test_external_tts_is_not_repositioned(changes):
    result = plan(_inventory(), mode="chat_claude", config=_config(**changes))["components"]["tts"]
    assert result["state"] == "external"
    assert result["device"] is None and result["reservation_mib"] is None


def test_plan_is_deterministic_and_does_not_mutate_inputs():
    inventory, config = _inventory(), _config()
    loaded = {"stt": {"device": "OV:NPU", "model": "turbo"}}
    before = copy.deepcopy((inventory, config, loaded))
    first = plan(inventory, mode="chat_local", config=config, loaded=loaded, generation=2)
    assert first == plan(inventory, mode="chat_local", config=config, loaded=loaded, generation=2)
    assert (inventory, config, loaded) == before
    assert first["components"]["stt"]["physical_id"] == "NPU-uuid"


@pytest.mark.parametrize("mode", ["dictation", "chat_local", "chat_claude"])
def test_logging_hook_runs_in_background_without_changing_config(monkeypatch, mode):
    from debora_whisper import dictation_engine as engine

    inventory = {**_inventory(), "elapsed_ms": 1.0, "timed_out": False}
    config = _config(voice_chat=mode != "dictation", voice_chat_backend="claude" if mode == "chat_claude" else "local")
    original = copy.deepcopy(config)
    calls, logs = [], []
    def capture(*args, **kwargs):
        calls.append(threading.current_thread())
        return plan(*args, **kwargs)
    monkeypatch.setattr(device_inventory, "probe_inventory", lambda *args: inventory)
    monkeypatch.setattr(device_planner, "plan", capture)
    monkeypatch.setattr(engine, "log", logs.append)
    thread = engine.log_hardware_inventory(config)
    thread.join(5)
    assert not thread.is_alive()
    assert calls == [thread]
    assert config == original
    lines = [line for line in logs if line.startswith("Device plan 1 ")]
    assert len(lines) == 3
    assert all(f"mode={mode}" in line and "confidence=" in line and "reservation=" in line for line in lines)


@pytest.mark.parametrize("failure", ["probe", "planner"])
def test_logging_failure_is_contained(monkeypatch, failure):
    from debora_whisper import dictation_engine as engine

    logs = []
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(engine, "log", logs.append)
    monkeypatch.setattr(device_inventory, "probe_inventory", fail if failure == "probe" else lambda *args: _inventory())
    if failure == "planner":
        monkeypatch.setattr(device_planner, "plan", fail)
    thread = engine.log_hardware_inventory(_config())
    thread.join(5)
    assert not thread.is_alive()
    if failure == "planner":
        assert any("Device plan failed:" in line for line in logs)
    else:
        assert any(line.startswith("Device plan 1 ") and "confidence=unknown" in line for line in logs)
