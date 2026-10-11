"""Shared hardware queries. Call these in an isolated process for inventory."""
import csv
import io
import os
import shutil
import subprocess
from pathlib import Path

from debora_whisper.processes import NO_WINDOW


def nvidia_smi():
    """Absolute path, so Windows never runs an nvidia-smi.exe from the
    current folder."""
    system = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
    if os.name == "nt" and system.is_file():
        return str(system)
    found = shutil.which("nvidia-smi")
    if found is None or Path(found).resolve().parent == Path.cwd().resolve():
        return None
    return str(Path(found).resolve())


def nvidia_output(fields=None, *, timeout=None, units=True):
    """Keep the legacy detection command, with optional inventory fields."""
    executable = nvidia_smi()
    if executable is None:
        return None
    args = [executable]
    if fields:
        args += ["--query-gpu=" + fields,
                 "--format=csv,noheader" + ("" if units else ",nounits")]
    return subprocess.check_output(args, stderr=subprocess.DEVNULL,
                                   creationflags=NO_WINDOW, timeout=timeout).decode("utf-8").strip()


def openvino_core():
    import openvino as ov
    return ov.Core()


def _known(value):
    value = str(value).strip()
    return None if not value or value.lower() in {"n/a", "[n/a]", "unknown", "[not supported]"} else value


def _mib(value):
    try:
        return max(0, int(value)) * 1024 * 1024
    except (ValueError, TypeError):
        return None


def nvidia_inventory(timeout):
    """One query, preserving physical UUIDs and indices on multi-GPU hosts."""
    output = nvidia_output("index,uuid,name,driver_version,memory.total,memory.free",
                           timeout=timeout, units=False)
    if output is None:
        return {"status": "unavailable", "devices": []}
    devices = []
    for row in csv.reader(io.StringIO(output), skipinitialspace=True):
        if len(row) != 6:
            continue
        index, uuid, name, driver, total, free = map(str.strip, row)
        if not index.isdigit() or not _known(name):
            continue
        devices.append({"id": "CUDA:" + index, "kind": "CUDA", "index": int(index),
                        "physical_id": _known(uuid), "name": name, "driver": _known(driver),
                        "memory_total": _mib(total), "memory_free": _mib(free),
                        "memory_shared": False})
    return {"status": "ok" if devices else "unknown", "devices": devices}


def openvino_inventory(emit):
    """Publish enumeration before optional properties: even a property can hang."""
    core = openvino_core()
    devices = [{"id": "OV:" + name, "kind": name.split(".")[0], "index": name,
                "physical_id": None, "name": name, "driver": None,
                "memory_total": None, "memory_free": None, "memory_shared": None}
               for name in core.available_devices]
    emit({"status": "ok", "devices": devices})
    for device in devices:
        for prop, key in (("FULL_DEVICE_NAME", "name"), ("DEVICE_UUID", "physical_id"),
                          ("DRIVER_VERSION", "driver")):
            try:
                device[key] = _known(core.get_property(device["index"], prop))
            except Exception:
                pass  # Property availability differs between plugins/drivers.
            emit({"status": "ok", "devices": devices})
    # OpenVINO memory may be shared with system RAM. Do not invent a separate
    # memory pool, or assume an unsupported property means zero bytes.

