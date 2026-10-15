"""Byte progress shared by telemetry and the standalone TTS scripts."""
from contextlib import contextmanager
import math
import threading


class DownloadProgress:
    """Aggregate discovered files, retaining completed bytes until reset."""

    def __init__(self):
        self._lock = threading.RLock()
        self._files = {}

    def update(self, key, done, total, *, closed=False):
        with self._lock:
            self._files[key] = (min(max(done, 0), total), total, closed)
            return self._totals()

    def _totals(self):
        return {"done": sum(f[0] for f in self._files.values()),
                "total": sum(f[1] for f in self._files.values())}

    def _snapshot(self):
        if not any(not closed and done < total for done, total, closed in self._files.values()):
            return None
        return self._totals()

    def snapshot(self):
        with self._lock:
            return self._snapshot()

    def clear(self):
        with self._lock:
            self._files.clear()


def format_download(label, progress):
    if not isinstance(progress, dict):
        return ""
    done, total = progress.get("done"), progress.get("total")
    if (not isinstance(done, (int, float)) or not isinstance(total, (int, float))
            or not math.isfinite(done) or not math.isfinite(total)
            or total <= 0 or done < 0 or done >= total):
        return ""
    return f" DL {label} {min(99, round(done * 100 / total))}% ({done / 1_000_000:.0f}/{total / 1_000_000:.0f} MB)"


STT_DOWNLOAD = DownloadProgress()


@contextmanager
def track_hf_downloads(tracker, callback=None):
    """Observe HF byte bars, including HTTP/Xet and resumed downloads.

    Patch methods on HF's shared tqdm class so aliases already imported by
    download helpers are covered too. File-count and model-loading bars are
    ignored. Counters remain valid when tqdm display is disabled.
    """
    from huggingface_hub.utils import tqdm

    original_init, original_update, original_close = tqdm.__init__, tqdm.update, tqdm.close
    bars = {}
    lock = threading.RLock()
    tracker.clear()

    def publish(bar, amount=0, closed=False):
        with lock:
            entry = bars.get(id(bar))
            if entry is None:
                return
            entry[0] += amount
            progress = tracker.update(id(bar), *entry, closed=closed)
            if callback and not closed:
                callback(progress["done"], progress["total"])

    def init(bar, *args, **kwargs):
        original_init(bar, *args, **kwargs)
        total = kwargs.get("total")
        if kwargs.get("unit") == "B" and total and total > 0:
            with lock:
                # Retain the bar to prevent object-id reuse between files.
                bars[id(bar)] = [kwargs.get("initial", 0), total]
                retained.append(bar)
                publish(bar)

    def update(bar, n=1):
        result = original_update(bar, n)
        publish(bar, n)
        return result

    def close(bar):
        publish(bar, closed=True)
        return original_close(bar)

    retained = []
    tqdm.__init__, tqdm.update, tqdm.close = init, update, close
    try:
        yield
    finally:
        tqdm.__init__, tqdm.update, tqdm.close = original_init, original_update, original_close
        tracker.clear()
