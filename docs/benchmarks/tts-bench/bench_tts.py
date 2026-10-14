# /// script
# requires-python = "==3.12.*"
# dependencies = []
# ///
"""Benchmark a running /tts server using only the Python standard library."""
import argparse
import csv
import io
import json
from pathlib import Path
import tempfile
import time
import urllib.error
import urllib.request
import wave


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def output_dir(path):
    target = Path(path).resolve() if path else Path(tempfile.gettempdir()) / "debora-tts-bench"
    target = target.resolve()
    repo = Path(__file__).resolve().parents[3]
    if target.is_relative_to(repo) or any((p / ".git").exists() for p in (target, *target.parents)):
        raise ValueError("--out must be outside a repository; use $env:TEMP\\debora-tts-bench")
    target.mkdir(parents=True, exist_ok=True)
    # A separate run directory avoids overwriting another backend's listening samples.
    return Path(tempfile.mkdtemp(prefix="run-", dir=target))


def synthesize(opener, url, text, voice, timeout):
    body = {"text": text, "language": "pt"}
    if voice:
        body["voice"] = str(Path(voice).resolve())
    request = urllib.request.Request(url.rstrip("/") + "/tts",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with opener.open(request, timeout=timeout) as response:
        wav = response.read()
    elapsed = time.perf_counter() - start
    with wave.open(io.BytesIO(wav), "rb") as audio:
        seconds = audio.getnframes() / audio.getframerate()
    if seconds <= 0:
        raise ValueError("Server returned an empty WAV")
    return wav, {"response_seconds": elapsed, "audio_seconds": seconds, "rtf": elapsed / seconds}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--sentences", type=Path, default=Path(__file__).with_name("frases_ptbr.txt"))
    parser.add_argument("--voice", help="Optional reference WAV; omitted uses the server's startup voice")
    parser.add_argument("--out", help="WAV/CSV/JSON output directory, outside any repository (default: TEMP)")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--warmup", type=int, default=1, help="Unmeasured requests before the list (default: 1)")
    args = parser.parse_args(argv)
    sentences = [line.strip() for line in args.sentences.read_text(encoding="utf-8-sig").splitlines()
                 if line.strip() and not line.startswith("#")]
    if not sentences or any(len(text) > 1000 for text in sentences):
        parser.error("Provide non-empty sentences of at most 1000 characters each")
    if args.warmup < 0 or args.timeout <= 0:
        parser.error("--warmup must be non-negative and --timeout positive")
    try:
        out = output_dir(args.out)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(args.url.rstrip("/") + "/health", timeout=args.timeout) as response:
            health = json.load(response)
        if not health.get("ok"):
            raise ValueError(f"Server not ready: {health}")
        for _ in range(args.warmup):
            synthesize(opener, args.url, sentences[0], args.voice, args.timeout)
        rows = []
        for index, text in enumerate(sentences, 1):
            wav, metrics = synthesize(opener, args.url, text, args.voice, args.timeout)
            filename = f"{index:02d}.wav"
            (out / filename).write_bytes(wav)
            rows.append({"file": filename, "text": text, **metrics})
            print(f"{index:02d}: response={metrics['response_seconds']:.3f}s "
                  f"audio={metrics['audio_seconds']:.3f}s RTF={metrics['rtf']:.3f}")
        summary = {"health": health, "count": len(rows), "warmup": args.warmup,
                   "url": args.url, "voice": args.voice, "metrics": {}}
        for metric in ("response_seconds", "audio_seconds", "rtf"):
            values = [row[metric] for row in rows]
            stats = {"p50": percentile(values, 0.5), "p90": percentile(values, 0.9)}
            summary["metrics"][metric] = stats
            print(f"{metric}: p50={stats['p50']:.3f} p90={stats['p90']:.3f}")
        with (out / "sentences.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Output: {out}")
    except urllib.error.HTTPError as exc:
        parser.exit(1, f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}\n")
    except (OSError, ValueError, wave.Error) as exc:
        parser.exit(1, f"Benchmark failed: {exc}\n")


if __name__ == "__main__":
    main()
