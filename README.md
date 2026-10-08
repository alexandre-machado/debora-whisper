# NPU Whisper

Local voice-to-text dictation for Windows, powered by Intel NPU via OpenVINO. Press a hotkey, speak, and text appears at your cursor. Zero cloud, zero cost, zero data leaving your machine.

![Windows](https://img.shields.io/badge/platform-Windows%2011-0078D4?logo=windows)
![Python](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **NPU-accelerated** — runs on Intel AI Boost (Meteor Lake / Lunar Lake / Arrow Lake)
- **6 models** — Whisper tiny/base/small/medium/turbo + Parakeet TDT 0.6B (best accuracy)
- **15 languages** — English, Russian, Spanish, French, German, Japanese, Chinese, Korean, and more
- **Dynamic Island overlay** — always-visible floating pill with waveform, timer, and status
- **System tray** — lives in your taskbar, right-click for settings/history
- **One-click settings** — language-first model picker with download status badges
- **Claude Code mode** — auto-paste + Enter for hands-free prompt submission
- **Audio chimes** — pleasant start/stop feedback
- **Device priority** — `device_priority` in the config (default RTX -> NPU -> iGPU -> CPU) picks the startup device; a DEVICE_LOST moves to the next healthy device on the list

## Requirements

- **Windows 11**
- **Intel Core Ultra** CPU with NPU (Meteor Lake / Lunar Lake / Arrow Lake), or any Intel CPU with iGPU
- **Python 3.10+**
- **16 GB RAM** recommended (NPU/GPU share system memory)

## Quick Start

### Install as a command (uv)

[uv](https://docs.astral.sh/uv/) installs the app in its own environment and
fetches a matching Python if you have none:

```powershell
winget install astral-sh.uv

# NPU / Intel iGPU / CPU only
uv tool install git+https://github.com/alexandre-machado/npu-whisper

# ...or with the NVIDIA RTX backend (adds ~2 GB of CUDA libraries)
uv tool install "npu-whisper[cuda] @ git+https://github.com/alexandre-machado/npu-whisper"

npu-whisper-cli --setup   # detect devices, download the model, warm the cache
npu-whisper               # tray app + overlay
npu-whisper-cli           # console-only mode
```

To launch it from the Start Menu (no console window), optionally also at sign-in:

```powershell
npu-whisper --install-shortcut              # Start Menu entry "NPU Whisper"
npu-whisper --install-shortcut --autostart  # ...and start with Windows
```

Only one tray app runs at a time; launching a second one exits quietly. The
shortcut points at the Python of the current install: after reinstalling
(`uv tool install --reinstall`, a new Python), run `--install-shortcut` again.

Update with `uv tool upgrade npu-whisper`. To remove, run
`npu-whisper --remove-shortcut` first, then `uv tool uninstall npu-whisper`
(models and config stay in `~/.npu-dictation/`). Once published to PyPI, the
same works with `uv tool install npu-whisper` or a one-off `uvx npu-whisper`.

### From a source checkout

```powershell
# 1. Clone
git clone https://github.com/Goodsmileduck/npu-whisper.git
cd npu-whisper

# 2. First-time setup (creates venv, installs deps, downloads model, warms NPU cache)
.\Start-Dictation.ps1 -Setup

# 3. Launch (GUI mode with system tray + Dynamic Island overlay)
.\Start-Dictation.ps1
```

**Ctrl+Space** works two ways:

- **Hold** it while you speak (push-to-talk): recording stops when you let go
  and the whole utterance is transcribed and pasted at your cursor.
- **Tap** it to start continuous listening: voice activity detection types each
  sentence as you finish it (a rising two-note chime confirms). Tap again to
  stop; a sentence in progress is still typed.

Set `"tap_action": "toggle"` to make a tap start a recording that the next tap
stops instead.

## Models

| Model | Params | Device | Speed (3s audio) | WER | Languages | Notes |
|-------|--------|--------|-------------------|-----|-----------|-------|
| **tiny** | 39M | NPU | — | — | All 15 | Lowest resource usage; useful for quick commands and CPU/NPU fallback |
| **base** | 74M | NPU | 0.2s | 5.0% | All 15 | Default, great for short commands |
| **small** | 244M | NPU | 0.6s | 3.4% | All 15 | Balanced speed/accuracy |
| **parakeet** | 600M | NPU+GPU | 0.2s | 3.7% | en, es, fr, de, it, nl, pl, pt, ru, uk (10 of 15) | Fast hybrid pipeline; validate Brazilian Portuguese separately |
| **medium** | 769M | GPU | — | 2.9% | All 15 | High accuracy, slower |
| **turbo** | 809M | GPU | 2.4s | 2.3% | All 15 | Best multilingual quality |

> **WER** = Word Error Rate (lower is better), publisher-reported, not independently reproduced here. Whisper WER on LibriSpeech test-clean from [HuggingFace model cards](https://huggingface.co/openai/whisper-base). Parakeet WER (3.7% LibriSpeech test-clean, 17.0% FLEURS multilingual) from the [NVIDIA model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).

Parakeet's upstream checkpoint (`nvidia/parakeet-tdt-0.6b-v3`) supports 25 European languages with automatic language identification (no language token needed at inference). This app exposes the intersection of that list with its own 15-language picker: English, Spanish, French, German, Italian, Dutch, Polish, Portuguese, Russian, Ukrainian. It does not cover Japanese, Chinese, Korean, Turkish, or Arabic, which stay Whisper-only.

For long-form audio, prefer Whisper turbo/medium on GPU or Whisper small on
NPU until Parakeet chunking is implemented. The current Parakeet integration
uses static encoder buckets up to approximately 16 seconds; longer inputs are
truncated and logged rather than transcribed in full. The published Parakeet
results also should not be assumed to represent Brazilian Portuguese, because
the upstream evaluation notes distinguish European Portuguese.

The Settings dialog shows which models are already downloaded and filters them by your selected language (e.g., selecting Japanese hides Parakeet, since the checkpoint wasn't trained on it; selecting Russian now shows Parakeet).

## Usage

### GUI Mode (default)

```powershell
.\Start-Dictation.ps1
```

Launches with a Dynamic Island overlay at the top of your screen and a system tray icon. Click the green dot or press Ctrl+Space to record.

### CLI Mode

```powershell
.\Start-Dictation.ps1 -CLI
```

Console-only, no GUI. Same hotkey gestures as the GUI.

### Claude Code Mode

```powershell
.\Start-Dictation.ps1 -AutoEnter
```

Pastes text and presses Enter automatically — speak your prompt and it submits to Claude Code.

### Override Settings

```powershell
.\Start-Dictation.ps1 -Device GPU -Model turbo -Language ru
```

## Configuration

Stored at `~/.npu-dictation/config.json`:

```json
{
  "device_priority": ["CUDA", "NPU", "GPU", "CPU"],
  "model_size": "base",
  "language": "en",
  "hotkey": "ctrl+space",
  "tap_action": "continuous",
  "auto_enter": false,
  "beep_on_start": true,
  "max_record_seconds": 60,
  "sample_rate": 16000
}
```

`device_priority` is the order devices are tried in. At startup the app uses
the first one present on the machine that can run the model (`CUDA` = NVIDIA
RTX via faster-whisper, Whisper models only; `GPU` = Intel iGPU). When the
active device is lost, it falls back to the next healthy one. `-Device` on the
command line overrides the list for that run. `device` is written by the app
and records the device last chosen.

`tap_action` is what a short hotkey press does: `continuous` (default) starts
continuous listening, `toggle` starts a recording that the next press stops.
Holding the hotkey is always push-to-talk. `--continuous` starts the app
already listening continuously. A session started by a tap stops by itself
after `continuous_idle_stop_seconds` (default 120) without speech; `null`
keeps it on until the next tap.

### Voice chat (OpenVINO LLM + Chatterbox)

With `"voice_chat": true` (or `--voice-chat`, or **Voice chat** in Settings),
nothing is typed: each final transcription goes to a local LLM, and its reply
is spoken by a local TTS server and shown in the overlay. The reply streams and
plays sentence by sentence. The conversation keeps the last 8 turns and starts
over after 10 minutes of silence. In continuous listening the microphone is
muted while the reply plays; a hotkey press cuts the reply short. A sentence
ends after 0.8 s of silence (`voice_chat_end_silence_seconds`; dictation
waits 1.5 s), and the reply's sentences play back to back on one audio stream.

Switch it on or off while the app runs, from **Voice chat** in the tray menu
or in Settings: no restart. Switching on starts the TTS server and loads the
LLM in the background (the overlay says "Voice chat ready"); a sentence spoken
meanwhile waits for it. Switching off cuts the reply in progress; both stay
loaded until the app exits, so switching back is instant.

```json
{
  "voice_chat": true,
  "llm_model": "OpenVINO/Qwen3-8B-int4-cw-ov",
  "llm_device": "GPU",
  "llm_prompt": null,
  "tts_voice": null,
  "tts_url": "http://127.0.0.1:8765",
  "tts_timeout_seconds": 60,
  "tts_server_command": null
}
```

The LLM runs with OpenVINO GenAI, the same runtime as Whisper, in a process
of its own (`npu_whisper/llm_server.py`, started by the app with its own
Python: nothing to install). Loading it holds Python's GIL for the whole
compile, which froze the app when it ran inside it. While it compiles on the
iGPU no transcription runs: both times the two overlapped, the NPU was lost.
A load that takes over 10 minutes is killed. It logs to
`~/.npu-dictation/logs/llm_server.log`. `llm_model` is a Hugging Face repo with an OpenVINO
export (downloaded on first use into the Hugging Face cache) or a local
directory. On a Core Ultra 9 185H, the default answers in under a second at
~15 tokens/s on the Arc iGPU (`"llm_device": "GPU"`); if the device fails it
loads on the CPU. Qwen3's thinking is turned off. `llm_prompt: null` uses the
built-in voice-assistant prompt.

The TTS is Chatterbox Multilingual on an NVIDIA GPU, served by
`npu_whisper/tts_server.py`. It needs torch 2.6 with CUDA, so it never runs in
the app's environment: the script declares its own dependencies and the app
starts it with `uv run --script` (uv builds that environment on the first run,
a few GB, and caches it). It is started only when nothing answers at
`tts_url`, stopped when the app exits, and logs to
`~/.npu-dictation/logs/tts_server.log`. `tts_server_command` replaces that
command line (a list of arguments).

`tts_voice` clones a voice from ~10 s of clean speech: a WAV path, or a name
looked up as `<name>.wav` in the voices folder (see File Paths). `null` uses
Chatterbox's own voice.

Where each part runs best on a Core Ultra laptop with an 8 GB RTX: Whisper on
the NPU (`--device NPU`), the LLM on the Arc iGPU, Chatterbox alone on the RTX.
Whisper on CUDA next to Chatterbox can run out of video memory. If the TTS
server is down, the reply is only shown. `tts_url` receives everything the
LLM says; keep it pointed at a server you trust.

Or change settings from the GUI: right-click the system tray icon and select **Settings**.

Changing the model, device, hotkey, chime, sample rate or maximum recording length rebuilds the engine. That is refused while a recording, transcription or model load is in progress: the Settings window says what is busy, nothing is saved, and you click **Apply** again once it finishes. This keeps two models from running on the same device at once and keeps an in-flight dictation from being dropped. If a load or transcription never finishes (for example a hung driver), quit and restart the app instead.

## File Paths

| Path | Purpose |
|------|---------|
| `~/.npu-dictation/config.json` | User configuration |
| `~/.npu-dictation/logs/` | `app.log`, `telemetry.log`, `tts_server.log`, `llm_server.log`; the first three with `[YYYY-MM-DD HH:MM:SS] message` lines; at startup a log over 5 MB moves to `<name>.1` |
| `~/.npu-dictation/npu_lost.json` | An NPU lost to DEVICE_LOST: startup uses the next device and probes the NPU in the background (cleared when a probe passes or Windows restarts) |
| `~/.npu-dictation/models/` | Downloaded model files |
| `~/.npu-dictation/ov-cache/` | OpenVINO compilation cache (do not delete) |
| `~/.npu-dictation/voices/` | Voices for `tts_voice` by name |

With the `MODELS_DIR` environment variable set (a shared models folder), the
models and cache move to `$MODELS_DIR/npu-whisper/` and the voices to
`$MODELS_DIR/voices/`. Models taken from the Hugging Face cache (the voice
chat LLM, faster-whisper on CUDA, Chatterbox) follow `HF_HOME`.

## How It Works

```
+-------------------+     +------------------+     +------------------+
|  Ctrl+Space       | --> |  Microphone      | --> |  Whisper or      |
|  (global hotkey)  |     |  16kHz mono      |     |  Parakeet model  |
+-------------------+     +------------------+     +--------+---------+
                                                            |
                          +------------------+              |
                          |  Clipboard paste | <------------+
                          |  into any window |
                          +------------------+
```

The model runs **100% locally** on your Intel NPU. No internet required after initial model download.

**Whisper** models use OpenVINO GenAI's `WhisperPipeline` with INT8 quantization on NPU or GPU.

**Parakeet TDT** uses a three-stage hybrid pipeline for best accuracy:

```
+---------------+     +----------------+     +----------------+
|  nemo128.onnx | --> |  Encoder       | --> |  TDT Decoder   |
|  Mel spectro  |     |  OpenVINO NPU  |     |  OpenVINO GPU  |
|  (CPU)        |     |                |     |  (CPU fallback)|
+---------------+     +----------------+     +----------------+
```

NPU only supports static (fixed) input shapes, so the Parakeet encoder can't
just size itself to each utterance. Instead it's **shape-bucketed**: several
fixed-size graphs (`ParakeetNPU.MEL_BUCKETS` in `npu_whisper/dictation_engine.py`) are
compiled and cached up front, and each utterance runs on the smallest bucket
it fits in, instead of always paying for the longest one. Audio longer than
the largest bucket is truncated (never crashes), and this is logged, not
silent. The current bucket sizes are chosen by reasoning about typical
dictation lengths, not by measurement — see the comment above `MEL_BUCKETS`
and `benchmarks/README.md`.

## Benchmarking

### Benchmark: turbo on NPU vs Intel iGPU vs RTX (2026-10-06)

Whisper turbo on one laptop: Intel Core Ultra (AI Boost NPU, driver
32.0.100.5540; Arc iGPU) and an NVIDIA RTX 4070 Laptop. OpenVINO 2026.4.1 with
`OpenVINO/whisper-large-v3-turbo-int8-ov` on NPU/iGPU; faster-whisper 1.2.1
(`int8_float16`) on the RTX. Same English speech clips on every device, app
closed, one process per device, median of 5 calls after a warm-up.

| Clip | NPU | Intel iGPU | RTX 4070 |
|------|-----|------------|----------|
| 2s | 1.56s | 0.98s | **0.20s** |
| 5s | 1.59s | 1.03s | **0.21s** |
| 10s | 1.80s | 1.16s | **0.27s** |
| 20s | 2.03s | 1.30s | **0.35s** |
| 26s | 2.13s | 1.60s | **0.39s** |
| 30s, dense speech | 2.79s | 1.92s | **0.56s** |
| Model load (cached) | 3.1s | 1.3s | 3.8s |
| CPU time per call | **0.05–0.23s** | ≈ latency | ≈ latency |

- The RTX is 5–7x faster than the NPU and ~4x faster than the iGPU, which is
  why `device_priority` puts `CUDA` first.
- The NPU takes ~1.5–2.8s per call almost regardless of clip length, but
  leaves the CPU idle; the iGPU and RTX keep a CPU core busy waiting on the
  device for the whole call.
- The RTX draws ~54 W while transcribing and ~3.4 W idle with the model
  loaded. Battery impact was not measured (the test machine's battery is dead).
- The dense 30s clip took 3.0s on the RTX before faster-whisper's temperature
  fallback was disabled: its compression ratio (2.53) passed the 2.4
  threshold, so it re-decoded up to 5 times and returned a temperature-1.0
  sample. The app now decodes greedily at temperature 0, like the OpenVINO path.
- All devices produced the same text.

Reproduce with `benchmarks/bench_devices.py` (usage in its docstring).

### Parakeet

`benchmarks/bench_parakeet.py` measures Parakeet pipeline load/compile time
and per-utterance latency by stage (mel / encoder / decoder), and reports
which shape bucket was selected. See `benchmarks/README.md` for usage and
for the exact commands to produce before/after numbers on real NPU hardware.

## Troubleshooting

### NPU device not found
- Check Device Manager for "Intel(R) AI Boost" or "Intel(R) NPU Accelerator"
- Install/update NPU driver from Windows Update or [Intel's site](https://www.intel.com/content/www/us/en/download/794734/)
- Minimum driver version: 32.0.100.3104

### Slow first run
OpenVINO compiles the model graph for your specific NPU on first launch. This takes 1-15 minutes depending on model size and is cached for subsequent runs. For Parakeet specifically, this means 4 sequential encoder-graph compiles (one per shape bucket), not 1 — see the architecture notes above and `benchmarks/README.md` for details — so first launch takes proportionally longer than a single-graph model.

### DEVICE_LOST error
When the error is attributed to one device, the GUI falls back to the next healthy device in `device_priority`. After an NPU loss it probes the NPU in the background (after 30 s, 1 min, 5 min and 15 min) and moves back once it works; a reboot also resets it. Each probe runs in a separate process, killed after 5 minutes: right after a loss, loading a model on the NPU can hang in the driver or recompile for ~3 minutes, and either one inside the app froze it, Ctrl+C included. Later runs also start on the next device and probe the NPU the same way, until a probe passes.

### GPU failed: restart required
OpenVINO GPU errors such as `CL_OUT_OF_RESOURCES`, or a device loss that cannot be pinned on the NPU (Parakeet runs its decoder on the GPU, and a loss during its GPU fallback compile is blamed on the GPU even if the NPU failed first), can leave the OpenCL context in a state where further calls hang. The app does not retry or reload after that, on the GPU or on any other device. Recording and transcription stay disabled, and Settings changes are saved but not applied (the tray menu marks them "after restart"), until you quit and restart the app. The original OpenVINO error is written to `~/.npu-dictation/logs/app.log`. If the failure repeats, select NPU or CPU in Settings, then restart. Disabling the retry only prevents a hang; it does not fix the driver or memory problem behind the error.

### Hotkey doesn't work
- PowerShell must run as **Administrator** (the `keyboard` library requires elevated privileges)
- Check no other app is capturing the same hotkey
- Try a different hotkey: `.\Start-Dictation.ps1 -Hotkey "ctrl+shift+space"`

### Audio not recording
- Check Windows Settings -> Privacy -> Microphone permissions
- Ensure your mic is the default recording device
- Select a specific mic in the Settings dialog

## Development

The [Tests workflow](.github/workflows/tests.yml) runs the full unit-test suite
on Windows with Python 3.10, 3.12 and 3.14 for pushes, pull requests and manual
runs. Each job uploads a JUnit report, including when tests fail. Tests use
simulated audio devices and inference backends; hardware latency still requires
the [manual audio checks](docs/AUDIO_LATENCY.md).

```powershell
# Install and run the same test dependencies as CI (no model downloads)
python -m pip install -r requirements-test.txt
python -m pytest tests/ -q

# Auto-reload during development
pip install watchfiles
watchfiles "python -m npu_whisper" npu_whisper
```

Code lives in the `npu_whisper` package: `app.py` (tray app, `npu-whisper`),
`dictation_engine.py` (engine and console mode, `npu-whisper-cli`) and `ui/`.
Dependencies are declared in `pyproject.toml`: runtime by default, plus the
`cuda` (RTX), `export` (custom model export) and `test` extras.

### Releasing

Bump `__version__` in `npu_whisper/__init__.py`, then push a `vX.Y.Z` tag that
matches it. The [Release workflow](.github/workflows/release.yml) builds the
wheel and sdist, installs the wheel with `uv tool install` as a smoke test and
publishes to PyPI through trusted publishing (the `pypi` environment must be
registered as a trusted publisher on the PyPI project).

## License

MIT
