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
- **Device fallback** — NPU -> GPU -> CPU when a model fails to load; an NPU DEVICE_LOST moves to the GPU, while a GPU failure stops dictation until the app is restarted

## Requirements

- **Windows 11**
- **Intel Core Ultra** CPU with NPU (Meteor Lake / Lunar Lake / Arrow Lake), or any Intel CPU with iGPU
- **Python 3.10+**
- **16 GB RAM** recommended (NPU/GPU share system memory)

## Quick Start

```powershell
# 1. Clone
git clone https://github.com/Goodsmileduck/npu-whisper.git
cd npu-whisper

# 2. First-time setup (creates venv, installs deps, downloads model, warms NPU cache)
.\Start-Dictation.ps1 -Setup

# 3. Launch (GUI mode with system tray + Dynamic Island overlay)
.\Start-Dictation.ps1
```

Press **Ctrl+Space** to start recording, press again to stop. Transcribed text is pasted at your cursor.

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

Console-only, no GUI. Press Ctrl+Space to toggle recording.

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
  "device": "NPU",
  "model_size": "base",
  "language": "en",
  "hotkey": "ctrl+space",
  "auto_enter": false,
  "beep_on_start": true,
  "max_record_seconds": 60,
  "sample_rate": 16000
}
```

Or change settings from the GUI: right-click the system tray icon and select **Settings**.

Changing the model, device, hotkey, chime, sample rate or maximum recording length rebuilds the engine. That is refused while a recording, transcription or model load is in progress: the Settings window says what is busy, nothing is saved, and you click **Apply** again once it finishes. This keeps two models from running on the same device at once and keeps an in-flight dictation from being dropped. If a load or transcription never finishes (for example a hung driver), quit and restart the app instead.

## File Paths

| Path | Purpose |
|------|---------|
| `~/.npu-dictation/config.json` | User configuration |
| `~/.npu-dictation/models/` | Downloaded model files |
| `~/.npu-dictation/ov-cache/` | OpenVINO compilation cache (do not delete) |
| `~/.npu-dictation/dictation.log` | Runtime log |
| `~/.npu-dictation/venv/` | Python virtual environment |

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
fixed-size graphs (`ParakeetNPU.MEL_BUCKETS` in `dictation_engine.py`) are
compiled and cached up front, and each utterance runs on the smallest bucket
it fits in, instead of always paying for the longest one. Audio longer than
the largest bucket is truncated (never crashes), and this is logged, not
silent. The current bucket sizes are chosen by reasoning about typical
dictation lengths, not by measurement — see the comment above `MEL_BUCKETS`
and `benchmarks/README.md`.

## Benchmarking

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
When the error is attributed to the NPU alone, the GUI falls back to the GPU for the rest of the session. Reboot to reset the NPU.

### GPU failed: restart required
OpenVINO GPU errors such as `CL_OUT_OF_RESOURCES`, or a device loss that cannot be pinned on the NPU (Parakeet runs its decoder on the GPU), can leave the OpenCL context in a state where further calls hang. The app does not retry or reload after that, on the GPU or on any other device. Recording and transcription stay disabled, and Settings changes are saved but not applied (the tray menu marks them "after restart"), until you quit and restart the app. The original OpenVINO error is written to `~/.npu-dictation/dictation.log`. If the failure repeats, select NPU or CPU in Settings, then restart. Disabling the retry only prevents a hang; it does not fix the driver or memory problem behind the error.

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
watchfiles "python app.py" .
```

## License

MIT
