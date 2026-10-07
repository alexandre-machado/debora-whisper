# Agent Context for NPU Whisper

## Project Overview
`npu-whisper` is a local voice-to-text dictation engine for Windows, originally built to leverage Intel NPU via OpenVINO. It has since been expanded to support NVIDIA GPUs through `faster-whisper`. It features a desktop overlay (Dynamic Island style), a system tray, and types transcribed text directly into the user's active window.

## Technology Stack
- **OS**: Windows 11 (requires Administrator privileges for global hotkeys)
- **Language**: Python 3.10+
- **Inference Backends**:
  - `openvino` / `openvino-genai`: Used for Whisper and Parakeet models on Intel NPU, Intel iGPU, and CPU.
  - `faster-whisper`: Used for Whisper models on NVIDIA GPUs (via CUDA).
- **Audio Capture**: `sounddevice`
- **UI**: Desktop overlay / system tray.

## Key Architectural Details
- **Packaging**: Code lives in the `npu_whisper` package (`app.py`, `dictation_engine.py`, `ui/`). `pyproject.toml` is the single source of dependencies (runtime by default; `cuda`, `export`, `test` extras) and defines the `npu-whisper` (tray app) and `npu-whisper-cli` commands. Users install with `uv tool install`; tagged `v*` releases publish to PyPI via `.github/workflows/release.yml`. The app must never pip-install at runtime.
- **Hardware Fallback**: Models attempt to load on the requested hardware. If OpenCL/CUDA fails or devices are lost, the engine gracefully falls back (e.g., NPU -> GPU -> CPU).
- **Parakeet Bucketing**: The Parakeet model requires static input shapes for OpenVINO NPU compilation, so it uses pre-compiled shape buckets for its encoder graph. The decoder runs on the GPU or CPU.
- **Logging Subsystem**: Logs are split between `~/.npu-dictation/logs/app.log` (startup events, transcription timings, and hardware names) and `~/.npu-dictation/logs/telemetry.log` (background stats like CPU, RAM, VRAM, and audio buffer health).
- **NVIDIA GPU Integration**: The project uses a CPU-only PyTorch installation to save disk space. To support `faster-whisper` on CUDA, it dynamically loads NVIDIA DLLs installed via PyPI (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`). NVIDIA cards are detected via a subprocess call to `nvidia-smi`, bypassing `torch.cuda.is_available()`.

## Recent Agent Modifications
- Added NVIDIA GPU (RTX) support using `faster-whisper`.
- Fixed symlink issues with HuggingFace Hub on Windows.
- Implemented robust hardware identification logging using `nvidia-smi` and OpenVINO properties so the logs state the exact GPU/NPU model name (e.g., "NVIDIA GeForce RTX 4070").
- Separated telemetry and application logs.
- Implemented real-time resource tracking (CPU, System RAM, App RAM, and VRAM) in the telemetry loop.
