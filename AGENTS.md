# Agent Context for Débora Whisper

## Project Overview
Débora Whisper (`debora-whisper`, formerly `npu-whisper`) is a local voice-to-text dictation engine for Windows, originally built to leverage Intel NPU via OpenVINO. It has since been expanded to support NVIDIA GPUs through `faster-whisper`. It features a desktop overlay (Dynamic Island style), a system tray, and types transcribed text directly into the user's active window.

## Key Architectural Details
- **OS**: Windows 11; requires Administrator privileges for global hotkeys.
- **Naming**: The display name is "Débora Whisper" (with the accent); every identifier, path, package and command uses ASCII (`debora-whisper`, `debora_whisper`, `~/.debora`). See `docs/BRANDING.md`. `paths.py` moves the pre-rename `~/.npu-dictation` and `$MODELS_DIR/npu-whisper` folders on first run; tests set `DEBORA_WHISPER_NO_MIGRATION` so they never touch them.
- **Packaging**: Code lives in the `debora_whisper` package (`app.py`, `dictation_engine.py`, `ui/`). `pyproject.toml` is the single source of dependencies (runtime by default; `cuda`, `export`, `test` extras) and defines the `debora` (tray app) and `debora-cli` commands. Users install with `uv tool install`; tagged `v*` releases publish to PyPI via `.github/workflows/release.yml`. The app must never pip-install at runtime.
- **Hardware Fallback**: Models attempt to load on the requested hardware. If OpenCL/CUDA fails or devices are lost, the engine gracefully falls back (e.g., NPU -> GPU -> CPU).
- **Parakeet Bucketing**: The Parakeet model requires static input shapes for OpenVINO NPU compilation, so it uses pre-compiled shape buckets for its encoder graph. The decoder runs on the GPU or CPU.
- **Logging Subsystem**: Logs are split between `~/.debora/logs/app.log` (startup events, transcription timings, and hardware names) and `~/.debora/logs/telemetry.log` (background stats like CPU, RAM, VRAM or NPU memory, and audio buffer health).
- **NVIDIA GPU Integration**: The project uses a CPU-only PyTorch installation to save disk space. To support `faster-whisper` on CUDA, it dynamically loads NVIDIA DLLs installed via PyPI (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`). NVIDIA cards are detected via a subprocess call to `nvidia-smi`, bypassing `torch.cuda.is_available()`.

## Token-Efficient Workflow
- Start with targeted `rg` searches and read only the relevant functions or line ranges. Avoid dumping entire files, workflows, logs, or test collections when a focused query answers the question.
- Keep track of relevant files, symbols, and findings. Re-read only changed sections or information needed to resolve a new uncertainty.
- Batch independent searches and keep tool output concise. Request summaries, failure details, or filtered results instead of large outputs that will be truncated.
- Finish the intended code and test edits before starting the full test suite. Do not edit source or test files while tests are running; this can invalidate results and cause source-inspection failures.
- Run focused tests for the affected behavior, then the full suite when appropriate or required. Repeat checks only after relevant changes, failures, or unresolved concerns; avoid redundant builds and test runs.
- For long-running tests and CI, use completion notifications or reasonably spaced status checks. Keep progress updates useful without repeatedly querying unchanged status.
- Preserve required validation and correctness. Reduce redundant context and operations rather than skipping necessary investigation or checks.

## Agent Pipeline (Codex implements, Claude Code reviews)
- Run each task with `python scripts/agents/task.py <name> <branch> --prompt P.md --title T.txt --body B.md --commit-msg M.txt`. It owns the whole loop (Codex implementation, full suite, commit/push/PR, headless `@rev` + `@sec` reviews posted on the PR, Codex fix rounds) and prints a few status lines plus one `RESULT` line. The exit code says what happened; see the script's docstring.
- Reviews default to Claude Code headless (`claude -p --agent marc:review|marc:security`, read-only tools). Use `--reviewer codex` only when Claude is unavailable.
- Prompts, outputs and full reports live in the run folder (`../dw-runs` by default). Read a specific file when a result needs it; never paste whole reports into the operator's context.
- Measure with `python scripts/agents/usage.py` (per task) or `--codex-history <worktree>...` (tasks run before the ledger).

## Tech-Lead Operator Sessions
- **One session per work item or small batch.** The queue and its state live in the issue/board or the queue file, not in the conversation. Start a new session when the item ships, or once the context passes ~100k tokens: every tool call re-reads the whole context, and a six-day operator session cost more than all the dispatched work.
- **The operator does not do engineering work.** Code investigation, edits, test runs and log digging go to the Codex task or a specialist; the operator writes specs, reads `RESULT` lines and verdicts, and decides. Do not search old session transcripts from the operator session.
- **Wait without polling.** Run `task.py` in the background and act on its completion notification. No `sleep`/`tail` loops or cron checks on a running task.
