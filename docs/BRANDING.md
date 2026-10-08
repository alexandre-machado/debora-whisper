# Branding: Débora Whisper

Decided 2026-10-08. The project is being renamed from `npu-whisper` to
**Débora Whisper**, dropping the NPU reference now that it also runs on NVIDIA
GPUs (faster-whisper), iGPU and CPU.

## Name rationale

- **Débora** (Portuguese spelling, with the accent) comes from the Hebrew
  *Devorah*, meaning **"bee"**. A bee buzzing softly at your ear fits a
  dictation app called "Whisper".
- The biblical Deborah was a prophetess and judge remembered for the *Song of
  Deborah*: a figure known for her **voice**, which is what this project is about.
- A personal name gives the app a personality (in the line of Alexa and Siri).
  The Brazilian spelling sets it apart from existing English-named tools.
- **Whisper** works as a verb ("sussurrar"), not only as a reference to the
  OpenAI model. The app also runs Parakeet, so the name doesn't promise a
  single model.

## Where the accent goes (and where it doesn't)

| Surface | Form |
|---|---|
| Display name (README title, tray tooltip, overlay, About window, artwork) | **Débora Whisper** |
| PyPI distribution | `debora-whisper` (PEP 508 names must be ASCII) |
| Python package | `debora_whisper` |
| CLI commands | `debora` (tray app) and `debora-cli` (console) |
| Data directory | `~/.debora/` (migrated from `~/.npu-dictation/`) |
| GitHub repo | `debora-whisper` |

Never use the accented form in identifiers, file paths, package names or
commands: it causes encoding issues on Windows, terminals and `uv tool install`.

PyPI availability checked 2026-10-08: `debora-whisper`, `debora`,
`deborah-whisper` and `debora-dictation` were all free.

## Visual direction (for the artwork)

- **Mascot / icon: a bee.** It ties directly to the meaning of the name.
- The icon has to read at tray size (16–32 px) and at the overlay's
  "Dynamic Island" scale. Keep the silhouette simple and recognizable in
  monochrome.
- Tone: friendly and quiet. The app *whispers*, so avoid loud or aggressive
  imagery.
- Wordmark: "Débora Whisper" with the accent kept visible, since it is part of
  the identity.

## Rename checklist

- [x] Package `npu_whisper` → `debora_whisper`, `pyproject.toml` name and entry points (`debora`, `debora-cli`)
- [x] Data dir migration `~/.npu-dictation` → `~/.debora` and `$MODELS_DIR/npu-whisper` → `$MODELS_DIR/debora-whisper` (`paths.py`, `Start-Dictation.ps1`)
- [x] Pre-rename "NPU Whisper" Start Menu/Startup shortcuts replaced by `--install-shortcut`
- [x] Display strings: tray, overlay, settings, history, onboarding, README, AGENTS.md
- [x] Workflows smoke-test `debora` / `debora-cli`
- [ ] Rename the GitHub repo to `debora-whisper` (README and pyproject URLs already point there)
- [ ] Add a PyPI trusted publisher for the new `debora-whisper` project before the first `v*` tag
- `npu-whisper` was never published on PyPI, so no redirect release is needed.
