# Provedores por sistema operacional

Tudo o que depende do sistema operacional passa por `debora_whisper.platforms`.
`platforms.current()` devolve o provedor do sistema em uso: `WindowsPlatform`
hoje e, no futuro, um provedor Linux (Ubuntu). O resto do app não testa
`sys.platform` nem importa `ctypes.windll`, `winreg` ou `keyboard` diretamente.

`Platform` (em `platforms/base.py`) define cada recurso com uma resposta
portátil ou "desconhecido" (`None`). Cada provedor sobrescreve só o que sabe
fazer. Assim, um sistema sem suporte a um recurso continua funcionando, apenas
sem esse recurso.

## Já no provedor

| Recurso | Windows | Linux (futuro) |
|---|---|---|
| Tipo do dispositivo de saída (fone ou alto-falante), usado pelo filtro de eco `"auto"` | Form factor do endpoint no registro (`MMDevices\Audio\Render`) | Tipo da porta no PipeWire/PulseAudio (`pactl`, `headphones`/`headset`) |
| Processos: janela oculta e encerramento da árvore | `CREATE_NO_WINDOW` + `taskkill /T` | `process.kill` / `killpg` em um grupo de processos |
| Ferramentas de GPU: caminho absoluto do `nvidia-smi` e carga das DLLs NVIDIA | `System32\nvidia-smi.exe` + `PATH` e `os.add_dll_directory` | `nvidia-smi` no `PATH` + `LD_LIBRARY_PATH`/preload via `ctypes` |
| Escrita no aplicativo em foco: digitar, apagar e rascunhos em tempo real | `platforms/windows_text.py`: clipboard + `keyboard`, fallback SendInput e comparação de HWNDs | Caminho portátil de clipboard + `keyboard`; alvo e fallback nativo desconhecidos. X11/Wayland precisam de estudo próprio |

## Migração, em PRs pequenos e sem mudar comportamento

| Ordem | Recurso | Onde está hoje |
|---|---|---|
| 1 | Atalhos globais e tecla isolada (Right Alt/AltGr), incluindo a exigência de administrador | `dictation_engine.py` (`import keyboard`), `solo_key.py` |
| 2 | Janela do overlay: DPI, sempre no topo, sem foco, fora da barra de tarefas | `ui/overlay.py` (`GetDpiForWindow`, `WS_EX_*`) |
| 3 | Instalação: atalhos do Menu Iniciar e início automático | `shortcuts.py` (`.lnk` via PowerShell) |
| 4 | Hora do boot, usada para saber se a NPU caiu nesta sessão | `_boot_time` em `dictation_engine.py` |
| 5 | Pastas de usuário e variáveis de ambiente persistentes | `paths.py` (`user_env`) |

Cada etapa move o código para `WindowsPlatform`, deixa em `Platform` o
comportamento atual para outros sistemas (ou `None`) e mantém os testes
existentes passando sem alteração de resultado. O provedor Linux começa
quando houver uma máquina Ubuntu para validar: primeiro processos, GPU e
saída de áudio, que têm equivalentes diretos; a escrita em tempo real e os
atalhos globais dependem de X11 ou Wayland e precisam de estudo próprio.
