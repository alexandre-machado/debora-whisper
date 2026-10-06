"""Dynamic icon generation for system tray states using Pillow."""

import math
import random
from PIL import Image, ImageDraw

ICON_SIZE = 64

# Cores solicitadas
C_ERROR = "#EF4444"      # Vermelho
C_READY = "#22C55E"      # Verde
C_RECORDING_IDLE = "#7E22CE"   # Roxo mais escuro (Ocioso/Escutando sem áudio)
C_RECORDING_ACTIVE = "#A855F7" # Roxo vibrante (Escutando com áudio passando)
C_PROCESSING = "#8B5CF6" # Roxo (animado)
C_LOADING = "#808080"    # Cinza (animado)

def render_bars(color, heights, size=ICON_SIZE):
    """Renderiza um ícone com 5 barras verticais de tamanhos variáveis."""
    s = 3
    ss = size * s
    img = Image.new("RGBA", (ss, ss), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    num_bars = 5
    bar_width = int(ss * 0.12)  # ~7-8px for 64px
    spacing = int(ss * 0.06)    # ~3-4px

    total_width = (num_bars * bar_width) + ((num_bars - 1) * spacing)
    start_x = (ss - total_width) // 2

    for i, h_pct in enumerate(heights):
        # Max height is 80% of the icon size
        max_h = ss * 0.8
        bar_h = int(max_h * h_pct)
        # Minimum height so it's always visible
        bar_h = max(bar_h, int(ss * 0.15))

        x0 = start_x + i * (bar_width + spacing)
        x1 = x0 + bar_width
        y0 = (ss - bar_h) // 2
        y1 = y0 + bar_h

        draw.rounded_rectangle([x0, y0, x1, y1], radius=bar_width//2, fill=color)

    return img.resize((size, size), Image.LANCZOS)

# ---- Pré-geração de frames para animação ----

# Transcrevendo (Processing): picos aleatórios simulando fala
_PROCESSING_FRAMES = []
rng = random.Random(42) # Seed fixa para manter testes consistentes
for _ in range(8):
    h = [rng.uniform(0.2, 0.9) for _ in range(5)]
    _PROCESSING_FRAMES.append(render_bars(C_PROCESSING, h))

# Carregando (Loading): onda senoidal
_LOADING_FRAMES = []
for i in range(8):
    offset = i * (math.pi / 4)
    h = [0.5 + 0.35 * math.sin(offset + (j * math.pi / 2.5)) for j in range(5)]
    _LOADING_FRAMES.append(render_bars(C_LOADING, h))

# Matriz de volume dinâmico (pré-gerada para poupar CPU)
# 11 níveis de volume (0.0 a 1.0)
_VOLUME_MATRIX = []
for vol_idx in range(11):
    vol = vol_idx / 10.0
    frames = []
    # Cria 4 variações de barras para cada nível de volume (para dar variação natural)
    for _ in range(4):
        # Base mais suave e picos seguindo o volume
        h = [max(0.15, rng.uniform(vol * 0.3, vol)) for _ in range(5)]
        frames.append(render_bars(C_RECORDING_ACTIVE, h))
    _VOLUME_MATRIX.append(frames)

# Ícones estáticos
_ICON_ERROR = render_bars(C_ERROR, [0.2, 0.2, 0.2, 0.2, 0.2])
_ICON_READY = render_bars(C_READY, [0.3, 0.5, 0.8, 0.5, 0.3])
_ICON_RECORDING_IDLE = render_bars(C_RECORDING_IDLE, [0.15, 0.2, 0.15, 0.2, 0.15])

_vol_frame_counter = 0

def get_volume_icon(level: float) -> Image.Image:
    """Retorna um ícone de gravação baseado no nível de áudio atual."""
    global _vol_frame_counter
    _vol_frame_counter += 1
    
    if level < 0.03:
        # Se estiver muito baixo, exibe estado ocioso (roxo escuro, baixo)
        return _ICON_RECORDING_IDLE
        
    # Clampa o nível de volume
    level = max(0.0, min(1.0, level))
    
    # Adiciona um "boost" visual para volumes normais parecerem vivos no tray
    boosted_level = min(1.0, level * 1.5 + 0.2)
    idx = int(boosted_level * 10)
    
    return _VOLUME_MATRIX[idx][_vol_frame_counter % 4]

def get_icon(state: str, frame: int = 0) -> Image.Image:
    """Retorna o frame correspondente ao estado atual."""
    if state == "loading":
        return _LOADING_FRAMES[frame % len(_LOADING_FRAMES)]
    elif state == "processing":
        return _PROCESSING_FRAMES[frame % len(_PROCESSING_FRAMES)]
    elif state == "recording":
        return _ICON_RECORDING_IDLE
    elif state == "error":
        return _ICON_ERROR
    # default to ready
    return _ICON_READY


# ---- Funções de compatibilidade para os testes e chamadas legacy ----

def render_app_icon(size=64, color=C_READY):
    """Fallback compatibility function used by app.py window icon."""
    return render_bars(color, [0.3, 0.5, 0.8, 0.5, 0.3], size=size)

def icon_loading() -> Image.Image: return get_icon("loading", 0)
def icon_ready() -> Image.Image: return get_icon("ready", 0)
def icon_recording() -> Image.Image: return get_icon("recording", 0)
def icon_processing() -> Image.Image: return get_icon("processing", 0)
def icon_error() -> Image.Image: return get_icon("error", 0)

STATE_ICONS = {
    "loading": icon_loading,
    "ready": icon_ready,
    "recording": icon_recording,
    "processing": icon_processing,
    "error": icon_error,
}
