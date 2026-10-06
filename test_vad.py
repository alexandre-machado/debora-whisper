import sys
from pathlib import Path
import numpy as np

# Mock MODEL_DIR
from dictation_engine import MODEL_DIR, NeuralVAD

vad = NeuralVAD()
print("VAD loaded.")

# Generate some silence
silence = np.zeros(512, dtype=np.float32)
prob1 = vad.process(silence)
print(f"Silence prob: {prob1}")

# Generate a sine wave (simulated speech)
t = np.linspace(0, 512/16000, 512, endpoint=False)
speech = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
prob2 = vad.process(speech)
print(f"Tone prob: {prob2}")
