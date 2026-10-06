import numpy as np
from dictation_engine import NeuralVAD
import time

vad = NeuralVAD()

# Generate a synthetic speech-like signal (e.g. noise mixed with tone)
t = np.linspace(0, 512/16000, 512, endpoint=False)
speech = (np.sin(2 * np.pi * 440 * t) + np.random.randn(512)*0.1).astype(np.float32)
# normalize
speech = speech / np.max(np.abs(speech))

print("Prob:", vad.process(speech))
for _ in range(10):
    print("Prob:", vad.process(speech))
