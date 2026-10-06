import numpy as np
from dictation_engine import NeuralVAD

vad = NeuralVAD()

# Generate random noise
noise = np.random.randn(512).astype(np.float32) * 0.001

# Run VAD on noise
vad.reset_state()
print("Noise prob:", vad.process(noise))

# Run VAD on scaled noise
vad.reset_state()
print("Scaled noise prob:", vad.process(noise * 10))
