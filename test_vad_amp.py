import numpy as np
import soundfile as sf
import urllib.request
from dictation_engine import NeuralVAD

vad = NeuralVAD()

# Download a short speech sample (16kHz)
url = "https://github.com/snakers4/silero-vad/raw/master/en.wav"
urllib.request.urlretrieve(url, "en.wav")

audio, sr = sf.read("en.wav")
if len(audio.shape) > 1:
    audio = audio[:, 0]

print(f"Original max amp: {np.max(np.abs(audio)):.4f}")

for scale in [1.0, 0.1, 0.01, 0.001]:
    scaled_audio = (audio * scale).astype(np.float32)
    vad.reset_state()
    max_prob = 0.0
    for i in range(0, len(scaled_audio), 512):
        block = scaled_audio[i:i+512]
        if len(block) == 512:
            prob = vad.process(block)
            max_prob = max(max_prob, prob)
    print(f"Scale: {scale:.3f} -> Max Prob: {max_prob:.4f}")
