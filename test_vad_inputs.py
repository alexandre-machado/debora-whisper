import onnxruntime as ort
from dictation_engine import MODEL_DIR
import sys

model_path = MODEL_DIR / "silero_vad" / "silero_vad.onnx"
sess = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
for i in sess.get_inputs():
    print(f"Input {i.name}: {i.shape} {i.type}")
for o in sess.get_outputs():
    print(f"Output {o.name}: {o.shape} {o.type}")
