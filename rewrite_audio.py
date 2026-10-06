import os

def rewrite():
    path = r"D:\repos\alexandre-machado\npu-whisper\dictation_engine.py"
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    start_idx = -1
    end_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("class AudioRecorder:"):
            start_idx = i
        if start_idx != -1 and line.startswith("def type_text("):
            # Backtrack to the comment block
            for j in range(i-1, 0, -1):
                if lines[j].startswith("# ---------------------------------------------------------------------------"):
                    end_idx = j
                    break
            break
            
    if start_idx == -1 or end_idx == -1:
        print("Could not find AudioRecorder bounds")
        return
        
    new_class = """class AudioRecorder:
    \"\"\"Record audio from microphone using sounddevice. Supports PTT and VAD.\"\"\"

    def __init__(self, sample_rate: int = 16000, channels: int = 1,
                 max_record_seconds: float = None, on_timeout=None, config=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_record_seconds = max_record_seconds
        self.on_timeout = on_timeout
        self.config = config or {}
        self.recording = False
        self.continuous = self.config.get("continuous_listening", False)
        
        self._frames = []

        import numpy as np
        import queue
        
        self._stream = None
        self._lock = threading.Lock()
        self._timer = None
        self._recording_generation = 0
        self.telemetry = {}
        self._audio_ready = threading.Event()
        self._last_callback = None
        self._expected_adc_time = None
        self._stable_callbacks = 0

        # Continuous VAD properties
        self.capacity = int(sample_rate * self.config.get("ring_buffer_seconds", 30))
        self._buffer = np.zeros((self.capacity, channels), dtype=np.float32)
        self._write_pos = 0
        self._read_pos = 0
        self._lookback_count = 0
        self._data_cv = threading.Condition(self._lock)
        self.segment_queue = queue.Queue(maxsize=10)
        self._vad_thread = None
        self._stop_vad = False
        
        # VAD thresholds
        self.energy_threshold = self.config.get("vad_energy_threshold", 0.015)
        self.min_speech_frames = int(sample_rate * self.config.get("vad_min_speech_seconds", 0.25))
        self.end_silence_frames = int(sample_rate * self.config.get("vad_end_silence_seconds", 0.8))
        self.max_segment_frames = int(sample_rate * self.config.get("segment_max_seconds", 15))
        self.lookback_frames = int(sample_rate * self.config.get("vad_lookback_seconds", 0.5))
        self.trailing_frames = int(sample_rate * self.config.get("vad_trailing_seconds", 0.3))

    def warmup(self, timeout=3.0):
        \"\"\"Open the stream continuously in the background.\"\"\"
        import sounddevice as sd
        if self._stream is not None:
            self.wait_ready(timeout)
            return

        def callback(indata, frames, time_info, status):
            now = time.perf_counter()
            adc_time = time_info.inputBufferAdcTime
            delivery_delay = max(0.0, time_info.currentTime - adc_time) if adc_time > 0 else 0.0
            
            with self._lock:
                gap = now - self._last_callback if self._last_callback is not None else 0.0
                adc_gap = (max(0.0, adc_time - self._expected_adc_time)
                           if adc_time > 0 and self._expected_adc_time is not None else 0.0)
                self._last_callback = now
                self._expected_adc_time = adc_time + frames / self.sample_rate if adc_time > 0 else None
                self._stable_callbacks = self._stable_callbacks + 1 if gap < 0.5 and not status else 0
                
                if self._stable_callbacks >= 3:
                    self._audio_ready.set()
                else:
                    self._audio_ready.clear()
                    
                # Always write to ring buffer
                capacity = self.capacity
                count = min(frames, capacity)
                data = indata[-count:]
                first = min(count, capacity - self._write_pos)
                self._buffer[self._write_pos:self._write_pos + first] = data[:first]
                self._buffer[:count - first] = data[first:]
                self._write_pos = (self._write_pos + count) % capacity
                self._lookback_count = min(capacity, self._lookback_count + count)
                
                if self.continuous:
                    self._data_cv.notify()

                if self.recording and not self.continuous:
                    self.telemetry.setdefault('first_frame', now)
                    self.telemetry['live_frames'] += frames
                    self.telemetry['input_overflows'] += int(status.input_overflow)
                    self.telemetry['max_callback_gap'] = max(self.telemetry['max_callback_gap'], gap)
                    self.telemetry['max_adc_gap'] = max(self.telemetry['max_adc_gap'], adc_gap)
                    self.telemetry['max_delivery_delay'] = max(self.telemetry['max_delivery_delay'], delivery_delay)
                    self._frames.append(indata.copy())

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=0,
                latency="low",
                callback=callback,
            )
            self._stream.start()
            _log_audio_stream(sd, self._stream, "input")
            self.wait_ready(timeout)
            
            if self.continuous:
                self._stop_vad = False
                self._vad_thread = threading.Thread(target=self._vad_loop, daemon=True)
                self._vad_thread.start()
                
        except Exception:
            self.close()
            raise

    def wait_ready(self, timeout=3.0):
        if not self._audio_ready.wait(timeout):
            raise RuntimeError("Microphone did not deliver stable audio callbacks during warmup")
        with self._lock:
            if (self._stream is None or not self._stream.active or
                    self._last_callback is None or time.perf_counter() - self._last_callback > 0.5):
                raise RuntimeError("Microphone audio stream is inactive or stalled")

    def _vad_loop(self):
        import numpy as np
        is_speaking = False
        speech_start_pos = 0
        silence_frames = 0
        speech_frames = 0
        
        while not self._stop_vad:
            with self._lock:
                self._data_cv.wait(timeout=0.1)
                if self._write_pos >= self._read_pos:
                    available = self._write_pos - self._read_pos
                else:
                    available = self.capacity - self._read_pos + self._write_pos
                    
                if available == 0:
                    continue
                    
                if self._write_pos > self._read_pos:
                    new_data = self._buffer[self._read_pos:self._write_pos].copy()
                else:
                    new_data = np.concatenate((
                        self._buffer[self._read_pos:], 
                        self._buffer[:self._write_pos]
                    ))
                
                start_read_pos = self._read_pos
                self._read_pos = self._write_pos
                
            block_size = int(self.sample_rate * 0.02)
            for i in range(0, len(new_data), block_size):
                block = new_data[i:i+block_size]
                if len(block) == 0:
                    continue
                    
                rms = float(np.sqrt(np.mean(block**2)))
                block_len = len(block)
                
                if rms > self.energy_threshold:
                    if not is_speaking:
                        is_speaking = True
                        speech_start_pos = (start_read_pos + i - self.lookback_frames) % self.capacity
                        silence_frames = 0
                        speech_frames = self.lookback_frames + block_len
                    else:
                        silence_frames = 0
                        speech_frames += block_len
                else:
                    if is_speaking:
                        silence_frames += block_len
                        speech_frames += block_len
                        
                # End of speech conditions
                cut_segment = False
                if is_speaking and silence_frames > self.end_silence_frames:
                    if speech_frames >= self.min_speech_frames:
                        cut_segment = True
                    else:
                        # Too short, discard
                        is_speaking = False
                        
                elif is_speaking and speech_frames >= self.max_segment_frames:
                    # Forced cut
                    cut_segment = True
                    
                if cut_segment:
                    is_speaking = False
                    # Extract segment
                    with self._lock:
                        end_pos = (start_read_pos + i + block_len + self.trailing_frames) % self.capacity
                        if end_pos > speech_start_pos:
                            audio = self._buffer[speech_start_pos:end_pos].copy()
                        else:
                            audio = np.concatenate((
                                self._buffer[speech_start_pos:],
                                self._buffer[:end_pos]
                            ))
                        
                    self.segment_queue.put(audio.flatten())

    def start(self):
        \"\"\"Start recording (PTT mode).\"\"\"
        if self.continuous:
            return # VAD handles recording
            
        with self._lock:
            if self.recording:
                return
            self.telemetry = dict(start_called=time.perf_counter(), live_frames=0,
                                  input_overflows=0, max_callback_gap=0.0,
                                  max_adc_gap=0.0, max_delivery_delay=0.0)
            
            # Extract lookback from continuous buffer
            count = min(self._lookback_count, int(self.sample_rate * 1.5))
            begin = (self._write_pos - count) % self.capacity
            first = min(count, self.capacity - begin)
            self._frames = []
            if first:
                self._frames.append(self._buffer[begin:begin + first].copy())
            if count > first:
                self._frames.append(self._buffer[:count - first].copy())
                
            self.recording = True
            self._recording_generation += 1
            if self.max_record_seconds:
                self._timer = threading.Timer(
                    self.max_record_seconds, self._timeout_stop,
                    args=(self._recording_generation,),
                )
                self._timer.daemon = True
                self._timer.start()

        log("Recording started...")

    def _timeout_stop(self, generation):
        with self._lock:
            if not self.recording or generation != self._recording_generation:
                return
            self.recording = False
            self._timer = None
        log(f"Max recording time ({self.max_record_seconds}s) reached, stopping.")
        if self.on_timeout is not None:
            self.on_timeout(generation)

    def stop(self):
        \"\"\"Stop recording and return audio as numpy array (PTT mode).\"\"\"
        import numpy as np

        if self.continuous:
            return np.array([], dtype=np.float32)

        with self._lock:
            self.recording = False
            frames = self._frames
            self._frames = []
            telemetry = dict(self.telemetry)
            if self._timer:
                self._timer.cancel()
                self._timer = None

        if not frames:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(frames, axis=0).flatten()
        duration = len(audio) / self.sample_rate
        log(f"Recording stopped. Duration: {duration:.1f}s")
        return audio

    def close(self):
        with self._lock:
            self.recording = False
            self._stop_vad = True
            self._data_cv.notify_all()
            if self._timer:
                self._timer.cancel()
                self._timer = None
        stream, self._stream = self._stream, None
        try:
            if stream is not None:
                try:
                    stream.stop()
                finally:
                    stream.close()
        finally:
            with self._lock:
                self.recording = False
                self._frames = []
                self._lookback_count = 0
                self._last_callback = None
                self._expected_adc_time = None
                self._stable_callbacks = 0
                self._audio_ready.clear()

    @property
    def audio_level(self) -> float:
        import numpy as np
        with self._lock:
            if self.continuous:
                if self._lookback_count == 0:
                    return 0.0
                last_frame = self._buffer[(self._write_pos - 1024) % self.capacity : self._write_pos].copy()
            else:
                if not self._frames or not self.recording:
                    return 0.0
                last_frame = self._frames[-1].copy()
                
        if len(last_frame) == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(last_frame ** 2)))
        return min(rms * 25.0, 1.0)
"""
    
    lines[start_idx:end_idx] = [new_class + "\n\n"]
    
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
        
if __name__ == "__main__":
    rewrite()
    print("Done")
