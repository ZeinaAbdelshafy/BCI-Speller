import numpy as np
import collections
import threading
from pylsl import StreamInlet, resolve_byprop
from config import FS

class UnicornLSLStreamer:
    def __init__(self, fs=FS):
        self.fs = fs
        self.inlet = None
        self.buffer = collections.deque(maxlen=int(fs * 10))  # 10s FIFO
        self.stop_event = threading.Event()
        self.channel_indices = list(range(8))  # Fallback

    def connect(self):
        print("🔍 Resolving Unicorn LSL stream...")
        streams = resolve_byprop("name", "UnicornRecorderDataLSLStream", timeout=5.0)
        if not streams:
            streams = resolve_byprop("type", "EEG", timeout=5.0)
        if not streams:
            print("❌ No stream found. Check Unicorn Host & Bluetooth.")
            return False

        self.inlet = StreamInlet(streams[0])
        print(f"✅ Connected: {self.inlet.info().name()}")

        # Parse channel labels (matches team leader's metadata extraction)
        ch = self.inlet.info().desc().child("channels").child("channel")
        labels = []
        while not ch.empty():
            labels.append(ch.child_value("label").upper())
            ch = ch.next_sibling("channel")

        target = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
        if len(labels) == 8:
            try:
                self.channel_indices = [labels.index(t) for t in target]
            except ValueError:
                print("⚠️  Labels mismatch. Using default order.")
        return True

    def start(self):
        self.stop_event.clear()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self.stop_event.is_set():
            samples, _ = self.inlet.pull_chunk(timeout=0.1, max_samples=32)
            for s in samples:
                self.buffer.append(s)

    def get_window(self, seconds=2.0):
        n = int(seconds * self.fs)
        if len(self.buffer) < n:
            return None
        return np.array(list(self.buffer)[-n:]).T[self.channel_indices, :]

    def stop(self):
        self.stop_event.set()