import threading
from typing import Callable
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard

from config import SpeechToTextConfig

class SpeechToTextInput:
    def __init__(
        self,
        config: SpeechToTextConfig,
        inject_callback: Callable[[str], None],
    ):
        self.config = config
        self.inject_callback = inject_callback
        self._model: WhisperModel | None = None
        self._recording = False
        self._audio_chunks: list = []
        self._stream: sd.InputStream | None = None
        self._listener: keyboard.Listener | None = None

    def start(self):
        """Load the model and start the key listener in a background thread."""
        threading.Thread(target=self._load_and_listen, daemon=True).start()

    def stop(self):
        if self._listener:
            self._listener.stop()

    def _load_and_listen(self):
        print(f"Loading speech-to-text model ({self.config.model_size})...")
        self._model = WhisperModel(
            self.config.model_size, device="cpu", compute_type="int8"
        )
        print(f"Speech-to-text ready — hold {self.config.trigger_key} to speak")

        trigger = getattr(keyboard.Key, self.config.trigger_key)

        def on_press(key):
            if key == trigger and not self._recording:
                self._start_recording()

        def on_release(key):
            if key == trigger and self._recording:
                self._stop_and_transcribe()

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()
        self._listener.join()  # blocks until stopped

    def _audio_callback(self, indata, frames, time, status):
        self._audio_chunks.append(indata.copy())

    def _start_recording(self):
        self._audio_chunks = []
        self._recording = True
        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()
        print("[Recording...]")

    def _stop_and_transcribe(self):
        assert(self._stream)
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self._recording = False
        print("[Transcribing...]")
        threading.Thread(target=self._transcribe, daemon=True).start()

    def _transcribe(self):
        if not self._audio_chunks:
            return
        assert(self._model)

        audio = np.concatenate(self._audio_chunks, axis=0).flatten()

        # Skip clips shorter than 0.5 seconds
        if len(audio) < self.config.sample_rate * 0.5:
            return

        segments, _ = self._model.transcribe(
            audio, beam_size=5, language=self.config.language
        )
        text = " ".join(s.text for s in segments).strip()

        if not text:
            return

        self.inject_callback(text)
