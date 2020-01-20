from abc import ABC, abstractmethod

import librosa
import numpy as np
import soundfile as sf


class AudioFile(ABC):
    def __init__(self, data, sr):
        super(AudioFile, self).__init__()
        self.data = data
        self.sr = sr

    @abstractmethod
    def resize(self, length):
        pass

    @abstractmethod
    def drop(self, length):
        pass

    @abstractmethod
    def append(self, data):
        pass

    @abstractmethod
    def tile(self, n):
        pass

    @abstractmethod
    def trim(self):
        pass

    @abstractmethod
    def is_mono(self):
        pass

    def _tile(self, n):
        return np.tile(self.data, n)

    def _trim(self):
        trimmed, _ = librosa.effects.trim(self.data)
        return trimmed

    def save(self, filename):
        sf.write(filename, self.data.T, self.sr)
