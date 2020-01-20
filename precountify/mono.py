import numpy as np

from audio_file import AudioFile


class Mono(AudioFile):
    def __init__(self, data, sr):
        assert data.ndim == 1
        super(Mono, self).__init__(data, sr)

    def resize(self, n):
        data = self.data.copy()
        data.resize(n)
        return Mono(data, self.sr)

    def drop(self, n):
        return Mono(self.data[n:], self.sr)

    def append(self, that):
        assert self.data.ndim == that.data.ndim
        concat = np.concatenate([self.data, that.data])
        return Mono(concat, self.sr)

    def tile(self, n):
        return Mono(self._tile(n), self.sr)

    def trim(self):
        return Mono(self._trim(), self.sr)

    def is_mono(self):
        return True
