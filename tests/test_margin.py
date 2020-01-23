from precountify.audio.mono import Mono
from precountify.audio.stereo import Stereo
from precountify.margin import Margin


def test_margin_mono():
    margin_in_seconds = 0.5
    sr = 44100
    margin = Margin(margin_in_seconds, sr, True)
    assert isinstance(margin.audio, Mono)
    assert margin.audio.data.shape == (margin.n_margin_samples(),)


def test_margin_stereo():
    margin_in_seconds = 0.5
    sr = 44100
    margin = Margin(margin_in_seconds, sr, False)
    assert isinstance(margin.audio, Stereo)
    assert margin.audio.data.shape == (2, margin.n_margin_samples())


def test_n_margin_samples():
    margin_in_seconds = 0.5
    sr = 44100
    margin = Margin(margin_in_seconds, sr, False)
    expected = margin_in_seconds * sr
    assert margin.n_margin_samples() == expected
