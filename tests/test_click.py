import os

from precountify.audio.mono import Mono
from precountify.audio.stereo import Stereo
from precountify.click import Click


filename = os.path.join(os.path.dirname(__file__), 'fixtures/click.wav')


def test_init_mono():
    click = Click(filename, 44100, 120, True)
    assert isinstance(click.audio, Mono)
    assert click.sr == 44100
    assert click.bpm == 120
    assert click.audio.data.shape == (click.samples_per_beat(),)


def test_init_stereo():
    click = Click(filename, 44100, 120, False)
    assert isinstance(click.audio, Stereo)
    assert click.sr == 44100
    assert click.bpm == 120
    assert click.audio.data.shape == (2, click.samples_per_beat())


def test_seconds_per_beat():
    click = Click(filename, 44100, 60, False)
    assert click.seconds_per_beat() == 1

    click = Click(filename, 44100, 120, False)
    assert click.seconds_per_beat() == 0.5

    click = Click(filename, 22050, 120, False)
    assert click.seconds_per_beat() == 0.5


def test_samples_per_beat():
    click = Click(filename, 44100, 60, False)
    assert click.samples_per_beat() == 44100

    click = Click(filename, 44100, 120, False)
    assert click.samples_per_beat() == 22050

    click = Click(filename, 22050, 120, False)
    assert click.samples_per_beat() == 11025


def test_preset():
    assert os.path.exists(Click.preset())
