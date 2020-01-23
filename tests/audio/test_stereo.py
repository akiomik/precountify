import numpy as np
import pytest

from precountify.audio.stereo import Stereo


def test_init():
    data = np.arange(100).reshape(2, 50)
    stereo = Stereo(data, 44100, 'foo.wav')
    assert np.array_equal(stereo.data, data)
    assert stereo.sr == 44100
    assert stereo.filename == 'foo.wav'


def test_init_assertion():
    with pytest.raises(AssertionError):
        data = np.arange(100)
        Stereo(data, 44100, 'foo.wav')


def test_resize_up():
    data = np.arange(10).reshape(2, 5)
    stereo = Stereo(data, 44100, 'foo.wav')
    actual = stereo.resize(10)
    expected = np.array([
        [0, 1, 2, 3, 4, 0, 0, 0, 0, 0],
        [5, 6, 7, 8, 9, 0, 0, 0, 0, 0],
    ])

    assert np.array_equal(actual.data, expected)
    assert not np.array_equal(actual.data, stereo.data)
    assert actual.sr == stereo.sr
    assert actual.filename == stereo.filename


def test_resize_down():
    data = np.arange(20).reshape(2, 10)
    stereo = Stereo(data, 44100, 'foo.wav')
    actual = stereo.resize(5)
    expected = np.array([
        [0, 1, 2, 3, 4],
        [10, 11, 12, 13, 14],
    ])

    assert np.array_equal(actual.data, expected)
    assert not np.array_equal(actual.data, stereo.data)
    assert actual.sr == stereo.sr
    assert actual.filename == stereo.filename


def test_drop():
    data = np.arange(20).reshape(2, 10)
    stereo = Stereo(data, 44100, 'foo.wav')
    actual = stereo.drop(5)
    expected = np.array([
        [5, 6, 7, 8, 9],
        [15, 16, 17, 18, 19],
    ])

    assert np.array_equal(actual.data, expected)
    assert not np.array_equal(actual.data, stereo.data)
    assert actual.sr == stereo.sr
    assert actual.filename == stereo.filename


def test_append():
    data1 = np.arange(6).reshape(2, 3)
    stereo1 = Stereo(data1, 44100, 'foo.wav')

    data2 = np.arange(4).reshape(2, 2)
    stereo2 = Stereo(data2, 44100, 'bar.wav')

    actual = stereo1.append(stereo2)
    expected = np.array([
        [0, 1, 2, 0, 1],
        [3, 4, 5, 2, 3],
    ])

    assert np.array_equal(actual.data, expected)
    assert not np.array_equal(actual.data, stereo1.data)
    assert not np.array_equal(actual.data, stereo2.data)
    assert actual.sr == stereo1.sr
    assert actual.filename == stereo1.filename


def test_tile():
    data = np.arange(4).reshape(2, 2)
    stereo = Stereo(data, 44100, 'foo.wav')

    actual = stereo.tile(3)
    expected = np.array([
        [0, 1, 0, 1, 0, 1],
        [2, 3, 2, 3, 2, 3],
    ])

    assert np.array_equal(actual.data, expected)
    assert not np.array_equal(actual.data, stereo.data)
    assert actual.sr == stereo.sr
    assert actual.filename == stereo.filename


def test_trim():
    data1 = np.zeros(2 * 4096, dtype=np.float32).reshape(2, 4096)
    stereo1 = Stereo(data1, 44100, 'foo.wav')

    data2 = np.arange(4, dtype=np.float32).reshape(2, 2)
    stereo2 = Stereo(data2, 44100, 'foo.wav')
    stereo = stereo1.append(stereo2)

    actual = stereo.trim()

    assert actual.data.shape == (2, 512 + 2)  # TODO
    assert not np.array_equal(actual.data, stereo.data)
    assert actual.sr == stereo.sr
    assert actual.filename == stereo.filename


def test_is_mono():
    data = np.arange(4).reshape(2, 2)
    stereo = Stereo(data, 44100, 'foo.wav')

    assert not stereo.is_mono()


def test_to_mono():
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    stereo = Stereo(data, 44100, 'foo.wav')

    actual = stereo.to_mono()
    expected = np.array([1.5, 2.5, 3.5])

    assert np.array_equal(actual.data, expected)
    assert actual.sr == stereo.sr
    assert actual.filename == stereo.filename


def test_empty():
    stereo = Stereo.empty(44100)

    assert stereo.data.size == 0
    assert stereo.data.shape == (2, 0)
    assert stereo.data.dtype == np.float32
    assert stereo.sr == 44100
    assert stereo.filename is None
