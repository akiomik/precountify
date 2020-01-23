import numpy as np
import pytest

from precountify.audio.mono import Mono


def test_init():
    data = np.arange(100)
    mono = Mono(data, 44100, 'foo.wav')
    assert np.array_equal(mono.data, data)
    assert mono.sr == 44100
    assert mono.filename == 'foo.wav'


def test_init_assertion():
    with pytest.raises(AssertionError):
        data = np.arange(100).reshape(2, 50)
        Mono(data, 44100, 'foo.wav')


def test_resize_up():
    data = np.arange(5)
    mono = Mono(data, 44100, 'foo.wav')
    actual = mono.resize(10)
    expected = np.array([0, 1, 2, 3, 4, 0, 0, 0, 0, 0])

    assert np.array_equal(actual.data, expected)
    assert not np.array_equal(actual.data, mono.data)
    assert actual.sr == mono.sr
    assert actual.filename == mono.filename


def test_resize_down():
    data = np.arange(10)
    mono = Mono(data, 44100, 'foo.wav')
    actual = mono.resize(5)
    expected = np.array([0, 1, 2, 3, 4])

    assert np.array_equal(actual.data, expected)
    assert not np.array_equal(actual.data, mono.data)
    assert actual.sr == mono.sr
    assert actual.filename == mono.filename


def test_drop():
    data = np.arange(10)
    mono = Mono(data, 44100, 'foo.wav')
    actual = mono.drop(5)
    expected = np.array([5, 6, 7, 8, 9])

    assert np.array_equal(actual.data, expected)
    assert not np.array_equal(actual.data, mono.data)
    assert actual.sr == mono.sr
    assert actual.filename == mono.filename


def test_append():
    data1 = np.arange(3)
    mono1 = Mono(data1, 44100, 'foo.wav')

    data2 = np.arange(2)
    mono2 = Mono(data2, 44100, 'bar.wav')

    actual = mono1.append(mono2)
    expected = np.array([0, 1, 2, 0, 1])

    assert np.array_equal(actual.data, expected)
    assert not np.array_equal(actual.data, mono1.data)
    assert not np.array_equal(actual.data, mono2.data)
    assert actual.sr == mono1.sr
    assert actual.filename == mono1.filename


def test_tile():
    data = np.arange(2)
    mono = Mono(data, 44100, 'foo.wav')

    actual = mono.tile(3)
    expected = np.array([0, 1, 0, 1, 0, 1])

    assert np.array_equal(actual.data, expected)
    assert not np.array_equal(actual.data, mono.data)
    assert actual.sr == mono.sr
    assert actual.filename == mono.filename


def test_trim():
    data1 = np.zeros(4096, dtype=np.float32)
    mono1 = Mono(data1, 44100, 'foo.wav')

    data2 = np.arange(2, dtype=np.float32)
    mono2 = Mono(data2, 44100, 'foo.wav')
    mono = mono1.append(mono2)

    actual = mono.trim()

    assert actual.data.shape == (512 + 2,)  # TODO
    assert not np.array_equal(actual.data, mono.data)
    assert actual.sr == mono.sr
    assert actual.filename == mono.filename


def test_is_mono():
    data = np.arange(2)
    mono = Mono(data, 44100, 'foo.wav')

    assert mono.is_mono()


def test_empty():
    mono = Mono.empty(44100)

    assert mono.data.size == 0
    assert mono.data.shape == (0,)
    assert mono.data.dtype == np.float32
    assert mono.sr == 44100
    assert mono.filename is None
