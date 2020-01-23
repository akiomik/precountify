import librosa

# from precountify.audio.mono import Mono
from precountify.audio.stereo import Stereo
from precountify.music import Music


def test_from_file_mono():
    # TODO
    pass


def test_from_file_stereo():
    filename = librosa.util.example_audio_file()
    music = Music.from_file(filename)
    assert isinstance(music.audio, Stereo)


def test_trimmed():
    filename = librosa.util.example_audio_file()
    music = Music.from_file(filename)
    trimmed = music.trimmed()
    assert trimmed.audio.data.size < music.audio.data.size


def test_offsetted():
    offset_in_seconds = 1
    filename = librosa.util.example_audio_file()
    music = Music.from_file(filename)
    offsetted = music.offsetted(offset_in_seconds)
    expected_shape = (
        music.audio.data.shape[0],
        music.audio.data.shape[1] - music.audio.sr * offset_in_seconds
    )
    assert offsetted.audio.data.shape == expected_shape


def test_prepend():
    # TODO
    pass
