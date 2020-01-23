import librosa

from precountify.margin import Margin
from precountify.music import Music
from precountify.click import Click
from precountify.precount import Precount
from precountify.precountify import precountify


def test_precountify_with_bpm():
    bpm = 120
    meter = 4
    measure = 2
    upbeat = 0
    filename = librosa.util.example_audio_file()

    precountified = precountify(
        filename, bpm=bpm,
        meter=meter, measure=measure, upbeat=upbeat)

    music = Music.from_file(filename).trimmed()
    click = Click(Click.preset(), music.audio.sr, bpm)
    precount = Precount.from_click(click, meter, measure, upbeat)
    expected_shape = (
        2,
        precount.audio.data.shape[1] + music.audio.data.shape[1]
    )
    assert precountified.data.shape == expected_shape


def test_precountify_with_bpm_and_margin():
    bpm = 120
    meter = 4
    measure = 2
    upbeat = 0
    margin = 2
    filename = librosa.util.example_audio_file()

    precountified = precountify(
        filename, bpm=bpm,
        meter=meter, measure=measure, upbeat=upbeat, margin=margin)

    music = Music.from_file(filename).trimmed()
    click = Click(Click.preset(), music.audio.sr, bpm)
    precount = Precount.from_click(click, meter, measure, upbeat)
    margin = Margin(margin, music.audio.sr)
    expected_shape = (
        2,
        (margin.audio.data.shape[1] +
         precount.audio.data.shape[1] +
         music.audio.data.shape[1])
    )
    assert precountified.data.shape == expected_shape


def test_precountify_with_bpm_and_offset():
    bpm = 120
    meter = 4
    measure = 2
    upbeat = 0
    offset = 2
    filename = librosa.util.example_audio_file()

    precountified = precountify(
        filename, bpm=bpm,
        meter=meter, measure=measure, upbeat=upbeat, offset=offset)

    music = Music.from_file(filename).trimmed().offsetted(offset)
    click = Click(Click.preset(), music.audio.sr, bpm)
    precount = Precount.from_click(click, meter, measure, upbeat)
    expected_shape = (
        2,
        precount.audio.data.shape[1] + music.audio.data.shape[1]
    )
    assert precountified.data.shape == expected_shape


def test_precountify_with_bpm_and_margin_and_offset():
    bpm = 120
    meter = 4
    measure = 2
    upbeat = 0
    margin = 2
    offset = 2
    filename = librosa.util.example_audio_file()

    precountified = precountify(
        filename, bpm=bpm,
        meter=meter, measure=measure, upbeat=upbeat,
        margin=margin, offset=offset)

    music = Music.from_file(filename).trimmed()
    click = Click(Click.preset(), music.audio.sr, bpm)
    precount = Precount.from_click(click, meter, measure, upbeat)
    margin = Margin(margin, music.audio.sr)
    expected_shape = (
        2,
        precount.audio.data.shape[1] + music.audio.data.shape[1]
    )
    assert precountified.data.shape == expected_shape
