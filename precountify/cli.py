import fire
import librosa

from mono import Mono
from stereo import Stereo


def load(filename, sr, mono=False):
    y, sr = librosa.load(filename, sr, mono=mono)

    if y.ndim > 2:
        print('`input_file` is supported only mono or stereo.')
        return

    if sr > 44100:
        print('`sr` is only supported up to 44100')
        return

    if y.ndim == 1:
        return Mono(y, sr)
    else:
        return Stereo(y, sr)


def run(
    input_file, output_file,
    sr=None, bpm=None, meter=4, measure=2, upbeat=0, offset=0,
    click='data/click.wav'
):
    # TODO
    assert sr is None or sr > 0
    assert bpm is None or bpm > 1
    assert meter >= 1
    assert measure >= 1
    assert meter > upbeat >= 0
    assert offset >= 0

    if bpm is None:
        bpm = 240  # TODO

    audio = load(input_file, sr)
    audio = audio.trim()

    seconds_per_beat = 1 / (bpm / 60)
    n_click_samples = librosa.time_to_samples(seconds_per_beat, audio.sr)
    click = load(click, audio.sr, audio.is_mono())
    click = click.resize(n_click_samples)

    n_beats = meter * measure - upbeat
    precount = click.tile(n_beats)

    n_offset_samples = librosa.time_to_samples(offset, audio.sr)
    offsetted = audio.drop(n_offset_samples)
    precountified = precount.append(offsetted)
    precountified.save(output_file)


def main():
    fire.Fire(run)


if __name__ == '__main__':
    main()
