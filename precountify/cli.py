import fire
import librosa
import soundfile as sf
import numpy as np


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

    y, sr = librosa.load(input_file, sr, mono=False)
    is_mono = y.ndim == 1
    if y.ndim > 2:
        print('`input_file` is only mono or stereo.')
        return

    if sr > 44100:
        print('`sr` is only supported up to 44100')
        return

    y_trimmed, _ = librosa.effects.trim(y)

    y_click, _ = librosa.load(click, sr, mono=is_mono)
    seconds_per_beat = 1 / (bpm / 60)
    n_click_samples = librosa.time_to_samples(seconds_per_beat, sr)

    if is_mono:
        y_click.resize((n_click_samples,))
    else:
        y_click_ = y_click.T.copy()
        y_click_.resize((n_click_samples, 2))
        y_click = y_click_.T

    n_beats = meter * measure - upbeat
    y_precount = np.tile(y_click, n_beats)

    n_offset_samples = librosa.time_to_samples(offset, sr)
    if is_mono:
        y_offsetted = y_trimmed[n_offset_samples:]
        precountified = np.concatenate([y_precount, y_offsetted])
    else:
        y_offsetted = y_trimmed[:, n_offset_samples:]
        precountified = np.concatenate([y_precount, y_offsetted], axis=1)

    sf.write(output_file, precountified.T, sr)


def main():
    fire.Fire(run)


if __name__ == '__main__':
    main()
