import fire
import os

from .click import Click
from .margin import Margin
from .music import Music
from .precount import Precount


def import_string(path):
    path_components = path.split('.')
    cls_name = path_components[-1]
    module_name = '.'.join(path_components[:-1])
    module = __import__(
        module_name, fromlist=[cls_name])
    return getattr(module, cls_name)


def run(
    input_file, output_file,
    sr=None, bpm=None, click=None,
    meter=4, measure=2, upbeat=0, offset=0, margin=0,
    estimator='precountify.estimator.librosa.LibrosaTempoEstimator'
):
    # TODO
    assert sr is None or sr > 0
    assert bpm is None or bpm > 1
    assert meter >= 1
    assert measure >= 1
    assert meter > upbeat >= 0
    assert offset >= 0
    assert margin >= 0

    music = Music.from_file(input_file, sr).trimmed()

    if offset > 0:
        music = music.offsetted(offset)

    if bpm is None:
        estimator_cls = import_string(estimator)
        bpm = estimator_cls.estimate(music.audio)
        print('[INFO] estimated bpm:', bpm)

    if click is None:
        click = os.path.join(os.path.dirname(__file__), 'data/click.wav')

    click = Click(click, music.audio.sr, bpm, music.audio.is_mono())
    precount = Precount.from_click(click, meter, measure, upbeat)

    if margin > 0:
        margin = Margin(margin, music.audio.sr, music.audio.is_mono())
        precount = precount.prepend(margin)

    precountified = music.prepend(precount)
    precountified.audio.save(output_file)


def main():
    fire.Fire(run)


if __name__ == '__main__':
    main()
