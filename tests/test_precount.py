import os

from precountify.click import Click
from precountify.precount import Precount


def test_prepend():
    # TODO
    pass


def test_from_click():
    filename = os.path.join(os.path.dirname(__file__), 'fixtures/click.wav')
    click = Click(filename, 44100, 120, False)
    precount = Precount.from_click(click, 4, 2, 0)
    expected_shape = (
        click.audio.data.shape[0],
        click.audio.data.shape[1] * Precount.n_beats(4, 2, 0)
    )
    assert precount.audio.data.shape == expected_shape


def test_n_beats():
    assert Precount.n_beats(4, 2, 0) == 8
    assert Precount.n_beats(3, 2, 0) == 6
    assert Precount.n_beats(4, 1, 0) == 4
    assert Precount.n_beats(4, 2, 1) == 7
