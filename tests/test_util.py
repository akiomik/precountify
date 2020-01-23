from precountify.util import import_string
from precountify.estimator.librosa import LibrosaTempoEstimator
from precountify.estimator.madmom import MadmomTempoEstimator


def test_import_string():
    librosa = import_string(
        'precountify.estimator.librosa.LibrosaTempoEstimator')
    assert type(librosa) == type(LibrosaTempoEstimator)

    madmom = import_string('precountify.estimator.madmom.MadmomTempoEstimator')
    assert type(madmom) == type(MadmomTempoEstimator)
