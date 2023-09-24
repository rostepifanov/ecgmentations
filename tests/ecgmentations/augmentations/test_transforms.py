import pytest

import numpy as np
import ecgmentations as E

TRANSFORMS = [
    E.Reverse,
    E.Invert,
    E.ChannelShuffle,
    E.ChannelDropout,
    E.GaussNoise,
]

@pytest.mark.parametrize('transform', TRANSFORMS)
def test_Transform_CASE_repr(transform):
    ecg = np.ones((12, 5000)).T

    transform = transform(always_apply=True)
    repr = str(transform)

    assert 'always_apply' in repr
    assert 'p' in repr

@pytest.mark.parametrize('transform', TRANSFORMS)
def test_Transform_CASE_call(transform):
    ecg = np.ones((12, 5000)).T

    transform = transform(always_apply=True)
    output = transform(ecg=ecg)['ecg']
