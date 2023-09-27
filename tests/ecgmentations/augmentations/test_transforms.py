import pytest

import numpy as np
import ecgmentations as E

TRANSFORMS = [
    E.Reverse,
    E.Invert,
    E.ChannelShuffle,
    E.ChannelDropout,
    E.GaussNoise,
    E.Blur,
    E.GaussBlur,
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
    ecg = np.random.uniform(size=(12, 5000)).T
    mask = np.zeros((1, 5000)).T

    transform = transform(always_apply=True)
    transformed = transform(ecg=ecg, mask=mask)

    tecg, tmask = transformed['ecg'], transformed['mask']

    assert tecg.shape == ecg.shape
    assert pytest.approx(tecg) != ecg

    assert tmask.shape == mask.shape
