import pytest

import numpy as np
import ecgmentations as E

SHAPE_PRESERVED_TRANSFORMS = [
    E.TimeReverse,
    E.AmplitudeInvert,
    E.ChannelShuffle,
    E.ChannelDropout,
    E.GaussNoise,
    E.Blur,
    E.GaussBlur,
    E.AmplitudeScale,
    E.TimeCutout,
    E.RandomTimeWrap,
]

SHAPE_UNPRESERVED_TRANSFORMS = [
    E.RandomTimeCrop
]

@pytest.mark.parametrize('transform', SHAPE_PRESERVED_TRANSFORMS + SHAPE_UNPRESERVED_TRANSFORMS)
def test_Transform_CASE_repr(transform):
    ecg = np.ones((12, 5000)).T

    transform = transform(always_apply=True)
    repr = str(transform)

    assert 'always_apply' in repr
    assert 'p' in repr

@pytest.mark.parametrize('transform', SHAPE_PRESERVED_TRANSFORMS)
def test_Transform_CASE_call(transform):
    ecg = np.random.uniform(size=(12, 5000)).T
    mask = np.zeros((1, 5000)).T

    transform = transform(always_apply=True)
    transformed = transform(ecg=ecg, mask=mask)

    tecg, tmask = transformed['ecg'], transformed['mask']

    assert tecg.shape == ecg.shape
    assert pytest.approx(tecg) != ecg

    assert tmask.shape == mask.shape
