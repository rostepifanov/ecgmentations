import pytest

import numpy as np
import ecgmentations as E

SHAPE_PRESERVED_TRANSFORMS = [
    E.AmplitudeInvert,
    E.ChannelShuffle,
    E.ChannelDropout,
    E.GaussNoise,
    E.GaussBlur,
    E.AmplitudeScale,
    E.TimeReverse,
    E.TimeShift,
    E.TimeSegmentShuffle,
    E.RandomTimeWrap,
    E.TimeCutout,
    E.Pooling,
    E.Blur,
    E.PowerlineNoise,
    E.SinePulse,
    E.SquarePulse,
    E.RespirationNoise,
    E.LowPassFilter,
    E.HighPassFilter,
    E.BandPassFilter,
    E.SigmoidCompression,
]

SHAPE_UNPRESERVED_TRANSFORMS = [
    E.TimeCrop,
    E.RandomTimeCrop,
    E.CenterTimeCrop,
    E.TimePadIfNeeded,
]

@pytest.mark.parametrize('transform', SHAPE_PRESERVED_TRANSFORMS + SHAPE_UNPRESERVED_TRANSFORMS)
def test_Transform_CASE_repr(transform):
    instance = transform(always_apply=True)
    repr = str(instance)

    assert 'always_apply' in repr
    assert 'p' in repr

@pytest.mark.parametrize('transform', SHAPE_PRESERVED_TRANSFORMS)
def test_Transform_CASE_call(transform):
    ecg = np.random.uniform(size=(5000, 12))
    mask = np.zeros((5000, 1))

    instance = transform(always_apply=True)
    transformed = instance(ecg=ecg, mask=mask)

    tecg, tmask = transformed['ecg'], transformed['mask']

    assert tecg.shape == ecg.shape
    assert not np.allclose(tecg, ecg)

    assert tmask.shape == mask.shape

    if isinstance(transform, E.EcgOnlyTransform):
        assert np.all(tmask == mask)
