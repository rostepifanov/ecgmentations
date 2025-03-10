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
def test_Transform_CASE_call_AND_mono_channel(transform):
    if transform == E.ChannelShuffle:
        return
    elif transform == E.ChannelDropout:
        return

    ecg = np.random.randn(5000).astype(np.float32)
    mask = np.zeros((5000, ), dtype=np.uint8)

    tecg = np.copy(ecg)
    tmask = np.copy(mask)

    instance = transform(always_apply=True)
    transformed = instance(ecg=tecg, mask=tmask)

    tecg, tmask = transformed['ecg'], transformed['mask']

    assert tecg.flags['C_CONTIGUOUS'] == True
    assert tecg.dtype == ecg.dtype
    assert tecg.shape == ecg.shape
    assert not np.allclose(tecg, ecg)

    assert tmask.flags['C_CONTIGUOUS'] == True
    assert tmask.dtype == mask.dtype
    assert tmask.shape == mask.shape

    if isinstance(transform, E.EcgOnlyAugmentation):
        assert np.all(tmask == mask)

@pytest.mark.parametrize('transform', SHAPE_PRESERVED_TRANSFORMS)
def test_Transform_CASE_call_AND_multi_channel(transform):
    ecg = np.random.randn(5000, 12).astype(np.float32)
    mask = np.zeros((5000, 1), dtype=np.uint8)

    tecg = np.copy(ecg)
    tmask = np.copy(mask)

    instance = transform(always_apply=True)
    transformed = instance(ecg=tecg, mask=tmask)

    tecg, tmask = transformed['ecg'], transformed['mask']

    assert tecg.flags['C_CONTIGUOUS'] == True
    assert tecg.dtype == ecg.dtype
    assert tecg.shape == ecg.shape
    assert not np.allclose(tecg, ecg)

    assert tmask.flags['C_CONTIGUOUS'] == True
    assert tmask.dtype == mask.dtype
    assert tmask.shape == mask.shape

    if isinstance(transform, E.EcgOnlyAugmentation):
        assert np.all(tmask == mask)