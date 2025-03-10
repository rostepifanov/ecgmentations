import pytest

import numpy as np
import ecgmentations as E
import ecgmentations.core.constants as C

def test_TimeCrop_CASE_left_crop():
    length = 5000
    expected_length = 4000

    ecg = np.random.randn(length, 12)
    mask = np.zeros((length, 1))

    instance = E.TimeCrop(length=expected_length, position=E.PositionType.LEFT, always_apply=True)
    transformed = instance(ecg=ecg, mask=mask)

    tecg, tmask = transformed['ecg'], transformed['mask']

    assert tecg.shape == (expected_length, ecg.shape[C.CHANNEL_DIM])
    assert tmask.shape == (expected_length, mask.shape[C.CHANNEL_DIM])

    assert np.allclose(tecg, ecg[:expected_length])

def test_TimePadIfNeeded_CASE_left_padding():
    length = 4000
    expected_length = 5000

    ecg = np.random.randn(length, 12)
    mask = np.zeros((length, 1))

    instance = E.TimePadIfNeeded(min_length=expected_length, position=E.PositionType.LEFT, always_apply=True)
    transformed = instance(ecg=ecg, mask=mask)

    tecg, tmask = transformed['ecg'], transformed['mask']

    assert tecg.shape == (expected_length, ecg.shape[C.CHANNEL_DIM])
    assert tmask.shape == (expected_length, mask.shape[C.CHANNEL_DIM])

    assert np.allclose(tecg[:length], ecg)
    assert np.allclose(tecg[length:], 0.)

def test_TimeCutout_CASE_mask_fill_value():
    ecg = np.random.randn(5000, 12)
    mask = np.zeros((5000, 1))

    mask_fill_value = 0

    instance = E.TimeCutout(mask_fill_value=mask_fill_value, always_apply=True)
    transformed = instance(ecg=ecg, mask=mask)

    tmask = transformed['mask']

    assert np.all(tmask[mask != tmask], mask_fill_value)

@pytest.mark.parametrize('reduction', list(map(lambda t: t.value, E.ReductionType)))
def test_Pooling_CASE_reduction(reduction):
    input = np.random.randn(5000, 12)

    instance = E.Pooling(reduction, always_apply=True)
    output = instance(ecg=input)['ecg']

    assert output.shape == input.shape
    assert not np.allclose(output, input)
