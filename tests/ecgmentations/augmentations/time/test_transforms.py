import pytest

import numpy as np
import ecgmentations as E

def test_TimeCrop_CASE_left_crop():
    ecg = np.random.uniform(size=(12, 5000)).T
    mask = np.zeros((1, 5000)).T

    expected_length = 4000

    transform = E.TimeCrop(length=expected_length, position='left', always_apply=True)
    transformed = transform(ecg=ecg, mask=mask)

    tecg, tmask = transformed['ecg'], transformed['mask']

    assert tecg.shape == (expected_length, *ecg.shape[1:])
    assert tmask.shape == (expected_length, *mask.shape[1:])

    assert pytest.approx(tecg) == ecg[-expected_length:]

def test_TimePadIfNeeded_CASE_left_padding():
    ecg = np.random.uniform(size=(12, 4000)).T
    mask = np.zeros((1, 4000)).T

    expected_length = 5000

    transform = E.TimePadIfNeeded(min_length=expected_length, position='left', always_apply=True)
    transformed = transform(ecg=ecg, mask=mask)

    tecg, tmask = transformed['ecg'], transformed['mask']

    assert tecg.shape == (expected_length, *ecg.shape[1:])
    assert tmask.shape == (expected_length, *mask.shape[1:])

    assert pytest.approx(tecg[:ecg.shape[0]]) == ecg
    assert pytest.approx(tecg[ecg.shape[0]:]) == 0.
