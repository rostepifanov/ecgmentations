import pytest

import numpy as np
import ecgmentations as E

def test_Sequential_CASE_call_AND_no_transfroms():
    ecg = np.ones((12, 5000))

    transform = E.Sequential([])

    output = transform(ecg=ecg)['ecg']
    expected = ecg

    assert pytest.approx(output) == expected

def test_Sequential_CASE_call_AND_one_flip():
    ecg = np.random.random((12, 5000))

    transform = E.Sequential([
        E.TimeReverse(always_apply=True)
    ])

    output = transform(ecg=ecg)['ecg']

    assert pytest.approx(output) != ecg

def test_Sequential_CASE_call_AND_double_flip():
    ecg = np.random.random((12, 5000))

    transform = E.Sequential([
        E.TimeReverse(always_apply=True),
        E.TimeReverse(always_apply=True)
    ])

    output = transform(ecg=ecg)['ecg']
    expected = ecg

    assert pytest.approx(output) == expected

def test_OneOf_CASE_call_AND_no_transfroms():
    ecg = np.ones((12, 5000))

    transform = E.OneOf([])

    output = transform(ecg=ecg)['ecg']
    expected = ecg

    assert pytest.approx(output) == expected
