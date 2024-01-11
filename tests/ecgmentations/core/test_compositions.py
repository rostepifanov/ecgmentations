import pytest

import numpy as np
import ecgmentations as E

@pytest.mark.core
def test_Sequential_CASE_create_AND_warning():
    with pytest.warns(UserWarning, match='transforms is single transform, but a sequence is expected! Transform will be wrapped into list.'):
        transform = E.Sequential(
            E.TimeReverse(always_apply=True)
        )

@pytest.mark.core
def test_Sequential_CASE_call_AND_no_transfroms():
    ecg = np.ones((12, 5000))

    transform = E.Sequential([])

    output = transform(ecg=ecg)['ecg']
    expected = ecg

    assert pytest.approx(output) == expected

@pytest.mark.core
def test_Sequential_CASE_call_AND_one_flip():
    ecg = np.random.random((12, 5000))

    transform = E.Sequential([
        E.TimeReverse(always_apply=True)
    ])

    output = transform(ecg=ecg)['ecg']

    assert pytest.approx(output) != ecg

@pytest.mark.core
def test_Sequential_CASE_call_AND_double_flip():
    ecg = np.random.random((12, 5000))

    transform = E.Sequential([
        E.TimeReverse(always_apply=True),
        E.TimeReverse(always_apply=True)
    ])

    output = transform(ecg=ecg)['ecg']
    expected = ecg

    assert pytest.approx(output) == expected

@pytest.mark.core
def test_OneOf_CASE_call_AND_no_transfroms():
    ecg = np.ones((12, 5000))

    transform = E.OneOf([])

    output = transform(ecg=ecg)['ecg']
    expected = ecg

    assert pytest.approx(output) == expected
