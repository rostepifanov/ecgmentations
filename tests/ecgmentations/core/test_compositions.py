import pytest

import numpy as np

from ecgmentations.core.transforms import Identity
from ecgmentations.core.compositions import Sequential, OneOf

def test_Sequential_CASE_call_AND_no_transfroms():
    ecg = np.ones((12, 5000))

    transform = Sequential([])

    output = transform(ecg=ecg)['ecg']
    expected = ecg

    assert pytest.approx(output) == expected

def test_OneOf_CASE_call_AND_no_transfroms():
    ecg = np.ones((12, 5000))

    transform = OneOf([])

    output = transform(ecg=ecg)['ecg']
    expected = ecg

    assert pytest.approx(output) == expected
