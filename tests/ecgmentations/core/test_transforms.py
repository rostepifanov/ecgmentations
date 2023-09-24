import pytest

import numpy as np
import ecgmentations as E

def test_Identity_CASE_repr():
    transform = E.Identity(always_apply=True)

    repr = str(transform)

    assert 'Identity' in repr
    assert 'always_apply' in repr
    assert 'p' in repr

def test_Identity_CASE_call():
    ecg = np.ones((12, 5000)).T

    transform = E.Identity(always_apply=True)

    output = transform(ecg=ecg)['ecg']
    expected = ecg

    assert pytest.approx(output) == expected
