import pytest

import numpy as np

from ecgmentations.core.transforms import Identity


def test_Identity_CASE_repr():
    transform = Identity(always_apply=True)

    repr = str(transform)

    assert 'Identity' in repr
    assert 'always_apply' in repr
    assert 'p' in repr

def test_Identity_CASE_call():
    ecg = np.ones((12, 5000))

    transform = Identity(always_apply=True)

    output = transform(ecg=ecg)['ecg']
    expected = ecg

    assert pytest.approx(output) == expected
