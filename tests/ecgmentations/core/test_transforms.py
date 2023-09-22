import pytest

import numpy as np

from ecgmentations.core.transforms import IdentityTransform


def test_IdentityTransform_CASE_repr():
    transform = IdentityTransform(always_apply=True)

    str(transform)

def test_IdentityTransform_CASE_apply():
    ecg = np.ones((12, 5000))

    transform = IdentityTransform(always_apply=True)

    output = transform(ecg=ecg)['ecg']
    expected = ecg

    assert pytest.approx(output) == expected
