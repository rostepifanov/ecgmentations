import pytest

import numpy as np
import ecgmentations as E

@pytest.mark.core
def test_Identity_CASE_repr():
    instance = E.Identity(always_apply=True)

    repr = str(instance)

    assert 'Identity' in repr
    assert 'always_apply' in repr
    assert 'p' in repr

@pytest.mark.core
def test_Identity_CASE_call():
    input = np.random.randn(5000, 12)

    instance = E.Identity(always_apply=True)

    output = instance(ecg=input)['ecg']
    expected = input

    assert np.allclose(output, expected)
