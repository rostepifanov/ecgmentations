import pytest

import numpy as np
import ecgmentations as E

@pytest.mark.core
def test_Sequential_CASE_create_AND_list_error():
    with pytest.raises(RuntimeError, match=r'transforms is type of <.+> that is not list'):
        instance = E.Sequential(
            E.TimeReverse(always_apply=True),
        )

@pytest.mark.core
def test_Sequential_CASE_create_AND_subtype_error():
    with pytest.raises(RuntimeError, match=r'object at \d+ position is not subtype of Transformation'):
        instance = E.Sequential([
            E.TimeReverse(always_apply=True),
            object(),
        ], always_apply=True)

@pytest.mark.core
def test_Sequential_CASE_call_AND_no_transfroms():
    input = np.random.randn(5000, 12)

    instance = E.Sequential([
    ], always_apply=True)

    output = instance(ecg=input)['ecg']

    assert np.allclose(output, input)

@pytest.mark.core
def test_Sequential_CASE_call_AND_one_flip():
    input = np.random.randn(5000, 12)

    instance = E.Sequential([
        E.TimeReverse(always_apply=True),
    ], always_apply=True)

    output = instance(ecg=input)['ecg']

    assert not np.allclose(output, input)

@pytest.mark.core
def test_Sequential_CASE_call_AND_double_flip():
    input = np.random.randn(5000, 12)

    instance = E.Sequential([
        E.TimeReverse(always_apply=True),
        E.TimeReverse(always_apply=True)
    ], always_apply=True)

    output = instance(ecg=input)['ecg']

    assert np.allclose(output, input)

@pytest.mark.core
def test_OneOf_CASE_call_AND_no_transfroms():
    input = np.random.randn(5000, 12)

    instance = E.OneOf([
    ], always_apply=True)

    output = instance(ecg=input)['ecg']

    assert np.allclose(output, input)

@pytest.mark.core
def test_OneOf_CASE_call_AND_check_application():
    input = np.random.randn(5000, 12)

    instance = E.Sequential([
        E.TimeReverse(always_apply=True),
        E.OneOf([
            E.TimeReverse(),
            E.TimeReverse(),
        ], always_apply=True),
    ], always_apply=True)

    output = instance(ecg=input)['ecg']

    assert np.allclose(output, input)
