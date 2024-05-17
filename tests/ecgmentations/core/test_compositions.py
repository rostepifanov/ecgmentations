import pytest

import numpy as np
import ecgmentations as E

@pytest.mark.core
def test_Sequential_CASE_create_AND_list_error():
    with pytest.raises(RuntimeError, match=r'transforms is type of <.+> that is not list'):
        transform = E.Sequential(
            E.TimeReverse(always_apply=True)
        )

@pytest.mark.core
def test_Sequential_CASE_create_AND_subtype_error():
    with pytest.raises(RuntimeError, match=r'object at \d+ position is not subtype of Apply'):
        transform = E.Sequential([
            E.TimeReverse(always_apply=True),
            object()
        ], always_apply=True)

@pytest.mark.core
def test_Sequential_CASE_call_AND_no_transfroms():
    ecg = np.ones((12, 5000))

    transform = E.Sequential([
    ], always_apply=True)

    transformed = transform(ecg=ecg)
    tecg = transformed['ecg']

    assert pytest.approx(tecg) == ecg

@pytest.mark.core
def test_Sequential_CASE_call_AND_one_flip():
    ecg = np.random.random((12, 5000))

    transform = E.Sequential([
        E.TimeReverse(always_apply=True)
    ], always_apply=True)

    transformed = transform(ecg=ecg)
    tecg = transformed['ecg']

    assert pytest.approx(tecg) != ecg

@pytest.mark.core
def test_Sequential_CASE_call_AND_double_flip():
    ecg = np.random.random((12, 5000))

    transform = E.Sequential([
        E.TimeReverse(always_apply=True),
        E.TimeReverse(always_apply=True)
    ], always_apply=True)

    transformed = transform(ecg=ecg)
    tecg = transformed['ecg']

    assert pytest.approx(tecg) == ecg

@pytest.mark.core
def test_OneOf_CASE_call_AND_no_transfroms():
    ecg = np.ones((12, 5000))

    transform = E.OneOf([
    ], always_apply=True)

    transformed = transform(ecg=ecg)
    tecg = transformed['ecg']

    assert pytest.approx(tecg) == ecg

@pytest.mark.core
def test_OneOf_CASE_call_AND_check_application():
    ecg = np.ones((12, 5000))

    transform = E.Sequential([
        E.TimeReverse(always_apply=True),
        E.OneOf([
            E.TimeReverse(),
            E.TimeReverse()
        ], always_apply=True)
    ], always_apply=True)

    transformed = transform(ecg=ecg)
    tecg = transformed['ecg']

    assert pytest.approx(tecg) == ecg
