import pytest

import numpy as np
import ecgmentations.augmentations.functional as F

def test_reverse_CASE_default():
    ecg = np.array([[1, 2, 3, 4, 5, 6], ]).T

    output = F.reverse(ecg)
    expected = np.array([[6, 5, 4, 3, 2, 1], ]).T

    assert pytest.approx(output) == expected

def test_invert_CASE_default():
    ecg = np.array([[1, 2, 3, 4, 5, 6], ]).T

    output = F.invert(ecg)
    expected = np.array([[-1, -2, -3, -4, -5, -6], ]).T

    assert pytest.approx(output) == expected

def test_channel_shuffle_CASE_direct_order():
    ecg = np.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1] ]).T

    output = F.channel_shuffle(ecg, (0, 1))
    expected = np.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1] ]).T

    assert pytest.approx(output) == expected

def test_channel_shuffle_CASE_inverse_order():
    ecg = np.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1] ]).T

    output = F.channel_shuffle(ecg, (1, 0))
    expected = np.array([[6, 5, 4, 3, 2, 1], [1, 2, 3, 4, 5, 6] ]).T

    assert pytest.approx(output) == expected

def test_channel_dropout_CASE_default():
    ecg = np.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1] ]).T

    output = F.channel_dropout(ecg, (0, ), 0)
    expected = np.array([[0, 0, 0, 0, 0, 0], [6, 5, 4, 3, 2, 1] ]).T

    assert pytest.approx(output) == expected
