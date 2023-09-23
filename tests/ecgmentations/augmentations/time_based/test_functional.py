import pytest

import numpy as np
import ecgmentations.augmentations.time_based.functional as F

def test_flip_CASE_default():
    ecg = np.array([[1, 2, 3, 4, 5, 6], ])

    output = F.flip(ecg)
    expected = np.array([[6, 5, 4, 3, 2, 1], ])

    assert pytest.approx(output) == expected
