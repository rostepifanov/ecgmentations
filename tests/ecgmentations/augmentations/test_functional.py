import pytest

import numpy as np
import ecgmentations.augmentations.functional as F

def test_time_reverse_CASE_default():
    ecg = np.array([[1, 2, 3, 4, 5, 6], ]).T

    output = F.time_reverse(ecg)
    expected = np.array([[6, 5, 4, 3, 2, 1], ]).T

    assert pytest.approx(output) == expected

def test_amplitude_invert_CASE_default():
    ecg = np.array([[1, 2, 3, 4, 5, 6], ]).T

    output = F.amplitude_invert(ecg)
    expected = np.array([[-1, -2, -3, -4, -5, -6], ]).T

    assert pytest.approx(output) == expected

def test_channel_shuffle_CASE_direct_order():
    ecg = np.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]).T

    output = F.channel_shuffle(ecg, (0, 1))
    expected = np.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]).T

    assert pytest.approx(output) == expected

def test_channel_shuffle_CASE_inverse_order():
    ecg = np.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]).T

    output = F.channel_shuffle(ecg, (1, 0))
    expected = np.array([[6, 5, 4, 3, 2, 1], [1, 2, 3, 4, 5, 6]]).T

    assert pytest.approx(output) == expected

def test_channel_dropout_CASE_default():
    ecg = np.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]).T

    channels_to_drop = (0, )
    fill_value = 0

    output = F.channel_dropout(ecg, channels_to_drop, fill_value)
    expected = np.array([[0, 0, 0, 0, 0, 0], [6, 5, 4, 3, 2, 1]]).T

    assert pytest.approx(output) == expected

def test_time_cutout_CASE_default():
    ecg = np.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]).T

    cutouts = [(0, 2)]
    fill_value = 0

    output = F.time_cutout(ecg, cutouts, fill_value)
    expected = np.array([[0, 0, 3, 4, 5, 6], [0, 0, 4, 3, 2, 1]]).T

    assert pytest.approx(output) == expected

def test_time_crop_CASE_default():
    ecg = np.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]).T

    left_bound = 1 / 4
    crop_length = 2

    output = F.time_crop(ecg, left_bound, crop_length)
    expected = np.array([[2, 3], [5, 4]]).T

    assert pytest.approx(output) == expected

def test_time_crop_CASE_equal_legth():
    ecg = np.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]).T

    left_bound = 1 / 6
    crop_length = 6

    output = F.time_crop(ecg, left_bound, crop_length)
    expected = np.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]).T

    assert pytest.approx(output) == expected

def test_time_crop_CASE_large_length():
    ecg = np.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]).T

    left_bound = 1 / 6
    crop_length = 8

    with pytest.raises(ValueError):
        F.time_crop(ecg, left_bound, crop_length)
