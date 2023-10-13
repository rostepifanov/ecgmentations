import pytest

import cv2
import numpy as np

import ecgmentations.augmentations.time.functional as F

def test_time_reverse_CASE_default():
    ecg = np.array([[1, 2, 3, 4, 5, 6], ]).T

    output = F.time_reverse(ecg)
    expected = np.array([[6, 5, 4, 3, 2, 1], ]).T

    assert pytest.approx(output) == expected

def test_time_shift_CASE_zero_shift():
    ecg = np.array([[1, 2, 3, 4, 5, 6], ]).T

    output = F.time_shift(ecg, 0., cv2.BORDER_CONSTANT, 0.)
    expected = np.array([[1, 2, 3, 4, 5, 6], ]).T

    assert pytest.approx(output) == expected

def test_time_shift_CASE_positive_shift():
    ecg = np.array([[1, 2, 3, 4, 5, 6], ]).T

    output = F.time_shift(ecg, 0.167, cv2.BORDER_CONSTANT, 0.)
    expected = np.array([[0, 1, 2, 3, 4, 5], ]).T

    assert pytest.approx(output) == expected

def test_time_shift_CASE_negative_shift():
    ecg = np.array([[1, 2, 3, 4, 5, 6], ]).T

    output = F.time_shift(ecg, -0.167, cv2.BORDER_CONSTANT, 0.)
    expected = np.array([[2, 3, 4, 5, 6, 0], ]).T

    assert pytest.approx(output) == expected

def time_segment_swap_CASE_default():
    ecg = np.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]).T

    segment_order = [2, 0, 1]

    output = F.time_segment_swap(ecg, segment_order)
    expected = np.array([[5, 6, 1, 2, 3, 4], [2, 1, 6, 5, 4, 3]]).T

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

def test_pad_CASE_boeder_constant():
    ecg = np.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]).T

    left_pad = 2
    rigth_pad = 2

    output = F.pad(ecg, left_pad, rigth_pad, cv2.BORDER_CONSTANT, 0.)
    expected = np.array([[0, 0, 1, 2, 3, 4, 5, 6, 0, 0], [0, 0, 6, 5, 4, 3, 2, 1, 0, 0]]).T

    assert pytest.approx(output) == expected

def test_pad_CASE_border_replicate():
    ecg = np.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]).T

    left_pad = 2
    rigth_pad = 2

    output = F.pad(ecg, left_pad, rigth_pad, cv2.BORDER_REPLICATE, None)
    expected = np.array([[1, 1, 1, 2, 3, 4, 5, 6, 6, 6], [6, 6, 6, 5, 4, 3, 2, 1, 1, 1]]).T

    assert pytest.approx(output) == expected
