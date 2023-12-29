import cv2
import numpy as np

import ecgmentations.augmentations.time.functional as F

def amplitude_invert(ecg):
    return np.negative(ecg)

def channel_shuffle(ecg, channel_order):
    return ecg[:, channel_order]

def channel_dropout(ecg, channels_to_drop, fill_value):
    ecg = np.array(ecg)
    ecg[:, channels_to_drop] = fill_value

    return ecg

def gauss_noise(ecg, gauss):
    return ecg + gauss

def conv(ecg, kernel, border_mode, fill_value):
    pad_width = kernel.size // 2

    ecg = F.pad(ecg, pad_width, pad_width, border_mode, fill_value)

    ecg = np.apply_along_axis(
        lambda ecg: np.correlate(ecg, kernel, mode='valid'),
        axis=0,
        arr=ecg
    )

    return ecg

def amplitude_scale(ecg, scaling_factor):
    return ecg * scaling_factor
