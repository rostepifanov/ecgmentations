import cv2
import numpy as np

def reverse(ecg):
    return np.flip(ecg, axis=0)

def invert(ecg):
    return np.negative(ecg)

def channel_shuffle(ecg, channel_order):
    return ecg[:, channel_order]

def channel_dropout(ecg, channels_to_drop, fill_value):
    ecg = np.array(ecg)
    ecg[:, channels_to_drop] = fill_value

    return ecg

def gauss_noise(ecg, gauss):
    return ecg + gauss

def conv(ecg, kernel, border_mode, border_value):
    pad_width = kernel.size // 2

    if border_mode == cv2.BORDER_CONSTANT:
        border_mode = 'constant'
    elif border_mode == cv2.BORDER_REPLICATE:
        border_mode = 'edge'
    else:
        raise ValueError('Get invalide border_mode: {}'.format(border_mode))

    ecg = np.apply_along_axis(
        lambda ecg: np.pad(ecg, pad_width=pad_width, mode=border_mode, constant_values=border_value),
        axis=0,
        arr=ecg
    )

    ecg = np.apply_along_axis(
        lambda ecg: np.correlate(ecg, kernel, mode='valid'),
        axis=0,
        arr=ecg
    )

    return ecg
