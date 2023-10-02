import cv2
import numpy as np

import ecgmentations.augmentations.misc as M

def time_reverse(ecg):
    return np.flip(ecg, axis=0)

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

def amplitude_scale(ecg, scaling_factor):
    return ecg * scaling_factor

def time_cutout(ecg, cutouts, fill_value):
    ecg = np.array(ecg)

    for cutout_start, cutout_length in cutouts:
        ecg[cutout_start: cutout_start+cutout_length] = fill_value

    return ecg

def time_crop(ecg, left_bound, crop_length):
    length = ecg.shape[0]

    if length < crop_length:
        raise ValueError(
            'Requested crop length {crop_length} is '
            'larger than the ecg length {length}'.format(
                crop_length=crop_length, length=length
            )
        )

    t1 = int((length - crop_length + 1) * left_bound)
    t2 = t1 + crop_length

    return ecg[t1:t2]

def time_wrap(ecg, cells, ncells):
    length = ecg.shape[0]

    bounds = (length * cells).astype(np.int32)
    nbounds = (length * ncells).astype(np.int32)

    necg = np.zeros_like(ecg)

    for (left_bound, rigth_bound), (left_nbound, rigth_nbound) in zip(M.pairwise(bounds), M.pairwise(nbounds)):
        necg[left_nbound: rigth_nbound] = np.apply_along_axis(
            lambda ecg: np.interp(
                np.linspace(0, 1, rigth_nbound - left_nbound),
                np.linspace(0, 1, rigth_bound - left_bound),
                ecg
            ),
            axis=0,
            arr=ecg[left_bound: rigth_bound]
        )

    return necg
