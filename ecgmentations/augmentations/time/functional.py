import cv2
import numpy as np

from itertools import tee

def time_reverse(ecg):
    return np.flip(ecg, axis=0)

def time_shift(ecg, shift, border_mode, fill_value):
    ecg = np.array(ecg)
    length = ecg.shape[0]

    pad_ = int(length*shift)

    if pad_ > 0:
        ecg = pad(ecg, pad_, 0, border_mode, fill_value)
        ecg = ecg[:-pad_]
    elif pad_ < 0:
        ecg = pad(ecg, 0, -pad_, border_mode, fill_value)
        ecg = ecg[-pad_:]

    return ecg

def time_segment_swap(ecg, segment_order):
    length = ecg.shape[0]
    time_point_order = np.arange(length)

    num_segments = len(segment_order)
    time_point_order = np.array(
        np.array_split(time_point_order, num_segments)
    )[segment_order]

    ecg = ecg[time_point_order]
    ecg.shape = (length, -1)

    return ecg

def pairwise(iterable):
    first, second = tee(iterable)
    next(second, None)
    yield from zip(first, second)

def time_wrap(ecg, cells, ncells):
    length = ecg.shape[0]

    bounds = (length * cells).astype(np.int32)
    nbounds = (length * ncells).astype(np.int32)

    necg = np.zeros_like(ecg)

    for (left_bound, rigth_bound), (left_nbound, rigth_nbound) in zip(pairwise(bounds), pairwise(nbounds)):
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

    t1 = int((length - crop_length) * left_bound)
    t2 = t1 + crop_length

    return ecg[t1:t2]

def pad(ecg, left_pad, rigth_pad, border_mode, fill_value):
    if border_mode == cv2.BORDER_CONSTANT:
        border_mode = 'constant'
    elif border_mode == cv2.BORDER_REPLICATE:
        border_mode = 'edge'
    else:
        raise ValueError('Get invalide border_mode: {}'.format(border_mode))

    if border_mode == 'constant':
        kwargs = { 'constant_values': fill_value }
    else:
        kwargs = dict()

    ecg = np.apply_along_axis(
        lambda ecg: np.pad( ecg,
                            pad_width=(left_pad, rigth_pad),
                            mode=border_mode,
                            **kwargs ),
        axis=0,
        arr=ecg
    )

    return ecg
