import numpy as np
import scipy as sp

from itertools import tee

import ecgmentations.core.enum as E
import ecgmentations.core.constants as C
import ecgmentations.augmentations.functional as F

def time_reverse(ecg):
    """Reverse spatial dim
    """
    ecg = np.flip(ecg, axis=C.SPATIAL_DIM)

    return np.require(ecg, requirements=['C_CONTIGUOUS'])

def time_shift(ecg, shift, border_mode, fill_value):
    length = ecg.shape[C.SPATIAL_DIM]

    pad_ = int(length*shift)

    if pad_ > 0:
        ecg = pad(ecg, pad_, 0, border_mode, fill_value)
        ecg = ecg[:-pad_]
    elif pad_ < 0:
        ecg = pad(ecg, 0, -pad_, border_mode, fill_value)
        ecg = ecg[-pad_:]

    return np.require(ecg, requirements=['C_CONTIGUOUS'])

def time_segment_swap(ecg, segment_order):
    shape = ecg.shape
    length = shape[C.SPATIAL_DIM]

    time_point_order = np.arange(length)

    num_segments = len(segment_order)
    time_point_order = np.array(
        np.array_split(time_point_order, num_segments)
    )[segment_order]

    ecg = ecg[time_point_order]

    if len(shape) == C.NUM_MULTI_CHANNEL_DIMENSIONS:
        ecg.shape = (length, -1)
    else:
        ecg.shape = (length, )

    return ecg

def pairwise(iterable):
    first, second = tee(iterable)
    next(second, None)
    yield from zip(first, second)

def time_wrap(ecg, cells, ncells):
    length = ecg.shape[C.SPATIAL_DIM]

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
            axis=C.SPATIAL_DIM,
            arr=ecg[left_bound: rigth_bound]
        )

    return necg

def time_cutout(ecg, cutouts, fill_value):
    ecg = np.copy(ecg)

    for cutout_start, cutout_length in cutouts:
        ecg[cutout_start: cutout_start+cutout_length] = fill_value

    return ecg

def time_crop(ecg, left_bound, crop_length):
    length = ecg.shape[C.SPATIAL_DIM]

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
    kwargs = dict()

    if border_mode == E.BorderType.CONSTANT:
        kwargs['constant_values'] = fill_value

    func = lambda arr: np.pad(
        arr,
        pad_width=(left_pad, rigth_pad),
        mode=C.MAP_BORDER_TYPE_TO_NUMPY[border_mode],
        **kwargs
    )

    if len(ecg.shape) == C.NUM_MULTI_CHANNEL_DIMENSIONS:
        ecg = F.apply_along_dim(ecg, func, C.CHANNEL_DIM)
    else:
        ecg = func(ecg)

    return ecg

def pooling(ecg, reduction, kernel_size, border_mode, fill_value):
    if reduction == E.ReductionType.MIN:
        filter = sp.ndimage.minimum_filter
    elif reduction == E.ReductionType.MEAN:
        filter = sp.ndimage.uniform_filter
    elif reduction == E.ReductionType.MAX:
        filter = sp.ndimage.maximum_filter
    elif reduction == E.ReductionType.MEDIAN:
        filter = sp.ndimage.median_filter
    else:
        raise ValueError('Get invalide reduction: {}'.format(reduction))

    pad_width = kernel_size // 2

    ecg = pad(ecg, pad_width, pad_width, border_mode, fill_value)
    ecg = filter(ecg, size=kernel_size)

    ecg = ecg[pad_width:-pad_width]

    return ecg
