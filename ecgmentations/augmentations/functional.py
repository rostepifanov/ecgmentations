import numpy as np

import ecgmentations.core.constants as C
import ecgmentations.augmentations.time.functional as TF

def apply_along_dim(data, func, dim):
    """Apply the same transformation along the dim
    """
    data = np.moveaxis(data, dim, 0)

    data = np.stack([*map(
        func,
        data,
    )], axis=dim)

    return data

def amplitude_invert(ecg):
    return np.negative(ecg)

def channel_shuffle(ecg, channel_order):
    ecg = ecg[:, channel_order]

    return np.require(ecg, requirements=['C_CONTIGUOUS'])

def channel_dropout(ecg, channels_to_drop, fill_value):
    ecg = np.copy(ecg)
    ecg[:, channels_to_drop] = fill_value

    return ecg

def add(ecg, other):
    if len(ecg.shape) > len(other.shape):
        other = np.expand_dims(other, axis=C.CHANNEL_DIM)

    return np.add(ecg, other, dtype=ecg.dtype)

def conv(ecg, kernel, border_mode, fill_value):
    pad_width = kernel.size // 2

    ecg = TF.pad(ecg, pad_width, pad_width, border_mode, fill_value)

    func = lambda arr: np.correlate(arr, kernel, mode='valid').astype(arr.dtype)

    if len(ecg.shape) == C.NUM_MULTI_CHANNEL_DIMENSIONS:
        ecg = apply_along_dim(ecg, func, C.CHANNEL_DIM)
    else:
        ecg = func(ecg)

    return np.require(ecg, requirements=['C_CONTIGUOUS'])

def multiply(ecg, factor):
    return np.multiply(ecg, factor, dtype=ecg.dtype)
