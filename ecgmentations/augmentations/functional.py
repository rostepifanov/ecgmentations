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
