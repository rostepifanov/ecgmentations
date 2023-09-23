import numpy as np

def flip(ecg):
    return np.flip(ecg, axis=0)

def invert(ecg):
    return np.negative(ecg)

def channel_shuffle(ecg, channel_order):
    return ecg[:, channel_order]
