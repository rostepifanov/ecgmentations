import numpy as np
import scipy as sp

import ecgmentations.core.constants as C

def add_sine_pulse(ecg, ecg_frequency, amplitude, frequency, phase):
    length = ecg.shape[C.SPATIAL_DIM]

    t = np.linspace(0, length / ecg_frequency, length)
    pulse = amplitude * np.sin(2 * np.pi * frequency * t + phase)

    if len(ecg.shape) == C.NUM_MULTI_CHANNEL_DIMENSIONS:
        pulse = np.expand_dims(pulse, axis=C.CHANNEL_DIM)

    return np.add(ecg, pulse, dtype=ecg.dtype)

def add_square_pulse(ecg, ecg_frequency, amplitude, frequency, phase):
    length = ecg.shape[C.SPATIAL_DIM]

    t = np.linspace(0, length / ecg_frequency, length)
    pulse = amplitude * sp.signal.square(2 * np.pi * frequency * t + phase)

    if len(ecg.shape) == C.NUM_MULTI_CHANNEL_DIMENSIONS:
        pulse = np.expand_dims(pulse, axis=C.CHANNEL_DIM)

    return np.add(ecg, pulse, dtype=ecg.dtype)
