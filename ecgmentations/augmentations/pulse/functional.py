import numpy as np
import scipy as sp

def add_sine_pulse(ecg, ecg_frequency, amplitude, frequency, phase):
    length = ecg.shape[0]

    t = np.linspace(0, length / ecg_frequency, length)
    ecg = ecg + amplitude * np.sin(2 * np.pi * frequency * t + phase)[:, None]

    return ecg

def add_square_pulse(ecg, ecg_frequency, amplitude, frequency, phase):
    length = ecg.shape[0]

    t = np.linspace(0, length / ecg_frequency, length)
    ecg = ecg + amplitude * sp.signal.square(2 * np.pi * frequency * t + phase)[:, None]

    return ecg
