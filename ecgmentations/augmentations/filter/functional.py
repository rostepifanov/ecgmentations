import numpy as np
import scipy as sp

def lowpass_filter(ecg, ecg_frequency, frequency_cutoff):
    params = sp.signal.butter(3, frequency_cutoff, 'low', analog=False, fs=ecg_frequency)
    ecg = sp.signal.lfilter(*params, ecg)

    return ecg

def highpass_filter(ecg, ecg_frequency, frequency_cutoff):
    params = sp.signal.butter(3, frequency_cutoff, 'high', analog=False, fs=ecg_frequency)
    ecg = sp.signal.lfilter(*params, ecg)

    return ecg
