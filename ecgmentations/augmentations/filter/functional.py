import scipy as sp

def lowpass_filter(ecg, ecg_frequency, cutoff_frequency):
    params = sp.signal.butter(3, cutoff_frequency, 'low', analog=False, fs=ecg_frequency)
    ecg = sp.signal.lfilter(*params, ecg).astype(ecg.dtype)

    return ecg

def highpass_filter(ecg, ecg_frequency, cutoff_frequency):
    params = sp.signal.butter(3, cutoff_frequency, 'high', analog=False, fs=ecg_frequency)
    ecg = sp.signal.lfilter(*params, ecg).astype(ecg.dtype)

    return ecg

def bandpass_filter(ecg, ecg_frequency, cutoff_frequencies):
    params = sp.signal.butter(3, cutoff_frequencies, 'bandpass', analog=False, fs=ecg_frequency)
    ecg = sp.signal.lfilter(*params, ecg).astype(ecg.dtype)

    return ecg

def sigmoid_compression(ecg):
    ecg = sp.special.expit(ecg)

    return ecg
