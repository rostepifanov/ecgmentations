import numpy as np

import ecgmentations.augmentations.misc as M
import ecgmentations.augmentations.pulse.functional as F

from ecgmentations.core.transforms import EcgOnlyTransform

class SinePulse(EcgOnlyTransform):
    """Add sine pulse to the input ecg.
    """
    def __init__(
            self,
            ecg_frequency=500.,
            pulse_frequency_range=(0., 1.),
            amplitude_limit=1.,
            always_apply=False,
            p=0.5
        ):
        """
            :NOTE:
                amplitude_limit:
                    1 mV (default) "RandECG: Data Augmentation for Deep Neural Network Based ECG Classification"

                pulse_frequency_range:
                    0 - 1 Hz (default) "RandECG: Data Augmentation for Deep Neural Network Based ECG Classification"

            :args:
                ecg_frequency (float): frequency of the input ecg
                pulse_frequency_range ((float, float)): range of pulse frequency
                amplitude_limit (float): limit of pulse amplitude
        """
        super(SinePulse, self).__init__(always_apply, p)

        self.ecg_frequency = M.prepare_non_negative_float(ecg_frequency, 'ecg_frequency')
        self.pulse_frequency_range = M.prepare_float_asymrange(pulse_frequency_range, 'pulse_frequency_range', 0.)

        self.pulse_frequency_min = self.pulse_frequency_range[0]
        self.pulse_frequency_max = self.pulse_frequency_range[1]
        self.pulse_frequency_delta = self.pulse_frequency_max - self.pulse_frequency_min

        self.amplitude_limit = M.prepare_non_negative_float(amplitude_limit, 'amplitude_limit')

    def apply(self, ecg, amplitude, frequency, phase, **params):
        return F.add_sine_pulse(ecg, self.ecg_frequency, amplitude, frequency, phase)

    def get_params(self):
        amplitude = np.random.random() * self.amplitude_limit
        frequency = np.random.random() * self.pulse_frequency_delta + self.pulse_frequency_min
        phase = np.random.random() * 2 * np.pi

        return {'amplitude': amplitude, 'frequency': frequency, 'phase': phase}

    def get_transform_init_args_names(self):
        return ('ecg_frequency', 'pulse_frequency_range', 'amplitude_limit')

class PowerlineNoise(SinePulse):
    """Add powerline noise to the input ecg.
    """
    def __init__(
            self,
            ecg_frequency=500.,
            powerline_frequency=50.,
            amplitude_limit=0.3,
            always_apply=False,
            p=0.5
        ):
        """
            :NOTE:
                powerline frequency:
                    50 Hz is for Europe
                    60 Hz is for USA or Asia

                amplitude_limit:
                    0.3 mV (default) "IIR digital filter design for powerline noise cancellation of ECG signal using arduino platform"
                    0.333 mV "A Comparison of the Noise Sensitivity of Nine QRS Detection Algorithms"

            :args:
                ecg_frequency (float): frequency of the input ecg
                powerline_frequency (float): frequency of powerline
                amplitude_limit (float): limit of noise amplitude
        """
        self.powerline_frequency = M.prepare_non_negative_float(powerline_frequency, 'powerline_frequency')
        powerline_frequency_range = (self.powerline_frequency, self.powerline_frequency)

        super(PowerlineNoise, self).__init__(ecg_frequency, powerline_frequency_range, amplitude_limit, always_apply, p)

    def get_transform_init_args_names(self):
        return ('ecg_frequency', 'powerline_frequency', 'amplitude_limit')

class RespirationNoise(SinePulse):
    """Add respiration noise to the input ecg
    """
    def __init__(
            self,
            ecg_frequency=500.,
            breathing_rate_range=(12, 18),
            amplitude_limit=1.,
            always_apply=False,
            p=0.5
        ):
        """
            :NOTE:
                breathing_rate:
                    0.333 Hz "A Comparison of the Noise Sensitivity of Nine QRS Detection Algorithms"
                    0.333 equals to 20 bpm

                amplitude_limit:
                    1 mV (default) "A Comparison of the Noise Sensitivity of Nine QRS Detection Algorithms"

            :args:
                ecg_frequency (float): frequency of the input ecg
                breathing_rate_range ((int, int)): breathing rate range in bpm
                amplitude_limit (float): limit of noise amplitude
        """
        self.breathing_rate_range = M.prepare_int_asymrange(breathing_rate_range, 'breathing_rate_range', 0)
        breathing_frequency_range = (breathing_rate_range[0] / 60, breathing_rate_range[1] / 60)

        super(RespirationNoise, self).__init__(ecg_frequency, breathing_frequency_range, amplitude_limit, always_apply, p)

    def get_transform_init_args_names(self):
        return ('ecg_frequency', 'breathing_rate_range', 'amplitude_limit')

class SquarePulse(EcgOnlyTransform):
    """Add square pulse to the input ecg.
    """
    def __init__(
            self,
            ecg_frequency=500.,
            pulse_frequency_range=(0., 5.),
            amplitude_limit=0.02,
            always_apply=False,
            p=0.5
        ):
        """
            :NOTE:
                amplitude_limit:
                    0.02 mV (default) "RandECG: Data Augmentation for Deep Neural Network Based ECG Classification"

                pulse_frequency_range:
                    0 - 5 Hz (default) "RandECG: Data Augmentation for Deep Neural Network Based ECG Classification"

            :args:
                ecg_frequency (float): frequency of the input ecg
                pulse_frequency_range (float): range of pulse frequency
                amplitude_limit (float): limit of pulse amplitude
        """
        super(SquarePulse, self).__init__(always_apply, p)

        self.ecg_frequency = M.prepare_non_negative_float(ecg_frequency, 'ecg_frequency')
        self.pulse_frequency_range = M.prepare_float_asymrange(pulse_frequency_range, 'pulse_frequency_range', 0.)

        self.pulse_frequency_min = self.pulse_frequency_range[0]
        self.pulse_frequency_max = self.pulse_frequency_range[1]
        self.pulse_frequency_delta = self.pulse_frequency_max - self.pulse_frequency_min

        self.amplitude_limit = M.prepare_non_negative_float(amplitude_limit, 'amplitude_limit')

    def apply(self, ecg, amplitude, frequency, phase, **params):
        return F.add_square_pulse(ecg, self.ecg_frequency, amplitude, frequency, phase)

    def get_params(self):
        amplitude = np.random.random() * self.amplitude_limit
        frequency = np.random.random() * self.pulse_frequency_delta + self.pulse_frequency_min
        phase = np.random.random() * 2 * np.pi

        return {'amplitude': amplitude, 'frequency': frequency, 'phase': phase}

    def get_transform_init_args_names(self):
        return ('ecg_frequency', 'pulse_frequency_range', 'amplitude_limit')
