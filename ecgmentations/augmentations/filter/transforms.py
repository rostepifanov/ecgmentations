import numpy as np

import ecgmentations.augmentations.misc as M
import ecgmentations.augmentations.filter.functional as F

from ecgmentations.core.transforms import EcgOnlyTransform

class LowPassFilter(EcgOnlyTransform):
    """Apply low-pass filter to the input ecg.
    """
    def __init__(
            self,
            ecg_frequency=500.,
            frequency_cutoff=47.,
            always_apply=False,
            p=1.0,
        ):
        """
            :NOTE:
                frequency_cutoff:
                    47 Hz (default) "Effective Data Augmentation, Filters, and Automation Techniques for Automatic 12-Lead ECG Classification Using Deep Residual "

            :args:
                ecg_frequency: float
                    frequency of the input ecg
                frequency_cutoff: float
                    cutoff frequency for filter
        """
        super(LowPassFilter, self).__init__(always_apply, p)

        self.ecg_frequency = M.prepare_non_negative_float(ecg_frequency, 'ecg_frequency')
        self.frequency_cutoff = M.prepare_non_negative_float(frequency_cutoff, 'ecg_frequency')

    def apply(self, ecg, **params):
        return F.lowpass_filter(ecg, self.ecg_frequency, self.frequency_cutoff)

    def get_transform_init_args_names(self):
        return ('ecg_frequency', 'frequency_cutoff')

class HighPassFilter(EcgOnlyTransform):
    """Apply high-pass filter to the input ecg.
    """
    def __init__(
            self,
            ecg_frequency=500.,
            frequency_cutoff=0.5,
            always_apply=False,
            p=1.0,
        ):
        """
            :NOTE:
                frequency_cutoff:
                    0.5 Hz (default) "Effective Data Augmentation, Filters, and Automation Techniques for Automatic 12-Lead ECG Classification Using Deep Residual "

            :args:
                ecg_frequency: float
                    frequency of the input ecg
                frequency_cutoff: float
                    cutoff frequency for filter
        """
        super(HighPassFilter, self).__init__(always_apply, p)

        self.ecg_frequency = M.prepare_non_negative_float(ecg_frequency, 'ecg_frequency')
        self.frequency_cutoff = M.prepare_non_negative_float(frequency_cutoff, 'ecg_frequency')

    def apply(self, ecg, **params):
        return F.highpass_filter(ecg, self.ecg_frequency, self.frequency_cutoff)

    def get_transform_init_args_names(self):
        return ('ecg_frequency', 'frequency_cutoff')
