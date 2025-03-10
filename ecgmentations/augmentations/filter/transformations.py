import ecgmentations.augmentations.misc as M
import ecgmentations.augmentations.filter.functional as F

from ecgmentations.core.augmentation import EcgOnlyAugmentation

class LowPassFilter(EcgOnlyAugmentation):
    """Apply low-pass filter to the input ecg.
    """
    def __init__(
            self,
            ecg_frequency=500.,
            cutoff_frequency=47.,
            always_apply=False,
            p=1.0,
        ):
        """
            :NOTE:
                cutoff_frequency:
                    47 Hz (default) "Effective Data Augmentation, Filters, and Automation Techniques for Automatic 12-Lead ECG Classification Using Deep Residual"

            :args:
                ecg_frequency: float
                    frequency of the input ecg
                cutoff_frequency: float
                    cutoff frequency for filter
        """
        super(LowPassFilter, self).__init__(always_apply, p)

        self.ecg_frequency = M.prepare_non_negative_float(ecg_frequency, 'ecg_frequency')
        self.cutoff_frequency = M.prepare_non_negative_float(cutoff_frequency, 'cutoff_frequency')

    def apply(self, ecg, **params):
        return F.lowpass_filter(ecg, self.ecg_frequency, self.cutoff_frequency)

    def get_transform_init_args_names(self):
        return ('ecg_frequency', 'cutoff_frequency')

class HighPassFilter(EcgOnlyAugmentation):
    """Apply high-pass filter to the input ecg.
    """
    def __init__(
            self,
            ecg_frequency=500.,
            cutoff_frequency=0.5,
            always_apply=False,
            p=1.0,
        ):
        """
            :NOTE:
                cutoff_frequency:
                    0.5 Hz (default) "Effective Data Augmentation, Filters, and Automation Techniques for Automatic 12-Lead ECG Classification Using Deep Residual"

            :args:
                ecg_frequency: float
                    frequency of the input ecg
                cutoff_frequency: float
                    cutoff frequency for filter
        """
        super(HighPassFilter, self).__init__(always_apply, p)

        self.ecg_frequency = M.prepare_non_negative_float(ecg_frequency, 'ecg_frequency')
        self.cutoff_frequency = M.prepare_non_negative_float(cutoff_frequency, 'cutoff_frequency')

    def apply(self, ecg, **params):
        return F.highpass_filter(ecg, self.ecg_frequency, self.cutoff_frequency)

    def get_transform_init_args_names(self):
        return ('ecg_frequency', 'cutoff_frequency')

class BandPassFilter(EcgOnlyAugmentation):
    """Apply band-pass filter to the input ecg.
    """
    def __init__(
            self,
            ecg_frequency=500.,
            cutoff_frequencies=(0.5, 47.),
            always_apply=False,
            p=1.0,
        ):
        """
            :NOTE:
                cutoff_frequencies:
                    see params of LowPassFilter and HighPassFilter

            :args:
                ecg_frequency: float
                    frequency of the input ecg
                cutoff_frequencies: tuple of float
                    cutoff frequencies for filter
        """
        super(BandPassFilter, self).__init__(always_apply, p)

        self.ecg_frequency = M.prepare_non_negative_float(ecg_frequency, 'ecg_frequency')
        self.cutoff_frequencies = M.prepare_float_asymrange(cutoff_frequencies, 'cutoff_frequencies', low=0.)

    def apply(self, ecg, **params):
        return F.bandpass_filter(ecg, self.ecg_frequency, self.cutoff_frequencies)

    def get_transform_init_args_names(self):
        return ('ecg_frequency', 'cutoff_frequencies')

class SigmoidCompression(EcgOnlyAugmentation):
    """Apply sigmoid compression to the input ecg.
    """
    def __init__(
            self,
            always_apply=False,
            p=1.0,
        ):
        super(SigmoidCompression, self).__init__(always_apply, p)

    def apply(self, ecg, **params):
        return F.sigmoid_compression(ecg)

    def get_transform_init_args_names(self):
        return tuple()
