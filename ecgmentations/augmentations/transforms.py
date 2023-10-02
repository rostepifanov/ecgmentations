import cv2
import numpy as np
import ecgmentations.augmentations.misc as M
import ecgmentations.augmentations.functional as F

from ecgmentations.core.transforms import EcgOnlyTransform, DualTransform

class TimeReverse(DualTransform):
    """Reverse the input ecg.
    """
    def apply(self, ecg, **params):
        return F.time_reverse(ecg)

    def get_transform_init_args_names(self):
        return tuple()

class AmplitudeInvert(EcgOnlyTransform):
    """Invert the input ecg.
    """
    def apply(self, ecg, **params):
        return F.amplitude_invert(ecg)

    def get_transform_init_args_names(self):
        return tuple()

class ChannelShuffle(EcgOnlyTransform):
    """Randomly rearrange channels of the input ecg.
    """
    def apply(self, ecg, channel_order, **params):
        return F.channel_shuffle(ecg, channel_order)

    @property
    def targets_as_params(self):
        return ['ecg']

    def get_params_dependent_on_targets(self, params):
        channel_order = np.arange(params['ecg'].shape[-1])
        np.random.shuffle(channel_order)

        return {'channel_order': channel_order}

    def get_transform_init_args_names(self):
        return ()

class ChannelDropout(EcgOnlyTransform):
    """Randomly drop channels in the input ecg.
    """
    def __init__(
            self,
            channel_drop_range=(1, 1),
            fill_value=0,
            always_apply=False,
            p=0.5
        ):
        """
            :args:
                channel_drop_range ((int, int)): range for select the number of dropping channels
                fill_value (int) : fill value for dropped channels
        """
        super(ChannelDropout, self).__init__(always_apply, p)

        self.channel_drop_range = M.prepare_int_asymrange(channel_drop_range, 'channel_drop_range',1)

        self.min_drop_channels = channel_drop_range[0]
        self.max_drop_channels = channel_drop_range[1]

        self.fill_value = M.prepare_float(fill_value, 'fill_value')

    def apply(self, ecg, channels_to_drop, **params):
        return F.channel_dropout(ecg, channels_to_drop, self.fill_value)

    @property
    def targets_as_params(self):
        return ['ecg']

    def get_params_dependent_on_targets(self, params):
        num_channels = params['ecg'].shape[-1]

        if num_channels == 1:
            raise NotImplementedError('Ecg has one channel. ChannelDropout is not defined.')

        if not ( self.max_drop_channels < num_channels ):
            raise ValueError('Can not drop all channels in ChannelDropout.')

        num_drop_channels = np.random.randint(low=self.min_drop_channels, high=self.max_drop_channels + 1)
        channels_to_drop = np.random.choice(num_channels, size=num_drop_channels)

        return {'channels_to_drop': channels_to_drop}

    def get_transform_init_args_names(self):
        return ('channel_drop_range', 'fill_value')

class GaussNoise(EcgOnlyTransform):
    """Randomly add gaussian noise to the input ecg.
    """
    def __init__(
            self,
            mean=0.,
            variance=0.01,
            per_channel=True,
            always_apply=False,
            p=0.5
        ):
        """
            :NOTE:
                default values is taken for paper "Self-supervised representation learning from 12-lead ECG data"

            :args:
                mean (float): mean of gaussian noise
                variance (float): variance of gaussian noise
                per_channel (bool) : if set to True, noise will be sampled for each channel independently
        """
        super(GaussNoise, self).__init__(always_apply, p)

        self.mean = M.prepare_float(mean, 'mean')
        self.variance = M.prepare_float(variance, 'variance')

        if variance < 0.:
            raise ValueError('Variance should be non negative.')

        self.per_channel = per_channel

    def apply(self, ecg, gauss, **params):
        return F.gauss_noise(ecg, gauss)

    @property
    def targets_as_params(self):
        return ['ecg']

    def get_params_dependent_on_targets(self, params):
        if self.per_channel:
            gauss = np.random.normal(self.mean, self.variance**0.5, params['ecg'].shape)
        else:
            gauss = np.random.normal(self.mean, self.variance**0.5, params['ecg'].shape[:-1])
            gauss = np.expand_dims(gauss, axis=-1)

        return {'gauss': gauss}

    def get_transform_init_args_names(self):
        return ('mean', 'variance', 'per_channel')

class Blur(EcgOnlyTransform):
    """Blur the input ecg.
    """
    def __init__(
            self,
            kernel_size_range=(3, 5),
            always_apply=False,
            p=0.5
        ):
        """
            :args:
                kernel_size_range ((int, int)): range for select kernel size of blur filter
        """
        super(Blur, self).__init__(always_apply, p)

        self.kernel_size_range = M.prepare_int_asymrange(kernel_size_range, 'kernel_size_range', 0)

        self.min_kernel_size = kernel_size_range[0]
        self.max_kernel_size = kernel_size_range[1]

        if self.min_kernel_size % 2 == 0 or self.max_kernel_size % 2 == 0:
            raise ValueError('Invalid range borders. Must be odd, but got: {}.'.format(kernel_size_range))

    def apply(self, ecg, kernel, **params):
        return F.conv(ecg, kernel, cv2.BORDER_CONSTANT, 0)

    def get_params(self):
        kernel_size = 2 * np.random.randint(self.min_kernel_size // 2, self.max_kernel_size // 2 + 1) + 1
        kernel = np.ones(kernel_size) / kernel_size

        return {'kernel': kernel}

    def get_transform_init_args_names(self):
        return ('kernel_size_range', )

class GaussBlur(EcgOnlyTransform):
    """Blur by gaussian the input ecg.
    """
    def __init__(
            self,
            variance=1.,
            kernel_size_range=(5, 5),
            always_apply=False,
            p=0.5
        ):
        """
            :NOTE:
                transformation is similar to gaussian blur in paper "Self-supervised representation learning from 12-lead ECG data"
                with variance equals one and kernel size equals

                code kernel is about of (0.05, 0.25, 0.40, 0.25, 0.05)
                paper kernel is (0.10, 0.20, 0.40, 0.20, 0.10)

            :args:
                variance (float): variance of gaussian kernel
                kernel_size_range ((int, int)): range for select kernel size of blur filter
        """
        super(GaussBlur, self).__init__(always_apply, p)

        if variance < 0.:
            raise ValueError('Variance should be non negative.')

        self.variance = M.prepare_float(variance, 'variance')

        self.kernel_size_range = M.prepare_int_asymrange(kernel_size_range, 'kernel_size_range', 0)

        self.min_kernel_size = kernel_size_range[0]
        self.max_kernel_size = kernel_size_range[1]

        if self.min_kernel_size % 2 == 0 or self.max_kernel_size % 2 == 0:
            raise ValueError('Invalid range borders. Must be odd, but got: {}.'.format(kernel_size_range))

    def apply(self, ecg, kernel, **params):
        return F.conv(ecg, kernel, cv2.BORDER_CONSTANT, 0)

    def get_params(self):
        kernel_size = 2 * np.random.randint(self.min_kernel_size // 2, self.max_kernel_size // 2 + 1) + 1

        kernel = np.exp(-0.5 * np.square(np.arange(-kernel_size, kernel_size+1)) / self.variance)
        kernel = kernel / np.sum(kernel)

        return {'kernel': kernel}

    def get_transform_init_args_names(self):
        return ('variance', 'kernel_size_range')

class AmplitudeScale(EcgOnlyTransform):
    """Scale amplitude of the input ecg.
    """
    def __init__(
            self,
            scaling_range=(-0.05, 0.05),
            always_apply=False,
            p=0.5
        ):
        """
            :args:
                scaling_range ((float, float)): range for selecting scaling factor
        """
        super(AmplitudeScale, self).__init__(always_apply, p)

        self.scaling_range = M.prepare_float_symrange(scaling_range, 'scaling_range')

        self.min_scaling_range = self.scaling_range[0]
        self.max_scaling_range = self.scaling_range[1]

    def apply(self, ecg, scaling_factor, **params):
        return F.amplitude_scale(ecg, scaling_factor)

    def get_params(self):
        scaling_factor = 1 + np.random.uniform(self.min_scaling_range, self.max_scaling_range)

        return {'scaling_factor': scaling_factor}

    def get_transform_init_args_names(self):
        return ('scaling_range', )

class TimeCutout(DualTransform):
    """Randomly cutout time ranges in the input ecg.
    """
    def __init__(
            self,
            num_ranges=(1, 5),
            length_range=(0, 50),
            fill_value=0.,
            mask_fill_value=None,
            always_apply=False,
            p=0.5,
        ):
        """
            :args:
                num_ranges ((int, int)): number of cutout ranges
                length_range ((int, int)): range for selecting cutout length
                fill_value (float): value to fill cutouted ranges in the input ecg
                mask_fill_value (int, None): value to fill cutouted ranges in the mask. if value is None, mask is not affected
        """
        super(TimeCutout, self).__init__(always_apply, p)

        self.num_ranges = M.prepare_int_asymrange(num_ranges, 'num_ranges', 0)

        self.min_num_ranges = num_ranges[0]
        self.max_num_ranges = num_ranges[1]

        self.length_range = M.prepare_int_asymrange(length_range, 'length_range', 0)

        self.min_length_range = length_range[0]
        self.max_length_range = length_range[1]

        self.fill_value = M.prepare_float(fill_value, 'fill_value')
        self.mask_fill_value = mask_fill_value

    def apply(self, ecg, cutouts, **params):
        return F.time_cutout(ecg, cutouts, self.fill_value)

    def apply_to_mask(self, mask, cutouts, **params):
        if self.mask_fill_value is None:
            return mask
        else:
            return F.time_cutout(ecg, cutouts, self.mask_fill_value)

    @property
    def targets_as_params(self):
        return ['ecg']

    def get_params_dependent_on_targets(self, params):
        length = params['ecg'].shape[0]

        cutouts = []

        for _ in range(np.random.randint(self.min_num_ranges, self.max_num_ranges + 1)):
            cutout_length = np.random.randint(self.min_length_range, self.max_length_range + 1)
            cutout_start = np.random.randint(0, length - cutout_length + 1)

            cutouts.append((cutout_start, cutout_length))

        return {'cutouts': cutouts}

    def get_transform_init_args_names(self):
        return ('num_ranges', 'length_range', 'fill_value', 'mask_fill_value')

class RandomTimeCrop(DualTransform):
    """Randomly crop time region from the input ecg.
    """
    def __init__(
            self,
            length=5000,
            always_apply=False,
            p=0.5,
        ):
        """
            :args:
                length (int): the length of cropped region
        """
        super(RandomTimeCrop, self).__init__(always_apply, p)

        self.length = length

        if length < 0:
            raise ValueError('Length should be non negative.')

    def apply(self, ecg, left_bound, **params):
        return F.time_crop(ecg, left_bound, self.length)

    def get_params(self):
        return {'left_bound': np.random.random()}

    def get_transform_init_args_names(self):
        return ('length', )

class RandomTimeWrap(DualTransform):
    """Randomly stretch and squeeze contiguous segments of the input ecg
    """

    def __init__(
            self,
            num_steps=5,
            wrap_limit=0.05,
            always_apply=False,
            p=0.5,
        ):
        """
            :args:
                num_steps (int): count of grid cells on the ecg
                wrap_limit (float): limit of stretching or squeezing
        """
        super(RandomTimeWrap, self).__init__(always_apply, p)

        self.num_steps = num_steps

        if num_steps < 0:
            raise ValueError('Number of steps should be non negative.')

        self.wrap_limit = M.prepare_float(wrap_limit, 'wrap_limit')

        if wrap_limit < 0.:
            raise ValueError('Wrap limit should be non negative.')

    def apply(self, ecg, cells, ncells, **params):
        return F.time_wrap(ecg, cells, ncells)

    def get_params(self):
        cells = np.linspace(0, 1, self.num_steps+1)
        ncells = np.linspace(0, 1, self.num_steps+1)

        if self.num_steps > 1:
            directions = np.random.choice([-1, 1], size=self.num_steps-1)
            shifts = np.random.random(size=self.num_steps-1) * self.wrap_limit * 0.5

            ncells[1:-1] += shifts * directions / (self.num_steps + 1)

        return {'cells': cells, 'ncells': ncells}

    def get_transform_init_args_names(self):
        return ('num_steps', 'wrap_limit')
