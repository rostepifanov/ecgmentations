import cv2
import numpy as np
import ecgmentations.augmentations.functional as F

from ecgmentations.core.transforms import EcgOnlyTransform, DualTransform

class Reverse(DualTransform):
    """Reverse the input ecg.
    """
    def apply(self, ecg, **params):
        return F.reverse(ecg)

    def get_transform_init_args_names(self):
        return tuple()

class Invert(EcgOnlyTransform):
    """Invert the input ecg.
    """
    def apply(self, ecg, **params):
        return F.invert(ecg)

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

        self.channel_drop_range = channel_drop_range

        self.min_channels = channel_drop_range[0]
        self.max_channels = channel_drop_range[1]

        if not (isinstance(channel_drop_range, tuple) and list(map(type, channel_drop_range)) == [int, int]):
            raise ValueError('Invalive type of channel_drop_range. Must be (int, int), but got {}'.format(channel_drop_range))

        if not 1 <= self.min_channels <= self.max_channels:
            raise ValueError('Invalid channel_drop_range. Got: {}'.format(channel_drop_range))

        self.fill_value = fill_value

    def apply(self, ecg, channels_to_drop, **params):
        return F.channel_dropout(ecg, channels_to_drop, self.fill_value)

    @property
    def targets_as_params(self):
        return ['ecg']

    def get_params_dependent_on_targets(self, params):
        num_channels = params['ecg'].shape[-1]

        if num_channels == 1:
            raise NotImplementedError('Ecg has one channel. ChannelDropout is not defined.')

        if self.max_channels >= num_channels:
            raise ValueError('Can not drop all channels in ChannelDropout.')

        num_drop_channels = np.random.randint(low=self.min_channels, high=self.max_channels+1)
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

        self.mean = mean
        self.variance = variance
        self.per_channel = per_channel

        if variance < 0:
            raise ValueError('Variance should be non negative.')

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

        if not (isinstance(kernel_size_range, tuple) and list(map(type, kernel_size_range)) == [int, int]):
            raise ValueError('Invalive type of kernel_size_range. Must be (int, int), but got {}'.format(kernel_size_range))

        self.kernel_size_range = kernel_size_range

        self.min_kernel_size = kernel_size_range[0]
        self.max_kernel_size = kernel_size_range[1]

        if self.min_kernel_size % 2 == 0 or self.max_kernel_size % 2 == 0:
            raise ValueError('Invalid range borders. Must be odd, but got: {}'.format(kernel_size_range))

        if not 1 <= self.min_kernel_size <= self.max_kernel_size:
            raise ValueError('Invalid kernel_size_range. Got: {}'.format(kernel_size_range))

    def apply(self, ecg, kernel, **params):
        return F.conv(ecg, kernel, cv2.BORDER_CONSTANT, 0)

    def get_params(self):
        kernel_size = 2 * np.random.randint(self.min_kernel_size // 2, self.max_kernel_size // 2 + 1) + 1
        kernel = np.ones(kernel_size) / kernel_size

        return {'kernel': kernel}

    def get_transform_init_args_names(self):
        return ('kernel_size_range', )

class GaussianBlur(EcgOnlyTransform):
    """Blur by gaussian the input ecg.
    """
    def __init__(
            self,
            variance=1.,
            kernel_size_range=(3, 5),
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
        super(GaussianBlur, self).__init__(always_apply, p)

        if variance < 0:
            raise ValueError('Variance should be non negative.')

        self.variance = variance

        if not (isinstance(kernel_size_range, tuple) and list(map(type, kernel_size_range)) == [int, int]):
            raise ValueError('Invalive type of kernel_size_range. Must be (int, int), but got {}'.format(kernel_size_range))

        self.kernel_size_range = kernel_size_range

        self.min_kernel_size = kernel_size_range[0]
        self.max_kernel_size = kernel_size_range[1]

        if self.min_kernel_size % 2 == 0 or self.max_kernel_size % 2 == 0:
            raise ValueError('Invalid range borders. Must be odd, but got: {}'.format(kernel_size_range))

        if not 1 <= self.min_kernel_size <= self.max_kernel_size:
            raise ValueError('Invalid kernel_size_range. Got: {}'.format(kernel_size_range))

    def apply(self, ecg, kernel, **params):
        return F.conv(ecg, kernel, cv2.BORDER_CONSTANT, 0)

    def get_params(self):
        kernel_size = 2 * np.random.randint(self.min_kernel_size // 2, self.max_kernel_size // 2 + 1) + 1

        kernel = np.exp(-0.5 * np.square(np.arange(-kernel_size, kernel_size+1)) / self.variance)
        kernel = kernel / np.sum(kernel)

        return {'kernel': kernel}

    def get_transform_init_args_names(self):
        return ('variance', 'kernel_size_range')
