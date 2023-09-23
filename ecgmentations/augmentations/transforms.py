import numpy as np
import ecgmentations.augmentations.functional as F

from ecgmentations.core.transforms import EcgOnlyTransform, DualTransform

class Flip(DualTransform):
    """Flip ecg.
    """
    def apply(self, ecg, **params):
        return F.flip(ecg)

    def get_transform_init_args_names(self):
        return tuple()

class Invert(EcgOnlyTransform):
    """Invert ecg.
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
        p=0.5,
    ):
        super(ChannelDropout, self).__init__(always_apply, p)

        self.channel_drop_range = channel_drop_range

        self.min_channels = channel_drop_range[0]
        self.max_channels = channel_drop_range[1]

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
