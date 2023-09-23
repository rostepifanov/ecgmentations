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