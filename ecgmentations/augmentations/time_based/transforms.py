import ecgmentations.augmentations.time_based.functional as F

from ecgmentations.core.transforms import EcgOnlyTransform, DualTransform

class Flip(DualTransform):
    """Flip ecg.
    """
    def apply(self, ecg, **params):
        return F.flip(ecg)

    def get_transform_init_args_names(self):
        return tuple()
