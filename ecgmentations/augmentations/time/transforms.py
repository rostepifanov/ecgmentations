import cv2
import numpy as np

import ecgmentations.augmentations.misc as M
import ecgmentations.augmentations.functional as F

from ecgmentations.augmentations.enum import PositionType
from ecgmentations.core.transforms import EcgOnlyTransform, DualTransform

class TimeReverse(DualTransform):
    """Reverse the input ecg.
    """
    def apply(self, ecg, **params):
        return F.time_reverse(ecg)

    def get_transform_init_args_names(self):
        return tuple()

class TimePadIfNeeded(DualTransform):
    """Pad lenght of the ecg to the minimal value.
    """
    def __init__(
            self,
            min_length=5000,
            position=PositionType.CENTER,
            border_mode=cv2.BORDER_CONSTANT,
            fill_value=0.,
            fill_mask_value=0,
            always_apply=False,
            p=1.0,
        ):
        """
            :args:
                min_length (int): minimal length to fill with padding
                position (PositionType, str): position of padding
                border_mode (OpenCV flag): OpenCV border mode
                fill_value (int, float, None): padding value if border_mode is cv2.BORDER_CONSTANT
                fill_mask_value (int, None): padding value for mask if border_mode is cv2.BORDER_CONSTANT
        """
        super(TimePadIfNeeded, self).__init__(always_apply, p)

        self.min_length = M.prepare_non_negative_int(min_length, 'min_length')
        self.position = PositionType(position)

        self.border_mode = border_mode
        self.fill_value = fill_value
        self.fill_mask_value = fill_mask_value

    def apply(self, ecg, left_pad, rigth_pad, **params):
        return F.pad(ecg, left_pad, rigth_pad, self.border_mode, self.fill_value)

    def apply_mask(self, mask, left_pad, rigth_pad, **params):
        return F.pad(ecg, left_pad, rigth_pad, self.border_mode, self.fill_mask_value)

    @property
    def targets_as_params(self):
        return ['ecg']

    def get_params_dependent_on_targets(self, params):
        length = params['ecg'].shape[0]

        pad_length = max(0, self.min_length - length)

        if self.position == PositionType.LEFT:
            left_pad = 0
            rigth_pad = pad_length
        elif self.position == PositionType.CENTER:
            left_pad = pad_length // 2
            rigth_pad = pad_length - left_pad
        elif self.position == PositionType.RIGHT:
            left_pad = pad_length
            rigth_pad = 0
        else:
            left_pad = np.random.randint(0, pad_length + 1)
            rigth_pad = pad_length - left_pad

        return {'left_pad': left_pad, 'rigth_pad': rigth_pad}

    def get_transform_init_args_names(self):
        return ('min_length', 'position', 'border_mode', 'fill_value', 'fill_mask_value')

class TimeCrop(DualTransform):
    """Crop time region from the input ecg.
    """
    def __init__(
            self,
            length=5000,
            position=PositionType.RANDOM,
            always_apply=False,
            p=1.0,
        ):
        """
            :args:
                length (int): the length of cropped region
                position (PositionType, str): position of padding
        """
        super(TimeCrop, self).__init__(always_apply, p)

        self.length = M.prepare_int(length, 'length')
        self.position = PositionType(position)

    def apply(self, ecg, left_bound, **params):
        return F.time_crop(ecg, left_bound, self.length)

    def get_params(self):
        if self.position == PositionType.LEFT:
            left_bound = 1.0
        elif self.position == PositionType.CENTER:
            left_bound = 0.5
        elif self.position == PositionType.RIGHT:
            left_bound = 0.0
        else:
            left_bound = np.random.random()

        return {'left_bound': left_bound}

    def get_transform_init_args_names(self):
        return ('length', 'position')

CenterTimeCrop = lambda *args, **kwargs: TimeCrop(*args, **kwargs, position='center')
RandomTimeCrop = lambda *args, **kwargs: TimeCrop(*args, **kwargs, position='random')
