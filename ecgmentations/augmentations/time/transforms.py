import cv2
import numpy as np

import ecgmentations.augmentations.misc as M
import ecgmentations.augmentations.time.functional as F

from ecgmentations.augmentations.enum import PositionType
from ecgmentations.core.transforms import DualTransform

class TimeReverse(DualTransform):
    """Reverse the input ecg.
    """
    def apply(self, ecg, **params):
        return F.time_reverse(ecg)

    def get_transform_init_args_names(self):
        return tuple()

class TimeShift(DualTransform):
    """Shift the input ecg along time axis.
    """
    def __init__(
            self,
            shift_limit=0.05,
            border_mode=cv2.BORDER_CONSTANT,
            fill_value=0.,
            fill_mask_value=0,
            always_apply=False,
            p=0.5,
        ):
        """
            :args:
                shift_limit (float): limit of shifting
                border_mode (OpenCV flag): OpenCV border mode
                fill_value (int, float, None): padding value if border_mode is cv2.BORDER_CONSTANT
                fill_mask_value (int, None): padding value for mask if border_mode is cv2.BORDER_CONSTANT
        """
        super(TimeShift, self).__init__(always_apply, p)

        self.shift_limit = M.prepare_non_negative_float(shift_limit, 'shift_limit')

        self.border_mode = border_mode
        self.fill_value = M.prepare_float(fill_value, 'fill_value')
        self.fill_mask_value = M.prepare_int(fill_mask_value, 'fill_mask_value')

    def apply(self, ecg, shift, **params):
        return F.time_shift(ecg, shift, self.border_mode, self.fill_value)

    def apply_mask(self, mask, shift, **params):
        return F.time_shift(ecg, shift, self.border_mode, self.fill_mask_value)

    def get_params(self):
        shift = np.random.random() * self.shift_limit

        return {'shift': shift}

    def get_transform_init_args_names(self):
        return ('shift_limit', 'border_mode', 'fill_value', 'fill_mask_value')

class TimeSegmentShuffle(DualTransform):
    """Randomly shuffle of the input ecg segments
    """
    def __init__(
            self,
            num_segments=5,
            always_apply=False,
            p=0.5,
        ):
        """
            :args:
                num_segments (int): count of grid cells on the ecg
        """
        super(TimeSegmentShuffle, self).__init__(always_apply, p)

        self.num_segments = M.prepare_non_negative_int(num_segments, 'num_segments')

    def apply(self, ecg, segment_order, **params):
        return F.time_segment_swap(ecg, segment_order)

    def get_params(self):
        segment_order = np.arange(self.num_segments)
        np.random.shuffle(segment_order)

        return {'segment_order': segment_order}

    def get_transform_init_args_names(self):
        return ('num_segments', )

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

        self.num_steps = M.prepare_non_negative_int(num_steps, 'num_steps')
        self.wrap_limit = M.prepare_non_negative_float(wrap_limit, 'wrap_limit')

    def apply(self, ecg, cells, ncells, **params):
        return F.time_wrap(ecg, cells, ncells)

    def get_params(self):
        cells = np.linspace(0, 1, self.num_steps + 1)
        ncells = np.linspace(0, 1, self.num_steps + 1)

        if self.num_steps > 1:
            directions = np.random.choice([-1, 1], size=self.num_steps - 1)
            shifts = np.random.random(size=self.num_steps-1) * self.wrap_limit * 0.5

            ncells[1:-1] += shifts * directions / (self.num_steps + 1)

        return {'cells': cells, 'ncells': ncells}

    def get_transform_init_args_names(self):
        return ('num_steps', 'wrap_limit')

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

class TimeCrop(DualTransform):
    """Crop time segment from the input ecg.
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
                length (int): the length of cropped time segment
                position (PositionType, str): position of cropped time segment
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

class CenterTimeCrop(TimeCrop):
    """Crop time segment to the center of the input ecg.
    """
    def __init__(
            self,
            length=5000,
            always_apply=False,
            p=1.0,
        ):
        """
            :args:
                length (int): the length of cropped region
        """
        super(CenterTimeCrop, self).__init__(length, PositionType.CENTER, always_apply, p)

    def get_transform_init_args_names(self):
        return ('length', )

class RandomTimeCrop(TimeCrop):
    """Crop a random time segment of the input ecg.
    """
    def __init__(
            self,
            length=5000,
            always_apply=False,
            p=1.0,
        ):
        """
            :args:
                length (int): the length of cropped region
        """
        super(RandomTimeCrop, self).__init__(length, PositionType.RANDOM, always_apply, p)

    def get_transform_init_args_names(self):
        return ('length', )

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
                position (PositionType, str): position of ecg
                border_mode (OpenCV flag): OpenCV border mode
                fill_value (int, float, None): padding value if border_mode is cv2.BORDER_CONSTANT
                fill_mask_value (int, None): padding value for mask if border_mode is cv2.BORDER_CONSTANT
        """
        super(TimePadIfNeeded, self).__init__(always_apply, p)

        self.min_length = M.prepare_non_negative_int(min_length, 'min_length')
        self.position = PositionType(position)

        self.border_mode = border_mode
        self.fill_value = M.prepare_float(fill_value, 'fill_value')
        self.fill_mask_value = M.prepare_int(fill_mask_value, 'fill_mask_value')

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
