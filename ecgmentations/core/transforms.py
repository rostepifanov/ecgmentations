import cv2
import random

from ecgmentations.core.utils import format_args, get_shortest_class_fullname

class Transform(object):
    def __init__(self, always_apply = False, p = 0.5):
        self.p = p
        self.always_apply = always_apply

    def __call__(self, *args, **kwargs):
        if args:
            raise KeyError('You have to pass data to augmentations as named arguments, for example: aug(ecg=ecg)')

        if (random.random() < self.p) or self.always_apply:
            params = self.get_params()

            return self.apply_with_params(params, **kwargs)

        return kwargs

    def apply_with_params(self, params, **kwargs):
        if params is None:
            return kwargs

        res = {}

        for key, arg in kwargs.items():
            if arg is not None:
                target_function = self._get_target_function(key)
                res[key] = target_function(arg, **params)
            else:
                res[key] = None

        return res

    def __repr__(self):
        state = self.get_base_init_args()
        state.update(self.get_transform_init_args())
        return '{name}({args})'.format(name=self.__class__.__name__, args=format_args(state))

    def _get_target_function(self, key):
        target_function = self.targets.get(key, lambda x, **p: x)
        return target_function

    def get_params(self):
        return {}

    @property
    def targets(self):
        """ NOTE

            you must specify targets in subclass

            for example: ('egc', ) or ('egc', 'mask')
        """
        raise NotImplementedError

    @classmethod
    def get_class_fullname(cls):
        return get_shortest_class_fullname(cls)

    def get_transform_init_args_names(self):
        raise NotImplementedError(
            'Class {name} is not serializable because the `get_transform_init_args_names` method is not '
            'implemented'.format(name=self.get_class_fullname())
        )

    def get_base_init_args(self):
        return {'always_apply': self.always_apply, 'p': self.p}

    def get_transform_init_args(self):
        return {k: getattr(self, k) for k in self.get_transform_init_args_names()}

class EcgOnlyTransform(Transform):
    """Transform applied to ecg only."""

    def apply(self, ecg, **params):
        raise NotImplementedError

    @property
    def targets(self):
        return { 'ecg': self.apply }

class IdentityTransform(EcgOnlyTransform):
    """Dummy transform"""

    def apply(self, ecg, **params):
        return ecg

    def get_transform_init_args_names(self):
        return tuple()

class DualTransform(Transform):
    """Transform for segmentation task."""

    @property
    def targets(self) -> Dict[str, Callable]:
        return {
            'image': self.apply,
            'mask': self.apply_to_mask,
        }

    def apply_to_mask(self, ecg, **params):
        return self.apply(ecg, **{k: cv2.INTER_NEAREST if k == 'interpolation' else v for k, v in params.items()})
