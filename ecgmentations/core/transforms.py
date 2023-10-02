import cv2
import numpy as np

from ecgmentations.core.utils import format_args, get_shortest_class_fullname

class Transform(object):
    def __init__(self, always_apply=False, p=0.5):
        self.p = p
        self.always_apply = always_apply

    def __call__(self, *args, force_apply = False, **data):
        if args:
            raise KeyError('You have to pass data to augmentations as named arguments, for example: aug(ecg=ecg)')

        if force_apply or self.always_apply or (np.random.random() < self.p):
            params = self.get_params()

            if self.targets_as_params:
                assert all(name in data for name in self.targets_as_params), '{} requires {}'.format(
                    self.__class__.__name__, self.targets_as_params
                )

                targets_as_params = {name: data[name] for name in self.targets_as_params}

                params_dependent_on_targets = self.get_params_dependent_on_targets(targets_as_params)
                params.update(params_dependent_on_targets)

            return self.apply_with_params(params, **data)

        return data

    def apply_with_params(self, params, **data):
        if params is None:
            return data

        pdata = {}

        for name, datum in data.items():
            if datum is not None:
                target_function = self._get_target_function(name)
                pdata[name] = target_function(datum, **params)
            else:
                pdata[name] = None

        return pdata

    def __repr__(self):
        state = self.get_base_init_args()
        state.update(self.get_transform_init_args())
        return '{name}({args})'.format(name=self.__class__.__name__, args=format_args(state))

    def _get_target_function(self, name):
        target_function = self.targets.get(name, lambda x, **p: x)
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

    @property
    def targets_as_params(self):
        return []

    def get_params_dependent_on_targets(self, params):
        raise NotImplementedError(
            'Method get_params_dependent_on_targets is not implemented in class {name}'.format(name=self.__class__.__name__)
        )

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

class DualTransform(Transform):
    """Transform for segmentation task."""

    @property
    def targets(self):
        return {
            'ecg': self.apply,
            'mask': self.apply_to_mask,
        }

    def apply_to_mask(self, mask, **params):
        return self.apply(mask, **{k: cv2.INTER_NEAREST if k == 'interpolation' else v for k, v in params.items()})

class Identity(DualTransform):
    """Identity transform"""

    def apply(self, ecg, **params):
        return ecg

    def get_transform_init_args_names(self):
        return tuple()
