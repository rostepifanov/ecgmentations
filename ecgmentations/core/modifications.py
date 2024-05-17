import numpy as np

from ecgmentations.core.application import Apply
from ecgmentations.core.utils import format_args, get_shortest_class_fullname

class Modify(Apply):
    def __init__(self, transform, always_apply, p):
        """
            :args:
                transform: list of Apply
                    list of operations to apply with modification
                always_apply: bool
                    the flag of force application
                p: float
                    the probability of application
        """
        super(Modify, self).__init__(always_apply, p)

        if not isinstance(transform, Apply):
            raise RuntimeError(
                'transform is type of {} that is not subtype of Apply'.format(type(transform))
                )

        self.transform = transform

    def __repr__(self):
        return self.repr()

    def repr(self, indent=Apply.REPR_INDENT_STEP):
        args = self.get_base_init_args()

        repr_string = self.get_class_name() + '('

        repr_string += '\n'

        if hasattr(self.transform, 'repr'):
            t_repr = self.transform.repr(indent + self.REPR_INDENT_STEP)
        else:
            t_repr = repr(self.transform)

        repr_string += ' ' * indent + t_repr + ','

        repr_string += '\n' + ' ' * (indent - self.REPR_INDENT_STEP) + ', {args})'.format(args=format_args(args))

        return repr_string

class ToChannels(Modify):
    """Apply transforms to selected channels
    """
    def __init__(self, transform, channels=[0, ], always_apply=False, p=0.5):
        """
            :args:
                transform: list of Apply
                    list of operations to apply with modification
                channels: list of int
                    selected channels to apply transform
                always_apply: bool
                    the flag of force application
                p: float
                    the probability of application
        """
        super(ToChannels, self).__init__(transform, always_apply, p)

        if not isinstance(channels, list):
            raise RuntimeError(
                'channels is type of {} that is not list'.format(type(channels))
            )
        elif not all(isinstance(ch, int) for ch in channels):
            for idx, ch in enumerate(channels):
                if not isinstance(ch, int):
                    raise RuntimeError(
                        'object at {} position is not subtype of int'.format(idx)
                    )

        self.channels = channels

    def __call__(self, *args, force_apply=False, **data):
        if self.whether_apply(force_apply):
            ecg = np.copy(data['ecg'])

            data['ecg'] = data['ecg'][:, self.channels]
            data = self.transform(**data)

            ecg[:, self.channels] = data['ecg']
            data['ecg'] = ecg
        else:
            data = self.transform(**data)

        return data
