import numpy as np

from ecgmentations.core.application import Apply
from ecgmentations.core.utils import format_args, get_shortest_class_fullname

class Compose(Apply):
    def __init__(self, transforms, always_apply, p):
        """
            :args:
                transforms: list of Apply
                    list of operations to compose
                always_apply: bool
                    the flag of force application
                p: float
                    the probability of application
        """
        super(Compose, self).__init__(always_apply, p)

        if not isinstance(transforms, list):
            raise RuntimeError(
                'transforms is type of {} that is not list'.format(type(transforms))
            )
        elif not all(isinstance(t, Apply) for t in transforms):
            for idx, t in enumerate(transforms):
                if not isinstance(t, Apply):
                    raise RuntimeError(
                        'object at {} position is not subtype of Apply'.format(idx)
                    )

        self.transforms = transforms

    def __len__(self):
        return len(self.transforms)

    def __getitem__(self, idx):
        return self.transforms[idx]

    def __repr__(self):
        return self.repr()

    def repr(self, indent=Apply.REPR_INDENT_STEP):
        args = self.get_base_init_args()

        repr_string = self.get_class_name() + '(['

        for t in self.transforms:
            repr_string += '\n'

            if hasattr(t, 'repr'):
                t_repr = t.repr(indent + self.REPR_INDENT_STEP)
            else:
                t_repr = repr(t)

            repr_string += ' ' * indent + t_repr + ','

        repr_string += '\n' + ' ' * (indent - self.REPR_INDENT_STEP) + '], {args})'.format(args=format_args(args))

        return repr_string

    @classmethod
    def get_class_fullname(cls):
        return get_shortest_class_fullname(cls)

class Sequential(Compose):
    """Compose transforms to apply sequentially.
    """
    def __init__(self, transforms, always_apply=False, p=1.0):
        """
            :args:
                transforms: list of Apply
                    list of operations to apply sequentially
                always_apply: bool
                    the flag of force application
                p: float
                    the probability of application
        """
        super(Sequential, self).__init__(transforms, always_apply, p)

    def __call__(self, *args, force_apply=False, **data):
        if self.whether_apply(force_apply):
            for transform in self.transforms:
                data = transform(**data)

        return data

class OneOf(Compose):
    """Select one of transforms to apply.
    """
    def __init__(self, transforms, always_apply=False, p=0.5):
        """
            :NOTE:
                transform probabilities will be normalized to one 1, so in this case transforms probabilities works as weights.

            :args:
                transforms: list of Apply
                    list of operations to select one to apply
                always_apply: bool
                    the flag of force application
                p: float
                    the probability of application
        """
        super(OneOf, self).__init__(transforms, always_apply, p)

        transforms_ps = [t.p for t in self.transforms]
        s = sum(transforms_ps)

        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, *args, force_apply = False, **data):
        if self.transforms_ps and self.whether_apply(force_apply):
            transform = np.random.choice(self.transforms, p=self.transforms_ps)
            data = transform(force_apply=True, **data)

        return data
