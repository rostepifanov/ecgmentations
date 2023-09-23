import numpy as np

from ecgmentations.core.transforms import Transform
from ecgmentations.core.utils import format_args, get_shortest_class_fullname

REPR_INDENT_STEP = 2

class Compose(object):
    def __init__(self, transforms, p):
        if isinstance(transforms, (Compose, Transform)):
            warnings.warn(
                'transforms is single transform, but a sequence is expected! Transform will be wrapped into list.'
            )
            transforms = [transforms]

        self.transforms = transforms
        self.p = p

    def __len__(self):
        return len(self.transforms)

    def __call__(self, *args, force_apply = False, **data):
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.transforms[idx]

    def __repr__(self):
        return self.repr()

    def repr(self, indent=REPR_INDENT_STEP):
        args = self.get_base_init_args()

        repr_string = self.__class__.__name__ + '(['

        for t in self.transforms:
            repr_string += '\n'

            if hasattr(t, 'repr'):
                t_repr = t.repr(indent + REPR_INDENT_STEP)
            else:
                t_repr = repr(t)

            repr_string += ' ' * indent + t_repr + ','

        repr_string += '\n' + ' ' * (indent - REPR_INDENT_STEP) + '], {args})'.format(args=format_args(args))

        return repr_string

    @classmethod
    def get_class_fullname(cls):
        return get_shortest_class_fullname(cls)

    def get_base_init_args(self):
        return {'p': self.p}

class Sequential(Compose):
    """Compose transforms to apply sequentially."""

    def __init__(self, transforms, p=0.5):
        super().__init__(transforms, p)

    def __call__(self, *args, force_apply = False, **data):
        for transform in self.transforms:
            data = transform(force_apply=force_apply, **data)

        return data

class OneOf(Compose):
    """Select one of transforms to apply.
    Transforms probabilities will be normalized to one 1, so in this case transforms probabilities works as weights.
    """

    def __init__(self, transforms, p=0.5):
        super(OneOf, self).__init__(transforms, p)

        transforms_ps = [t.p for t in self.transforms]
        s = sum(transforms_ps)

        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, *args, force_apply = False, **data):
        if (force_apply or (np.random.random() < self.p)) and self.transforms_ps:
            transform = np.random.choice(self.transforms, p=self.transforms_ps)
            data = transform(force_apply=True, **data)

        return data
