import numpy as np

from ecgmentations.core.utils import get_shortest_class_fullname

class Apply(object):
    """Root class for single and compound augmentations
    """

    REPR_INDENT_STEP=2

    def __init__(self, always_apply, p):
        """
            :args:
                always_apply: bool
                    the flag of force application
                p: float
                    the probability of application
        """
        self.always_apply = always_apply
        self.p = p

    def whether_apply(self, force_apply):
        return force_apply or self.always_apply or (np.random.random() < self.p)

    def __call__(self, *args, force_apply=False, **data):
        raise NotImplementedError

    def get_class_name(self):
        """
            :return:
                output: str
                    the name of class
        """
        return self.__class__.__name__

    @classmethod
    def get_class_fullname(cls):
        """
            :return:
                output: str
                    the full name of class
        """
        return get_shortest_class_fullname(cls)

    def get_base_init_args(self):
        """
            :return:
                output: dict
                    initialization parameters
        """
        return {'always_apply': self.always_apply, 'p': self.p}
