from ecgmentations.core.utils import get_shortest_class_fullname

class Apply(object):
    """Root class for single and compound augmentations
    """

    REPR_INDENT_STEP=2

    def __init__(self, always_apply, p):
        """
            :args:
                always_apply (bool): the flag of force application
                p (float): the probability of application
        """
        self.always_apply = always_apply
        self.p = p

    @classmethod
    def get_class_fullname(cls):
        """
            :return:
                the name of class as str
        """
        return get_shortest_class_fullname(cls)

    def get_base_init_args(self):
        """
            :return:
                initialization parameters
        """
        return {'always_apply': self.always_apply, 'p': self.p}
