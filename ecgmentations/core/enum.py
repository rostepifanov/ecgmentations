import enum

class BorderType(enum.Enum):
    CONSTANT = 'constant'
    REPLICATE = 'replicate'
    REFLECT_1001 = 'reflect'
    REFLECT_101 = 'reflect'
    WRAP = 'wrap'
    DEFAULT = CONSTANT

class PositionType(enum.Enum):
    CENTER = 'center'
    LEFT = 'left'
    RIGHT = 'right'
    RANDOM = 'random'

class ReductionType(enum.Enum):
    MIN = 'min'
    MEAN = 'mean'
    MAX = 'max'
    MEDIAN  = 'median'
