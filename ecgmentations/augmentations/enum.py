import enum

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
