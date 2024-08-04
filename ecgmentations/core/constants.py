import ecgmentations.core.enum as E

SPATIAL_DIM = 0
CHANNEL_DIM = 1

NUM_SPATIAL_DIMENSIONS = 1
NUM_MONO_CHANNEL_DIMENSIONS = 1
NUM_MULTI_CHANNEL_DIMENSIONS = 2

MAP_BORDER_TYPE_TO_NUMPY = {
    E.BorderType.CONSTANT: 'constant',
    E.BorderType.REPLICATE: 'edge',
    E.BorderType.REFLECT_1001: 'symmetric',
    E.BorderType.REFLECT_101: 'reflect',
    E.BorderType.WRAP: 'wrap',
}

MAP_BORDER_TYPE_TO_SC = {
    E.BorderType.CONSTANT: 'constant',
    E.BorderType.REPLICATE: 'nearest',
    E.BorderType.REFLECT_1001: 'reflect',
    E.BorderType.REFLECT_101: 'mirror',
    E.BorderType.WRAP: 'wrap',
}
