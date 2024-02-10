from ecgmentations.core.transforms import EcgOnlyTransform, Identity, DualTransform
from ecgmentations.core.compositions import Sequential, OneOf
from ecgmentations.core.modifications import ToChannels