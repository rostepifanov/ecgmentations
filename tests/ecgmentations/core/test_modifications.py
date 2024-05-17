import pytest

import numpy as np
import ecgmentations as E

@pytest.mark.core
def test_ToChannels_CASE_create_AND_type_error():
    with pytest.raises(RuntimeError, match=r'transform is type of <.+> that is not subtype of Apply'):
        transform = E.ToChannels(
            object()
        , always_apply=True)

@pytest.mark.core
def test_ToChannels_CASE_create_AND_channels_error():
    with pytest.raises(RuntimeError, match=r'object at \d+ position is not subtype of int'):
        transform = E.ToChannels(
            E.TimeReverse(always_apply=True)
        , channels=[0, 1.], always_apply=True)

@pytest.mark.core
def test_ToChannels_CASE_call_AND_zero_channel():
    ecg = np.random.random((12, 5000)).T
    mask = np.arange(1000)[: None]

    channels = [0, ]
    exchannels = [ ch for ch in np.arange(12) if ch not in channels]

    transform = E.ToChannels(
        E.TimeReverse(always_apply=True)
    , channels=channels, always_apply=True)

    transformed = transform(ecg=ecg, mask=mask)
    tecg, tmask = transformed['ecg'], transformed['mask']

    assert tecg.shape == ecg.shape
    assert pytest.approx(tecg[:, channels]) != ecg[:, channels]
    assert pytest.approx(tecg[:, exchannels]) == ecg[:, exchannels]

    assert tmask.shape == mask.shape
    assert pytest.approx(tmask) != mask
