from itertools import tee

def pairwise(iterable):
    first, second = tee(iterable)
    next(second, None)
    yield from zip(first, second)

def prepare_float(param, name):
    if not isinstance(param, (float, int)):
        raise ValueError(
            '{} must be scalar (float).'.format(
                name
            )
        )

    return float(param)

def prepare_int_asymrange(param, name, low):
    if isinstance(param, int):
        if param >= low:
            param = (low, param)
        else:
            raise ValueError(
                'Invalid value of {}. Got {} that less than {}.'.format(
                    name, param, low
                )
            )
    elif isinstance(param, (tuple, list)):
        if len(param) == 2:
            if not (list(map(type, param)) == [int, int]):
                raise ValueError(
                    '{} must be tuple (int, int).'.format(
                        name
                    )
                )

            if param[0] > param[1]:
                param = (param[1], param[0])

            if param[0] < low:
                raise ValueError(
                    'Invalid value of {}. Got: {} that less than {}.'.format(
                        name, param, low
                    )
                )
        else:
            raise ValueError(
                'Invalid value of {}. Got {}'.format(
                    name, param
                )
            )
    else:
        raise ValueError(
            '{} must be either scalar (int) or tuple (int, int).'.format(
                name
            )
        )

    return tuple(param)

def prepare_float_symrange(param, name):
    if isinstance(param, float):
        if param >= 0:
            param = (-param, param)
        else:
            param = (param, -param)
    elif isinstance(param, (tuple, list)):
        if len(param) == 2:
            if param[0] > param[1]:
                param = (param[1], param[0])
        else:
            raise ValueError(
                'Invalid value of {}. Got: {}.'.format(
                    name, param
                )
            )
    else:
        raise ValueError(
            '{} must be either scalar (float) or tuple (float, float).'.format(
                name
            )
        )

    return tuple(param)
