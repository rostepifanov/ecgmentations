def _non_negative_number(param, name):
    if param < 0:
        raise ValueError('{} should be non negative.'.format(name))

    return param

def _prepare_int(param, name, check):
    if not isinstance(param, int):
        raise ValueError(
            '{} must be scalar (int).'.format(
                name
            )
        )

    param = int(param)

    return check(param, name)

prepare_int = lambda param, name: _prepare_int(param, name, lambda x, _: x)
prepare_non_negative_int = lambda param, name: _prepare_int(param, name, _non_negative_number)

def _prepare_float(param, name, check):
    if not isinstance(param, (float, int)):
        raise ValueError(
            '{} must be scalar (float).'.format(
                name
            )
        )

    param = float(param)

    return check(param, name)

prepare_float = lambda param, name: _prepare_float(param, name, lambda x, _: x)
prepare_non_negative_float = lambda param, name: _prepare_float(param, name, _non_negative_number)

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

def prepare_float_asymrange(param, name, low):
    if isinstance(param, float):
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
            if not (list(map(type, param)) == [float, float]):
                raise ValueError(
                    '{} must be tuple (float, float).'.format(
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
            '{} must be either scalar (float) or tuple (float, float).'.format(
                name
            )
        )

    return tuple(param)
