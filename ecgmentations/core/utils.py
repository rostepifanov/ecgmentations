def format_args(args_dict):
    formatted_args = []

    for k, v in args_dict.items():
        if isinstance(v, str):
            v = f"'{v}'"
        formatted_args.append(f'{k}={v}')

    return ', '.join(formatted_args)

def shorten_class_name(class_fullname):
    splitted = class_fullname.split('.')

    if len(splitted) == 1:
        return class_fullname

    top_module, *_, class_name = splitted

    if top_module == 'ecgmentations':
        return class_name

    return class_fullname

def get_shortest_class_fullname(cls):
    class_fullname = '{cls.__module__}.{cls.__name__}'.format(cls=cls)
    return shorten_class_name(class_fullname)
