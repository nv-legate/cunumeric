import importlib

def _filter(obj, n):
    try:
        return (n not in ['test']  # not in blacklist 
            and callable(getattr(obj, n))  # callable
            and not isinstance(getattr(obj, n), type)  # not class
            and n[0].islower()  # starts with lower char
            and not n.startswith('__')  # not special methods
        )
    except:
        return False

def _get_functions(obj):
    return set([
        n for n in dir(obj)
        if (_filter(obj, n))
    ])


def _import(mod, klass):
    try:
        obj = importlib.import_module(mod)
    except ModuleNotFoundError:
        return None, None

    if klass:
        obj = getattr(obj, klass)
        return obj, ':meth:`{}.{}.{{}}`'.format(mod, klass)
    else:
        # ufunc is not a function
        return obj, ':obj:`{}.{{}}`'.format(mod)


def _generate_comparison_rst(
        base_mod, legate_mod, base_type, klass, exclude_mod):
    base_obj, base_fmt = _import(base_mod, klass)
    base_funcs = _get_functions(base_obj)
    lg_obj, lg_fmt = _import(legate_mod, klass)

    lg_funcs = []
    for f in _get_functions(lg_obj):
        obj = getattr(lg_obj, f)
        if obj.__doc__ is None or 'Unimplemented' not in obj.__doc__:
            lg_funcs.append(f)
    lg_funcs = set(lg_funcs)

    if exclude_mod:
        exclude_obj, _ = _import(exclude_mod, klass)
        exclude_funcs = _get_functions(exclude_obj)
        base_funcs -= exclude_funcs
        lg_funcs -= exclude_funcs

    buf = []
    buf += [
        '.. csv-table::',
        '   :header: {}, Legate'.format(base_type),
        '',
    ]
    for f in sorted(base_funcs):
        base_cell = base_fmt.format(f)
        lg_cell = r'\-'
        if f in lg_funcs:
            lg_cell = lg_fmt.format(f)
            if getattr(base_obj, f) is getattr(lg_obj, f):
                lg_cell = '{} (*alias of* {})'.format(lg_cell, base_cell)
        line = '   {}, {}'.format(base_cell, lg_cell)
        buf.append(line)

    buf += [
        '',
        '.. Summary:',
        '   Number of NumPy functions: {}'.format(len(base_funcs)),
        '   Number of functions covered by Legate: {}'.format(
            len(lg_funcs & base_funcs)),
        '   Legate specific functions:',
    ] + [
        '   - {}'.format(f) for f in (lg_funcs - base_funcs)
    ]
    return buf


def _section(
        header, base_mod, legate_mod,
        base_type='NumPy', klass=None, exclude=None):
    return [
        header,
        '~' * len(header),
        '',
    ] + _generate_comparison_rst(
        base_mod, legate_mod, base_type, klass, exclude
    ) + [
        '',
    ]


def generate():
    buf = []

    buf += [
        'NumPy / Legate APIs',
        '-----------------',
        '',
    ]
    buf += _section(
        'Module-Level',
        'numpy', 'legate.numpy')
    buf += _section(
        'Multi-Dimensional Array',
        'numpy', 'legate.numpy', klass='ndarray')
    buf += _section(
        'Linear Algebra',
        'numpy.linalg', 'legate.numpy.linalg')
    buf += _section(
        'Discrete Fourier Transform',
        'numpy.fft', 'legate.numpy.fft')
    buf += _section(
        'Random Sampling',
        'numpy.random', 'legate.numpy.random')

    return '\n'.join(buf)