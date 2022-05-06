import importlib

import numpy

blocklist = [
    "abs",
    "add_docstring",
    "add_newdoc",
    "add_newdoc_ufunc",
    "alen",
    "alltrue",
    "bitwise_not",
    "compare_chararrays",
    "cumproduct",
    "fastCopyAndTranspose",
    "get_array_wrap",
    "iterable",
    "max",
    "min",
    "ndim",
    "product",
    "recfromcsv",
    "recfromtxt",
    "round",
    "safe_eval",
    "set_numeric_ops",
    "size",
    "sometrue",
    "test",
]

# these do not have valid intersphinx references
missing_numpy_refs = {
    "loads",
    "mafromtxt",
    "ndfromtxt",
}


def check_ufunc(obj, n):
    try:
        return isinstance(getattr(obj, n), numpy.ufunc)
    except:  # noqa E722
        return False


def _filter(obj, n, ufuncs=False):
    is_ufunc = check_ufunc(obj, n)
    if not ufuncs:
        is_ufunc = not is_ufunc

    try:
        return (
            n not in blocklist
            and callable(getattr(obj, n))  # callable
            and not isinstance(getattr(obj, n), type)  # not class
            and n[0].islower()  # starts with lower char
            and not n.startswith("__")  # not special methods
            and is_ufunc
        )
    except:  # noqa: E722
        return False


def _get_functions(obj, ufuncs=False):
    return set([n for n in dir(obj) if (_filter(obj, n, ufuncs))])


def _import(mod, klass):
    try:
        obj = importlib.import_module(mod)
    except ModuleNotFoundError:
        return None, None

    if klass:
        obj = getattr(obj, klass)
        return obj, ":meth:`{}.{}.{{}}`".format(mod, klass)
    else:
        # ufunc is not a function
        return obj, ":obj:`{}.{{}}`".format(mod)


def _section(
    header, mod_ext, other_lib, klass=None, exclude_mod=None, ufuncs=False
):
    base_mod = "numpy" + mod_ext
    other_mod = other_lib + mod_ext

    base_funcs = []
    base_obj, base_fmt = _import(base_mod, klass)
    base_funcs = _get_functions(base_obj, ufuncs)
    lg_obj, lg_fmt = _import(other_mod, klass)

    lg_funcs = []
    for f in _get_functions(lg_obj):
        obj = getattr(lg_obj, f)
        if hasattr(obj, "_cunumeric") and obj._cunumeric.implemented:
            lg_funcs.append(f)
    lg_funcs = set(lg_funcs)

    if exclude_mod:
        exclude_obj, _ = _import(exclude_mod, klass)
        exclude_funcs = _get_functions(exclude_obj)
        base_funcs -= exclude_funcs
        lg_funcs -= exclude_funcs

    buf = [
        header,
        "~" * len(header),
        "",
    ]

    buf += [
        ".. currentmodule:: cunumeric",
        "",
        ".. autosummary::",
        "   :toctree: generated/",
        "",
    ]

    buf += [
        ".. csv-table::",
        "   :header: NumPy, {}, single-GPU/CPU, multi-GPU/CPU".format(
            other_mod
        ),
        "",
    ]
    for f in sorted(base_funcs):
        if f not in missing_numpy_refs:
            base_cell = base_fmt.format(f)
        else:
            base_cell = f"``numpy.{f}``"
        lg_cell = r"\-"
        single_gpu_cell = ""
        multi_gpu_cell = ""
        if f in lg_funcs:
            lg_cell = lg_fmt.format(f)
            obj = getattr(lg_obj, f)
            if obj.__doc__ is not None and "Single GPU" in obj.__doc__:
                multi_gpu_cell = "No"
                single_gpu_cell = "Yes"
            elif obj.__doc__ is not None and "Multiple GPUs" in obj.__doc__:
                multi_gpu_cell = "Yes"
                single_gpu_cell = "Yes"
            if getattr(base_obj, f) is getattr(lg_obj, f):
                lg_cell = "{} (*alias of* {})".format(lg_cell, base_cell)
        line = "   {}, {}, {}, {}".format(
            base_cell, lg_cell, single_gpu_cell, multi_gpu_cell
        )
        buf.append(line)

    buf += [
        "",
        ".. Summary:",
        "   Number of NumPy functions: {}".format(len(base_funcs)),
        "   Number of functions covered by "
        f"{other_lib}: {len(lg_funcs & base_funcs)}",
    ]
    buf += [
        "",
    ]
    return buf


def generate(other_lib):
    buf = []
    buf += [
        "NumPy vs cuNumeric APIs",
        "------------------------",
        "",
    ]
    buf += _section("Module-Level", "", other_lib)
    buf += _section("Ufuncs", "", other_lib, ufuncs=True)
    buf += _section("Multi-Dimensional Array", "", other_lib, klass="ndarray")
    buf += _section("Linear Algebra", ".linalg", other_lib)
    buf += _section("Discrete Fourier Transform", ".fft", other_lib)
    buf += _section("Random Sampling", ".random", other_lib)

    return "\n".join(buf)


if __name__ == "__main__":
    print(generate("cunumeric"))
