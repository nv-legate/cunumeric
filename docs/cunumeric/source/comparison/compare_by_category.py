import importlib

import numpy

blocklist = [
    "test",
    "add_docstring",
    "add_newdoc",
    "add_newdoc_ufunc",
    "alen",
    "alltrue",
    "compare_chararrays",
    "fastCopyAndTranspose",
    "get_array_wrap",
    "iterable",
    "recfromcsv",
    "recfromtxt",
    "safe_eval",
    "set_numeric_ops",
    "sometrue",
    "loads",
    "mafromtxt",
    "ndfromtxt",
]

used_api = []


def check_ufunc(obj, n):
    try:
        return isinstance(getattr(obj, n), numpy.ufunc)
    except:  # noqa E722
        return False


def _filter(obj, n, ufuncs=False, api_list=None):
    is_ufunc = check_ufunc(obj, n)
    if not ufuncs:
        is_ufunc = not is_ufunc

    if api_list is None:

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

    else:
        try:
            return (
                n not in blocklist
                and n not in used_api
                and n in api_list
                and callable(getattr(obj, n))  # callable
                and not isinstance(getattr(obj, n), type)  # not class
                and n[0].islower()  # starts with lower char
                and not n.startswith("__")  # not special methods
            )
        except:  # noqa: E722
            return False


def _get_functions(obj, ufuncs=False, api_list=None):
    return set([n for n in dir(obj) if (_filter(obj, n, ufuncs, api_list))])


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
    header,
    mod_ext,
    other_lib,
    klass=None,
    exclude_mod=None,
    ufuncs=False,
    api_list=None,
):
    base_mod = "numpy" + mod_ext
    other_mod = other_lib + mod_ext

    base_funcs = []
    base_obj, base_fmt = _import(base_mod, klass)
    base_funcs = _get_functions(base_obj, ufuncs, api_list)
    lg_obj, lg_fmt = _import(other_mod, klass)

    lg_funcs = []

    for f in _get_functions(lg_obj, api_list=api_list):
        obj = getattr(lg_obj, f)
        if (
            other_lib == "cunumeric"
            and (
                not hasattr(obj, "_cunumeric")
                or not obj._cunumeric.implemented
            )
        ) or (
            other_lib != "cunumeric"
            and (obj.__doc__ is None or "Unimplemented" not in obj.__doc__)
        ):
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
        base_cell = base_fmt.format(f)
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
        used_api.append(f)

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


def _check_convolve_and_correlate(other_lib):
    list_of_api = ["convolve", "correlate"]
    return _section(
        "Convole and Correlate", "", other_lib, api_list=list_of_api
    )


def _logic_functions(other_lib):
    list_of_api = [
        "all",
        "any",
        "isfinite",
        "isinf",
        "isnan",
        "isnat",
        "isneginf",
        "isposinf",
        "iscomplex",
        "iscomplexobj",
        "isfortran",
        "isreal",
        "isrealobj",
        "isscalar",
        "logical_and",
        "logical_or",
        "logical_not",
        "logical_xor",
        "allclose",
        "isclose",
        "array_equal",
        "array_equiv",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "equal",
        "not_equal",
    ]
    return _section("Logical functions", "", other_lib, api_list=list_of_api)


def _einsum_and_related(other_lib):
    list_of_api = [
        "matmul",
        "tensordot",
        "dot",
        "inner",
        "outer",
        "trace",
        "einsum",
    ]
    return _section(
        "Einsum and related algorithms", "", other_lib, api_list=list_of_api
    )


def _set_routines(other_lib):
    list_of_api = [
        "arraysetops",
        "unique",
        "in1d",
        "intersect1d",
        "isin",
        "setdiff1d",
        "setxor1d",
        "union1d",
    ]
    return _section("Set routines", "", other_lib, api_list=list_of_api)


def _array_manipulation(other_lib):
    list_of_api = [
        "copyto",
        "shape",
        "reshape",
        "ravel",
        "flat",
        "flatten",
        "moveaxis",
        "rollaxis",
        "swapaxes",
        "T",
        "transpose",
        "atleast_1d",
        "atleast_2d",
        "atleast_3d",
        "broadcast",
        "broadcast_to",
        "broadcast_arrays",
        "expand_dims",
        "squeeze",
        "asarray",
        "asanyarray",
        "asmatrix",
        "asfarray",
        "asfortranarray",
        "ascontiguousarray",
        "asarray_chkfinite",
        "asscalar",
        "require",
        "concatenate",
        "stack",
        "block",
        "vstack",
        "hstack",
        "dstack",
        "column_stack",
        "row_stack",
        "split",
        "array_split",
        "dsplit",
        "hsplit",
        "vsplit",
        "tile",
        "repeat",
        "delete",
        "insert",
        "append",
        "resize",
        "trim_zeros",
        "unique",
        "flip",
        "fliplr",
        "flipud",
        "reshape",
        "roll",
        "rot90",
    ]
    buf = _section(
        "Array manipulation routines", "", other_lib, api_list=list_of_api
    )
    buf += _section(
        "Array manipulation routines (remaining)",
        "",
        other_lib,
        klass="ndarray",
        api_list=list_of_api,
    )
    return buf


def _factorizations(other_lib):
    list_of_api = ["qr", "cholesky"]
    return _section(
        "Factorizations", ".linalg", other_lib, api_list=list_of_api
    )


def _svd_and_related(other_lib):
    list_of_api = ["svd", "matrix_rank", "pinv", "lstsq"]
    return _section(
        "SVD and related algorithms",
        ".linalg",
        other_lib,
        api_list=list_of_api,
    )


def _eigenvalue(other_lib):
    list_of_api = ["eig", "eigh", "eigvals", "eigvalsh"]
    return _section(
        "Eigenvalue functions", ".linalg", other_lib, api_list=list_of_api
    )


def _LU_factorization_and_related(other_lib):
    list_of_api = [
        "det",
        "slogdet",
        "solve",
        "inv",
        "pinv",
        "tensorinv",
        "tensorsolve",
    ]
    return _section(
        "LU factorization and related algoeithms",
        ".linalg",
        other_lib,
        api_list=list_of_api,
    )


def _array_creation(other_lib):
    list_of_api = [
        "empty",
        "empty_like",
        "eye",
        "identity",
        "ones",
        "ones_like",
        "zeros",
        "zeros_like",
        "full",
        "full_like",
        "array",
        "asarray",
        "asanyarray",
        "ascontiguousarray",
        "asmatrix",
        "copy",
        "frombuffer",
        "fromfile",
        "fromfunction",
        "fromiter",
        "fromstring",
        "loadtxt",
        "core.records.array",
        "core.records.fromarrays",
        "core.records.fromrecords",
        "core.records.fromstring",
        "core.records.fromfile",
        "core.defchararray.array",
        "core.defchararray.asarray",
        "arange",
        "linspace",
        "logspace",
        "geomspace",
        "meshgrid",
        "mgrid",
        "ogrid",
        "diag",
        "diagflat",
        "tri",
        "tril",
        "triu",
        "vander",
        "mat",
        "bmat",
    ]
    buf = _section(
        "Array creation routines", "", other_lib, api_list=list_of_api
    )
    buf += _section(
        "Array creation routines (remaining)",
        "",
        other_lib,
        klass="ndarray",
        api_list=list_of_api,
    )
    return buf


def _io_routines(other_lib):
    list_of_api = [
        "load",
        "save",
        "savez",
        "savez_compressed",
        "loadtxt",
        "savetxt",
        "genfromtxt",
        "fromregex",
        "fromstring",
        "tofile",
        "tolist",
        "array2string",
        "array_repr",
        "array_str",
        "format_float_positional",
        "format_float_scientific",
        "memmap",
        "open_memmap",
        "set_printoptions",
        "get_printoptions",
        "set_string_function",
        "printoptions",
        "binary_repr",
        "base_repr",
        "DataSource",
        "format",
    ]
    buf = _section("Input and Output", "", other_lib, api_list=list_of_api)
    buf += _section(
        "Input and Output (remaining)",
        "",
        other_lib,
        klass="ndarray",
        api_list=list_of_api,
    )
    return buf


def _mathematics_function(other_lib):
    list_of_api = [
        "sin",
        "cos",
        "tan",
        "arcsin",
        "arccos",
        "arctan",
        "hypot",
        "arctan2",
        "degrees",
        "radians",
        "unwrap",
        "deg2rad",
        "rad2deg",
        "sinh",
        "cosh",
        "tanh",
        "arcsinh",
        "arccosh",
        "arctanh",
        "around",
        "round_",
        "rint",
        "fix",
        "floor",
        "ceil",
        "trunc",
        "prod",
        "sum",
        "nanprod",
        "nansum",
        "cumprod",
        "cumsum",
        "nancumprod",
        "nancumsum",
        "diff",
        "ediff1d",
        "gradient",
        "cross",
        "trapz",
        "exp",
        "expm1",
        "exp2",
        "log",
        "log10",
        "log2",
        "log1p",
        "logaddexp",
        "logaddexp2",
        "i0",
        "sinc",
        "signbit",
        "copysign",
        "frexp",
        "ldexp",
        "nextafter",
        "spacing",
        "lcm",
        "gcd",
        "add",
        "reciprocal",
        "positive",
        "negative",
        "multiply",
        "divide",
        "power",
        "subtract",
        "true_divide",
        "floor_divide",
        "float_power",
        "fmod",
        "mod",
        "modf",
        "remainder",
        "divmod",
        "angle",
        "real",
        "imag",
        "conj",
        "conjugate",
        "maximum",
        "fmax",
        "amax",
        "nanmax",
        "minimum",
        "fmin",
        "amin",
        "nanmin",
        "convolve",
        "clip",
        "sqrt",
        "cbrt",
        "square",
        "absolute",
        "fabs",
        "sign",
        "heaviside",
        "nan_to_num",
        "real_if_close",
        "interp",
    ]
    return _section(
        "Mathematics functions", "", other_lib, api_list=list_of_api
    )


def _sorting_searching_counting(other_lib):
    list_of_api = [
        "sort",
        "lexsort",
        "argsort",
        "msort",
        "sort_complex",
        "partition",
        "argpartition",
        "argmax",
        "nanargmax",
        "argmin",
        "nanargmin",
        "argwhere",
        "nonzero",
        "flatnonzero",
        "where",
        "searchsorted",
        "extract",
        "count_nonzero",
    ]
    return _section(
        "Sorting, Searchin and Counting", "", other_lib, api_list=list_of_api
    )


def _advanced_statistics(other_lib):
    list_of_api = [
        "ptp",
        "percentile",
        "nanpercentile",
        "quantile",
        "nanquantile",
        "median",
        "average",
        "mean",
        "std",
        "var",
        "nanmedian",
        "nanmean",
        "nanstd",
        "nanvar",
        "corrcoef",
        "correlate",
        "cov",
        "histogram",
        "histogram2d",
        "histogramdd",
        "bincount",
        "histogram_bin_edges",
        "digitize",
    ]
    return _section("Advanced Statistics", "", other_lib, api_list=list_of_api)


def _miscellaneous_matrix_routines(other_lib):
    list_of_api = ["matrix_power", "kron", "norm"]
    return _section(
        "Miscellaneous matrix routiness", "", other_lib, api_list=list_of_api
    )


def _pack_unpack(other_lib):
    list_of_api = ["packbits", "unpackbits"]
    return _section("Pack/unpackbits", "", other_lib, api_list=list_of_api)


def _index_routines(other_lib):
    list_of_api = [
        "c_",
        "r_",
        "s_",
        "nonzero",
        "where",
        "indices",
        "ix_",
        "ogrid",
        "ravel_multi_index",
        "unravel_index",
        "diag_indices",
        "diag_indices_from",
        "mask_indices",
        "tril_indices",
        "tril_indices_from",
        "triu_indices",
        "triu_indices_from",
        "take",
        "take_along_axis",
        "choose",
        "compress",
        "diag",
        "diagonal",
        "select",
        "sliding_window_view",
        "as_strided",
        "place",
        "put",
        "put_along_axis",
        "putmask",
        "fill_diagonal",
        "nditer",
        "ndenumerate",
        "ndindex",
        "nested_iters",
        "flatiter",
        "Arrayterator",
    ]
    return _section("Indexing routines", "", other_lib, api_list=list_of_api)


def _padding_arrays(other_lib):
    list_of_api = ["pad"]
    return _section("Padding arrays", "", other_lib, api_list=list_of_api)


def _functional_programming(other_lib):
    list_of_api = [
        "apply_along_axis",
        "apply_over_axes",
        "vectorize",
        "frompyfunc",
        "piecewise",
    ]
    return _section(
        "Functional programming", "", other_lib, api_list=list_of_api
    )


def generate(other_lib):
    buf = []
    buf += [
        "NumPy vs cuNumeric APIs",
        "------------------------",
        "",
    ]
    buf += _check_convolve_and_correlate(other_lib)
    buf += _section("Ufuncs", "", other_lib, ufuncs=True)
    buf += _logic_functions(other_lib)
    buf += _einsum_and_related(other_lib)
    buf += _section("Discrete Fourier Transform", ".fft", other_lib)
    buf += _set_routines(other_lib)
    buf += _array_manipulation(other_lib)
    buf += _factorizations(other_lib)
    buf += _svd_and_related(other_lib)
    buf += _eigenvalue(other_lib)
    buf += _LU_factorization_and_related(other_lib)
    buf += _io_routines(other_lib)
    buf += _array_creation(other_lib)
    buf += _mathematics_function(other_lib)
    buf += _sorting_searching_counting(other_lib)
    buf += _advanced_statistics(other_lib)
    buf += _miscellaneous_matrix_routines(other_lib)
    buf += _pack_unpack(other_lib)
    buf += _index_routines(other_lib)
    buf += _padding_arrays(other_lib)
    buf += _section("Random Sampling", ".random", other_lib)
    buf += _functional_programming(other_lib)

    return "\n".join(buf)


if __name__ == "__main__":
    print(generate("cupy"))
