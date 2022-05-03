# Copyright 2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import annotations

from dataclasses import dataclass
from types import (
    BuiltinFunctionType,
    FunctionType,
    MethodDescriptorType,
    MethodType,
)

import numpy

__all__ = (
    "GROUPED_CONFIGS",
    "MISSING_NP_REFS",
    "NUMPY_CONFIGS",
    "SKIP",
)

SKIP = {
    "abs",
    "add_docstring",
    "add_newdoc_ufunc",
    "add_newdoc",
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
}

# these do not have valid intersphinx references
MISSING_NP_REFS = {
    "numpy.loads",
    "numpy.mafromtxt",
    "numpy.ndarray.flat",
    "numpy.ndarray.shape",
    "numpy.ndarray.T",
    "numpy.ndfromtxt",
}


@dataclass(frozen=True)
class SectionConfig:
    title: str
    attr: str | None
    types: tuple(type) | None = None
    names: tuple(str) | None = None


FUNCTIONS = (FunctionType, BuiltinFunctionType)
METHODS = (MethodType, MethodDescriptorType)
UFUNCS = (numpy.ufunc,)

NUMPY_CONFIGS = [
    SectionConfig("Module-Level", None, types=FUNCTIONS),
    SectionConfig("Ufuncs", None, types=UFUNCS),
    SectionConfig("Multi-Dimensional Array", "ndarray", types=METHODS),
    SectionConfig("Linear Algebra", "linalg", types=FUNCTIONS),
    SectionConfig("Discrete Fourier Transform", "fft", types=FUNCTIONS),
    SectionConfig("Random Sampling", "random", types=FUNCTIONS),
]

CONVOLVE = ("convolve", "correlate")

LOGICAL = (
    "all",
    "allclose",
    "any",
    "array_equal",
    "array_equiv",
    "equal",
    "greater_equal",
    "greater",
    "isclose",
    "iscomplex",
    "iscomplexobj",
    "isfinite",
    "isfortran",
    "isinf",
    "isnan",
    "isnat",
    "isneginf",
    "isposinf",
    "isreal",
    "isrealobj",
    "isscalar",
    "less_equal",
    "less",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "not_equal",
)

EINSUM = (
    "dot",
    "einsum",
    "inner",
    "matmul",
    "outer",
    "tensordot",
    "trace",
)

SET = (
    "in1d",
    "intersect1d",
    "isin",
    "setdiff1d",
    "setxor1d",
    "union1d",
    "unique",
)

MANIP = (
    "flat",
    "flatten",
    "ravel",
    "repeat",
    "reshape",
    "reshape",
    "resize",
    "shape",
    "squeeze",
    "swapaxes",
    "T",
    "transpose",
)

FACTOR = ("cholesky", "qr")

SVD = ("lstsq", "matrix_rank", "pinv", "svd")

EIGEN = ("eig", "eigh", "eigvals", "eigvalsh")

LU = (
    "det",
    "inv",
    "pinv",
    "slogdet",
    "solve",
    "tensorinv",
    "tensorsolve",
)

CREATION = (
    "arange",
    "array",
    "asanyarray",
    "asarray",
    "ascontiguousarray",
    "asmatrix",
    "bmat",
    "copy",
    "diag",
    "diagflat",
    "empty_like",
    "empty",
    "eye",
    "frombuffer",
    "fromfile",
    "fromfunction",
    "fromiter",
    "fromstring",
    "full_like",
    "full",
    "geomspace",
    "identity",
    "linspace",
    "loadtxt",
    "logspace",
    "mat",
    "meshgrid",
    "mgrid",
    "ogrid",
    "ones_like",
    "ones",
    "tri",
    "tril",
    "triu",
    "vander",
    "zeros_like",
    "zeros",
)

# "core.defchararray.array",
# "core.defchararray.asarray",
# "core.records.array",
# "core.records.fromarrays",
# "core.records.fromfile",
# "core.records.fromrecords",
# "core.records.fromstring",

CREATION_ND = ("copy",)

IO = (
    "array_repr",
    "array_str",
    "array2string",
    "base_repr",
    "binary_repr",
    "DataSource",
    "format_float_positional",
    "format_float_scientific",
    "fromregex",
    "fromstring",
    "genfromtxt",
    "get_printoptions",
    "load",
    "loadtxt",
    "memmap",
    "printoptions",
    "save",
    "savetxt",
    "savez_compressed",
    "savez",
    "set_printoptions",
    "set_string_function",
)

IO_ND = ("tofile", "tolist")

MATH = (
    "absolute",
    "add",
    "amax",
    "amin",
    "angle",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "around",
    "cbrt",
    "ceil",
    "clip",
    "conj",
    "conjugate",
    "convolve",
    "copysign",
    "cos",
    "cosh",
    "cross",
    "cumprod",
    "cumsum",
    "deg2rad",
    "degrees",
    "diff",
    "divide",
    "divmod",
    "ediff1d",
    "exp",
    "exp2",
    "expm1",
    "fabs",
    "fix",
    "float_power",
    "floor_divide",
    "floor",
    "fmax",
    "fmin",
    "fmod",
    "frexp",
    "gcd",
    "gradient",
    "heaviside",
    "hypot",
    "i0",
    "imag",
    "interp",
    "lcm",
    "ldexp",
    "log",
    "log10",
    "log1p",
    "log2",
    "logaddexp",
    "logaddexp2",
    "maximum",
    "minimum",
    "mod",
    "modf",
    "multiply",
    "nan_to_num",
    "nancumprod",
    "nancumsum",
    "nanmax",
    "nanmin",
    "nanprod",
    "nansum",
    "negative",
    "nextafter",
    "positive",
    "power",
    "prod",
    "rad2deg",
    "radians",
    "real_if_close",
    "real",
    "reciprocal",
    "remainder",
    "rint",
    "round_",
    "sign",
    "signbit",
    "sin",
    "sinc",
    "sinh",
    "spacing",
    "sqrt",
    "square",
    "subtract",
    "sum",
    "tan",
    "tanh",
    "trapz",
    "true_divide",
    "trunc",
    "unwrap",
)

SEARCHING = (
    "argmax",
    "argmin",
    "argpartition",
    "argsort",
    "argwhere",
    "count_nonzero",
    "extract",
    "flatnonzero",
    "lexsort",
    "msort",
    "nanargmax",
    "nanargmin",
    "nonzero",
    "partition",
    "searchsorted",
    "sort_complex",
    "sort",
    "where",
)

STATS = (
    "average",
    "bincount",
    "corrcoef",
    "correlate",
    "cov",
    "digitize",
    "histogram_bin_edges",
    "histogram",
    "histogram2d",
    "histogramdd",
    "mean",
    "median",
    "nanmean",
    "nanmedian",
    "nanpercentile",
    "nanquantile",
    "nanstd",
    "nanvar",
    "percentile",
    "ptp",
    "quantile",
    "std",
    "var",
)

MISC = ("kron",)

PACK = ("packbits", "unpackbits")

INDEX = (
    "c_",
    "choose",
    "compress",
    "diag_indices_from",
    "diag_indices",
    "diag",
    "diagonal",
    "fill_diagonal",
    "flatiter",
    "indices",
    "ix_",
    "mask_indices",
    "ndenumerate",
    "ndindex",
    "nditer",
    "nested_iters",
    "nonzero",
    "ogrid",
    "place",
    "put_along_axis",
    "put",
    "putmask",
    "r_",
    "ravel_multi_index",
    "s_",
    "select",
    "take_along_axis",
    "take",
    "tril_indices_from",
    "tril_indices",
    "triu_indices_from",
    "triu_indices",
    "unravel_index",
    "where",
)

PAD = ("pad",)

FUNCTIONAL = (
    "apply_along_axis",
    "apply_over_axes",
    "frompyfunc",
    "piecewise",
    "vectorize",
)

GROUPED_CONFIGS = [
    SectionConfig("Convolve and Correlate", None, names=CONVOLVE),
    SectionConfig("Ufuncs", None, UFUNCS),
    SectionConfig("Logical operations", None, names=LOGICAL),
    SectionConfig("Einsum and related", None, names=EINSUM),
    SectionConfig("Discrete Fourier transform", "fft", types=FUNCTIONS),
    SectionConfig("Set operations", None, names=SET),
    SectionConfig("Array manipulation", "ndarray", names=MANIP),
    SectionConfig("Factorizations", "linalg", names=FACTOR),
    SectionConfig("SVD and related", "linalg", names=SVD),
    SectionConfig("Eigenvalues", "linalg", names=EIGEN),
    SectionConfig("LU factorization and related", "linalg", names=LU),
    SectionConfig("Input and output", None, names=IO),
    SectionConfig("Input and output (ndarray)", "ndarray", names=IO_ND),
    SectionConfig("Array creation", None, names=CREATION),
    SectionConfig("Array creation (ndarray)", "ndarray", names=CREATION_ND),
    SectionConfig("Mathematical functions", None, names=MATH),
    SectionConfig("Searching, sorting, and counting", None, names=SEARCHING),
    SectionConfig("Advanced statistics", None, names=STATS),
    SectionConfig("Miscellaneous matrix routines", None, names=MISC),
    SectionConfig("Packing and unpacking bits", None, names=PACK),
    SectionConfig("Indexing", None, names=INDEX),
    SectionConfig("Padding arrays", None, names=PAD),
    SectionConfig("Random sampling", "random", types=FUNCTIONS),
    SectionConfig("Functional programming", None, names=FUNCTIONAL),
]
