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

import numpy as np

from ._ufunc.comparison import logical_and
from ._ufunc.floating import isinf, signbit
from .array import convert_to_cunumeric_ndarray, ndarray
from .module import full


def isneginf(x, out=None):
    """

    Test element-wise for negative infinity, return result as bool array.

    Parameters
    ----------
    x : array_like
        The input array.
    out : array_like, optional
        A location into which the result is stored. If provided, it must have a
        shape that the input broadcasts to. If not provided or None, a
        freshly-allocated boolean array is returned.

    Returns
    -------
    out : ndarray
        A boolean array with the same dimensions as the input.
        If second argument is not supplied then a numpy boolean array is
        returned with values True where the corresponding element of the
        input is negative infinity and values False where the element of
        the input is not negative infinity.

        If a second argument is supplied the result is stored there. If the
        type of that array is a numeric type the result is represented as
        zeros and ones, if the type is boolean then as False and True. The
        return value `out` is then a reference to that array.

    See Also
    --------
    numpy.isneginf

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    x = convert_to_cunumeric_ndarray(x)
    if out is not None:
        out = convert_to_cunumeric_ndarray(out, share=True)
    rhs1 = isinf(x)
    rhs2 = signbit(x)
    return logical_and(rhs1, rhs2, out=out)


def isposinf(x, out=None):
    """

    Test element-wise for positive infinity, return result as bool array.

    Parameters
    ----------
    x : array_like
        The input array.
    out : array_like, optional
        A location into which the result is stored. If provided, it must have a
        shape that the input broadcasts to. If not provided or None, a
        freshly-allocated boolean array is returned.

    Returns
    -------
    out : ndarray
        A boolean array with the same dimensions as the input.
        If second argument is not supplied then a boolean array is returned
        with values True where the corresponding element of the input is
        positive infinity and values False where the element of the input is
        not positive infinity.

        If a second argument is supplied the result is stored there. If the
        type of that array is a numeric type the result is represented as zeros
        and ones, if the type is boolean then as False and True.
        The return value `out` is then a reference to that array.

    See Also
    --------
    numpy.isposinf

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    x = convert_to_cunumeric_ndarray(x)
    if out is not None:
        out = convert_to_cunumeric_ndarray(out, share=True)
    rhs1 = isinf(x)
    rhs2 = ~signbit(x)
    return logical_and(rhs1, rhs2, out=out)


def iscomplex(x):
    """

    Returns a bool array, where True if input element is complex.

    What is tested is whether the input has a non-zero imaginary part, not if
    the input type is complex.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    out : ndarray[bool]
        Output array.

    See Also
    --------
    numpy.iscomplex

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    x = convert_to_cunumeric_ndarray(x)
    if x.dtype.kind != "c":
        return full(x.shape, False, dtype=bool)
    else:
        return x.imag != 0


def iscomplexobj(x):
    """

    Check for a complex type or an array of complex numbers.

    The type of the input is checked, not the value. Even if the input
    has an imaginary part equal to zero, `iscomplexobj` evaluates to True.

    Parameters
    ----------
    x : any
        The input can be of any type and shape.

    Returns
    -------
    iscomplexobj : bool
        The return value, True if `x` is of a complex type or has at least
        one complex element.

    See Also
    --------
    numpy.iscomplexobj

    Availability
    --------
    Single CPU
    """
    if isinstance(x, ndarray):
        return x.dtype.kind == "c"
    else:
        return np.iscomplexobj(x)


def isreal(x):
    """

    Returns a bool array, where True if input element is real.

    If element has complex type with zero complex part, the return value
    for that element is True.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    out : ndarray, bool
        Boolean array of same shape as `x`.


    See Also
    --------
    numpy.isreal

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    x = convert_to_cunumeric_ndarray(x)
    if x.dtype.kind != "c":
        return full(x.shape, True, dtype=bool)
    else:
        return x.imag == 0


def isrealobj(x):
    """

    Return True if x is a not complex type or an array of complex numbers.

    The type of the input is checked, not the value. So even if the input
    has an imaginary part equal to zero, `isrealobj` evaluates to False
    if the data type is complex.

    Parameters
    ----------
    x : any
        The input can be of any type and shape.

    Returns
    -------
    y : bool
        The return value, False if `x` is of a complex type.

    See Also
    --------
    numpy.isrealobj

    Availability
    --------
    Single CPU
    """
    return not iscomplexobj(x)


def isscalar(x):
    """

    Returns True if the type of `element` is a scalar type.

    Parameters
    ----------
    element : any
        Input argument, can be of any type and shape.

    Returns
    -------
    val : bool
        True if `element` is a scalar type, False if it is not.

    See Also
    --------
    numpy.isscalar

    Notes
    -----
    This function falls back to NumPy for all object types but cuNumeric's
    ndarray, which always returns `False`.

    Availability
    --------
    Single CPU
    """

    # Since the input can be any value, we can't just convert it to cunumeric
    # ndarray. Instead we check if the input is cunumeric ndarray and, if not,
    # fall back to Numpy
    if isinstance(x, ndarray):
        return False
    else:
        return np.isscalar(x)
