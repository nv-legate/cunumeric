# Copyright 2021-2022 NVIDIA Corporation
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

import math
import operator
import re
from collections import Counter
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence, Union, cast

import numpy as np
import opt_einsum as oe  # type: ignore [import]
from numpy.core.multiarray import (  # type: ignore [attr-defined]
    normalize_axis_index,
)
from numpy.core.numeric import (  # type: ignore [attr-defined]
    normalize_axis_tuple,
)

from cunumeric.coverage import is_implemented

from ._ufunc.comparison import maximum, minimum
from ._ufunc.floating import floor
from ._ufunc.math import add, multiply
from .array import add_boilerplate, convert_to_cunumeric_ndarray, ndarray
from .config import BinaryOpCode, ScanCode, UnaryRedCode
from .runtime import runtime
from .types import NdShape, NdShapeLike, OrderType, SortSide
from .utils import AxesPairLike, inner_modes, matmul_modes, tensordot_modes

if TYPE_CHECKING:
    from typing import Callable

    import numpy.typing as npt

    from ._ufunc.ufunc import CastingKind
    from .types import BoundsMode, ConvolveMode, SelectKind, SortType

_builtin_abs = abs
_builtin_all = all
_builtin_any = any
_builtin_max = max
_builtin_min = min
_builtin_sum = sum

casting_kinds: tuple[CastingKind, ...] = (
    "no",
    "equiv",
    "safe",
    "same_kind",
    "unsafe",
)

#########################
# Array creation routines
#########################

# From shape or value


def empty(shape: NdShapeLike, dtype: npt.DTypeLike = np.float64) -> ndarray:
    """
    empty(shape, dtype=float)

    Return a new array of given shape and type, without initializing entries.

    Parameters
    ----------
    shape : int or tuple[int]
        Shape of the empty array.
    dtype : data-type, optional
        Desired output data-type for the array. Default is `cunumeric.float64`.

    Returns
    -------
    out : ndarray
        Array of uninitialized (arbitrary) data of the given shape and dtype.

    See Also
    --------
    numpy.empty

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return ndarray(shape=shape, dtype=dtype)


@add_boilerplate("a")
def empty_like(
    a: ndarray,
    dtype: Optional[npt.DTypeLike] = None,
    shape: Optional[NdShapeLike] = None,
) -> ndarray:
    """

    empty_like(prototype, dtype=None)

    Return a new array with the same shape and type as a given array.

    Parameters
    ----------
    prototype : array_like
        The shape and data-type of `prototype` define these same attributes
        of the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    shape : int or tuple[int], optional
        Overrides the shape of the result.

    Returns
    -------
    out : ndarray
        Array of uninitialized (arbitrary) data with the same shape and type as
        `prototype`.

    See Also
    --------
    numpy.empty_like

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    shape = a.shape if shape is None else shape
    if dtype is not None:
        dtype = np.dtype(dtype)
    else:
        dtype = a.dtype
    return ndarray(shape, dtype=dtype, inputs=(a,))


def eye(
    N: int,
    M: Optional[int] = None,
    k: int = 0,
    dtype: Optional[npt.DTypeLike] = np.float64,
) -> ndarray:
    """

    Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
      Number of rows in the output.
    M : int, optional
      Number of columns in the output. If None, defaults to `N`.
    k : int, optional
      Index of the diagonal: 0 (the default) refers to the main diagonal,
      a positive value refers to an upper diagonal, and a negative value
      to a lower diagonal.
    dtype : data-type, optional
      Data-type of the returned array.

    Returns
    -------
    I : ndarray
      An array  of shape (N, M) where all elements are equal to zero, except
      for the `k`-th diagonal, whose values are equal to one.

    See Also
    --------
    numpy.eye

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if dtype is not None:
        dtype = np.dtype(dtype)
    if M is None:
        M = N
    k = operator.index(k)
    result = ndarray((N, M), dtype)
    result._thunk.eye(k)
    return result


def identity(n: int, dtype: npt.DTypeLike = float) -> ndarray:
    """

    Return the identity array.

    The identity array is a square array with ones on
    the main diagonal.

    Parameters
    ----------
    n : int
        Number of rows (and columns) in `n` x `n` output.
    dtype : data-type, optional
        Data-type of the output.  Defaults to ``float``.

    Returns
    -------
    out : ndarray
        `n` x `n` array with its main diagonal set to one, and all other
        elements 0.

    See Also
    --------
    numpy.identity

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return eye(N=n, M=n, dtype=dtype)


def ones(shape: NdShapeLike, dtype: npt.DTypeLike = np.float64) -> ndarray:
    """

    Return a new array of given shape and type, filled with ones.

    Parameters
    ----------
    shape : int or tuple[int]
        Shape of the new array.
    dtype : data-type, optional
        The desired data-type for the array. Default is `cunumeric.float64`.

    Returns
    -------
    out : ndarray
        Array of ones with the given shape and dtype.

    See Also
    --------
    numpy.ones

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return full(shape, 1, dtype=dtype)


def ones_like(
    a: ndarray,
    dtype: Optional[npt.DTypeLike] = None,
    shape: Optional[NdShapeLike] = None,
) -> ndarray:
    """

    Return an array of ones with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of the
        returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    shape : int or tuple[int], optional
        Overrides the shape of the result.

    Returns
    -------
    out : ndarray
        Array of ones with the same shape and type as `a`.

    See Also
    --------
    numpy.ones_like

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    usedtype = a.dtype
    if dtype is not None:
        usedtype = np.dtype(dtype)
    return full_like(a, 1, dtype=usedtype, shape=shape)


def zeros(shape: NdShapeLike, dtype: npt.DTypeLike = np.float64) -> ndarray:
    """
    zeros(shape, dtype=float)

    Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or tuple[int]
        Shape of the new array.
    dtype : data-type, optional
        The desired data-type for the array.  Default is `cunumeric.float64`.

    Returns
    -------
    out : ndarray
        Array of zeros with the given shape and dtype.

    See Also
    --------
    numpy.zeros

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if dtype is not None:
        dtype = np.dtype(dtype)
    return full(shape, 0, dtype=dtype)


def zeros_like(
    a: ndarray,
    dtype: Optional[npt.DTypeLike] = None,
    shape: Optional[NdShapeLike] = None,
) -> ndarray:
    """

    Return an array of zeros with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    shape : int or tuple[int], optional
        Overrides the shape of the result.

    Returns
    -------
    out : ndarray
        Array of zeros with the same shape and type as `a`.

    See Also
    --------
    numpy.zeros_like

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    usedtype = a.dtype
    if dtype is not None:
        usedtype = np.dtype(dtype)
    return full_like(a, 0, dtype=usedtype, shape=shape)


def full(
    shape: NdShapeLike,
    value: Union[int, float],
    dtype: Optional[npt.DTypeLike] = None,
) -> ndarray:
    """

    Return a new array of given shape and type, filled with `fill_value`.

    Parameters
    ----------
    shape : int or tuple[int]
        Shape of the new array.
    fill_value : scalar
        Fill value.
    dtype : data-type, optional
        The desired data-type for the array  The default, None, means
         `cunumeric.array(fill_value).dtype`.

    Returns
    -------
    out : ndarray
        Array of `fill_value` with the given shape and dtype.

    See Also
    --------
    numpy.full

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if dtype is None:
        val = np.array(value)
    else:
        dtype = np.dtype(dtype)
        val = np.array(value, dtype=dtype)
    result = empty(shape, dtype=val.dtype)
    result._thunk.fill(val)
    return result


def full_like(
    a: ndarray,
    value: Union[int, float],
    dtype: Optional[npt.DTypeLike] = None,
    shape: Optional[NdShapeLike] = None,
) -> ndarray:
    """

    Return a full array with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    fill_value : scalar
        Fill value.
    dtype : data-type, optional
        Overrides the data type of the result.
    shape : int or tuple[int], optional
        Overrides the shape of the result.

    Returns
    -------
    out : ndarray
        Array of `fill_value` with the same shape and type as `a`.

    See Also
    --------
    numpy.full_like

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if dtype is not None:
        dtype = np.dtype(dtype)
    else:
        dtype = a.dtype
    result = empty_like(a, dtype=dtype, shape=shape)
    val = np.array(value, dtype=result.dtype)
    result._thunk.fill(val)
    return result


# From existing data


def array(
    obj: Any,
    dtype: Optional[np.dtype[Any]] = None,
    copy: bool = True,
    order: Union[OrderType, Literal["K"]] = "K",
    subok: bool = False,
    ndmin: int = 0,
) -> ndarray:
    """
    array(object, dtype=None, copy=True)

    Create an array.

    Parameters
    ----------
    object : array_like
        An array, any object exposing the array interface, an object whose
        __array__ method returns an array, or any (nested) sequence.
    dtype : data-type, optional
        The desired data-type for the array.  If not given, then the type will
        be determined as the minimum type required to hold the objects in the
        sequence.
    copy : bool, optional
        If true (default), then the object is copied.  Otherwise, a copy will
        only be made if __array__ returns a copy, if obj is a nested sequence,
        or if a copy is needed to satisfy any of the other requirements
        (`dtype`, `order`, etc.).
    order : ``{'K', 'A', 'C', 'F'}``, optional
        Specify the memory layout of the array. If object is not an array, the
        newly created array will be in C order (row major) unless 'F' is
        specified, in which case it will be in Fortran order (column major).
        If object is an array the following holds.

        ===== ========= ===================================================
        order  no copy                     copy=True
        ===== ========= ===================================================
        'K'   unchanged F & C order preserved, otherwise most similar order
        'A'   unchanged F order if input is F and not C, otherwise C order
        'C'   C order   C order
        'F'   F order   F order
        ===== ========= ===================================================

        When ``copy=False`` and a copy is made for other reasons, the result is
        the same as if ``copy=True``, with some exceptions for 'A', see the
        Notes section. The default order is 'K'.
    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise
        the returned array will be forced to be a base-class array (default).
    ndmin : int, optional
        Specifies the minimum number of dimensions that the resulting
        array should have.  Ones will be pre-pended to the shape as
        needed to meet this requirement.

    Returns
    -------
    out : ndarray
        An array object satisfying the specified requirements.

    See Also
    --------
    numpy.array

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    if not isinstance(obj, ndarray):
        thunk = runtime.get_numpy_thunk(obj, share=(not copy), dtype=dtype)
        result = ndarray(shape=None, thunk=thunk)
    else:
        result = obj
    if dtype is not None and result.dtype != dtype:
        result = result.astype(dtype)
    elif copy and obj is result:
        result = result.copy()
    if result.ndim < ndmin:
        shape = (np.newaxis,) * (ndmin - result.ndim) + result.shape
        result = result.reshape(shape)
    return result


def asarray(a: Any, dtype: Optional[np.dtype[Any]] = None) -> ndarray:
    """
    Convert the input to an array.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array.  This
        includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists and ndarrays.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.

    Returns
    -------
    out : ndarray
        Array interpretation of `a`.  No copy is performed if the input is
        already an ndarray with matching dtype.  If `a` is a subclass of
        ndarray, a base class ndarray is returned.

    See Also
    --------
    numpy.asarray

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if not isinstance(a, ndarray):
        thunk = runtime.get_numpy_thunk(a, share=True, dtype=dtype)
        array = ndarray(shape=None, thunk=thunk)
    else:
        array = a
    if dtype is not None and array.dtype != dtype:
        array = array.astype(dtype)
    return array


@add_boilerplate("a")
def copy(a: ndarray) -> ndarray:
    """

    Return an array copy of the given object.

    Parameters
    ----------
    a : array_like
        Input data.

    Returns
    -------
    arr : ndarray
        Array interpretation of `a`.

    See Also
    --------
    numpy.copy

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    result = empty_like(a, dtype=a.dtype)
    result._thunk.copy(a._thunk, deep=True)
    return result


# Numerical ranges


def arange(
    start: Union[int, float] = 0,
    stop: Optional[Union[int, float]] = None,
    step: Optional[Union[int, float]] = 1,
    dtype: Optional[npt.DTypeLike] = None,
) -> ndarray:
    """
    arange([start,] stop[, step,], dtype=None)

    Return evenly spaced values within a given interval.

    Values are generated within the half-open interval ``[start, stop)``
    (in other words, the interval including `start` but excluding `stop`).
    For integer arguments the function is equivalent to the Python built-in
    `range` function, but returns an ndarray rather than a list.

    When using a non-integer step, such as 0.1, the results will often not
    be consistent.  It is better to use `cunumeric.linspace` for these cases.

    Parameters
    ----------
    start : int or float, optional
        Start of interval.  The interval includes this value.  The default
        start value is 0.
    stop : int or float
        End of interval.  The interval does not include this value, except
        in some cases where `step` is not an integer and floating point
        round-off affects the length of `out`.
    step : int or float, optional
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.  The default
        step size is 1.  If `step` is specified as a position argument,
        `start` must also be given.
    dtype : data-type
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.

    Returns
    -------
    arange : ndarray
        Array of evenly spaced values.

        For floating point arguments, the length of the result is
        ``ceil((stop - start)/step)``.  Because of floating point overflow,
        this rule may result in the last element of `out` being greater
        than `stop`.

    See Also
    --------
    numpy.arange

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if stop is None:
        stop = start
        start = 0

    if step is None:
        step = 1

    if dtype is None:
        dtype = np.array([stop]).dtype
    else:
        dtype = np.dtype(dtype)

    N = math.ceil((stop - start) / step)
    result = ndarray((N,), dtype)
    result._thunk.arange(start, stop, step)
    return result


@add_boilerplate("start", "stop")
def linspace(
    start: ndarray,
    stop: ndarray,
    num: int = 50,
    endpoint: bool = True,
    retstep: bool = False,
    dtype: Optional[npt.DTypeLike] = None,
    axis: int = 0,
) -> Union[ndarray, tuple[ndarray, Union[float, ndarray]]]:
    """

    Return evenly spaced numbers over a specified interval.

    Returns `num` evenly spaced samples, calculated over the
    interval [`start`, `stop`].

    The endpoint of the interval can optionally be excluded.

    Parameters
    ----------
    start : array_like
        The starting value of the sequence.
    stop : array_like
        The end value of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced samples, so that `stop` is excluded.  Note that the step
        size changes when `endpoint` is False.
    num : int, optional
        Number of samples to generate. Default is 50. Must be non-negative.
    endpoint : bool, optional
        If True, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    retstep : bool, optional
        If True, return (`samples`, `step`), where `step` is the spacing
        between samples.
    dtype : data-type, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.
    axis : int, optional
        The axis in the result to store the samples.  Relevant only if start
        or stop are array-like.  By default (0), the samples will be along a
        new axis inserted at the beginning. Use -1 to get an axis at the end.

    Returns
    -------
    samples : ndarray
        There are `num` equally spaced samples in the closed interval
        ``[start, stop]`` or the half-open interval ``[start, stop)``
        (depending on whether `endpoint` is True or False).
    step : float or ndarray, optional
        Only returned if `retstep` is True

        Size of spacing between samples.

    See Also
    --------
    numpy.linspace

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if num < 0:
        raise ValueError("Number of samples, %s, must be non-negative." % num)
    div = (num - 1) if endpoint else num

    common_kind = np.find_common_type((start.dtype, stop.dtype), ()).kind
    dt = np.complex128 if common_kind == "c" else np.float64
    if dtype is None:
        dtype = dt

    delta = stop - start
    y = arange(0, num, dtype=dt)

    out: tuple[Any, ...]  # EllipsisType not even in typing_extensions yet

    # Reshape these arrays into dimensions that allow them to broadcast
    if delta.ndim > 0:
        if axis is None or axis == 0:
            # First dimension
            y = y.reshape((-1,) + (1,) * delta.ndim)
            # Nothing else needs to be reshaped here because
            # they should all broadcast correctly with y
            if endpoint and num > 1:
                out = (-1,)
        elif axis == -1 or axis == delta.ndim:
            # Last dimension
            y = y.reshape((1,) * delta.ndim + (-1,))
            if endpoint and num > 1:
                out = (Ellipsis, -1)
            # Extend everything else with extra dimensions of 1 at the end
            # so that they can broadcast with y
            delta = delta.reshape(delta.shape + (1,))
            start = start.reshape(start.shape + (1,))
        elif axis < delta.ndim:
            # Somewhere in the middle
            y = y.reshape((1,) * axis + (-1,) + (1,) * (delta.ndim - axis))
            # Start array might be smaller than delta because of broadcast
            startax = start.ndim - len(delta.shape[axis:])
            start = start.reshape(
                start.shape[0:startax] + (1,) + start.shape[startax:]
            )
            if endpoint and num > 1:
                out = (Ellipsis, -1) + (slice(None, None, None),) * len(
                    delta.shape[axis:]
                )
            delta = delta.reshape(
                delta.shape[0:axis] + (1,) + delta.shape[axis:]
            )
        else:
            raise ValueError(
                "axis "
                + str(axis)
                + " is out of bounds for array of dimension "
                + str(delta.ndim + 1)
            )
    else:
        out = (-1,)
    # else delta is a scalar so start must be also
    # therefore it will trivially broadcast correctly

    step: Union[float, ndarray]
    if div > 0:
        step = delta / div
        if delta.ndim == 0:
            y *= step
        else:
            y = y * step
    else:
        # sequences with 0 items or 1 item with endpoint=True (i.e. div <= 0)
        # have an undefined step
        step = np.NaN
        if delta.ndim == 0:
            y *= delta
        else:
            y = y * delta

    y += start.astype(y.dtype, copy=False)

    if endpoint and num > 1:
        y[out] = stop.astype(y.dtype, copy=False)

    if np.issubdtype(dtype, np.integer):
        floor(y, out=y)

    if retstep:
        return y.astype(dtype, copy=False), step
    else:
        return y.astype(dtype, copy=False)


# Building matrices


@add_boilerplate("v")
def diag(v: ndarray, k: int = 0) -> ndarray:
    """

    Extract a diagonal or construct a diagonal array.

    See the more detailed documentation for ``cunumeric.diagonal`` if you use
    this function to extract a diagonal and wish to write to the resulting
    array; whether it returns a copy or a view depends on what version of numpy
    you are using.

    Parameters
    ----------
    v : array_like
        If `v` is a 2-D array, return a copy of its `k`-th diagonal.
        If `v` is a 1-D array, return a 2-D array with `v` on the `k`-th
        diagonal.
    k : int, optional
        Diagonal in question. The default is 0. Use `k>0` for diagonals
        above the main diagonal, and `k<0` for diagonals below the main
        diagonal.

    Returns
    -------
    out : ndarray
        The extracted diagonal or constructed diagonal array.

    See Also
    --------
    numpy.diag

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if v.ndim == 0:
        raise ValueError("Input must be 1- or 2-d")
    elif v.ndim == 1:
        return v.diagonal(offset=k, axis1=0, axis2=1, extract=False)
    elif v.ndim == 2:
        return v.diagonal(offset=k, axis1=0, axis2=1, extract=True)
    else:
        raise ValueError("diag requires 1- or 2-D array, use diagonal instead")


def tri(
    N: int,
    M: Optional[int] = None,
    k: int = 0,
    dtype: npt.DTypeLike = float,
    *,
    like: Optional[ndarray] = None,
) -> ndarray:
    """
    An array with ones at and below the given diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
        Number of rows in the array.
    M : int, optional
        Number of columns in the array.
        By default, `M` is taken equal to `N`.
    k : int, optional
        The sub-diagonal at and below which the array is filled.
        `k` = 0 is the main diagonal, while `k` < 0 is below it,
        and `k` > 0 is above.  The default is 0.
    dtype : dtype, optional
        Data type of the returned array.  The default is float.
    like : array_like
        Reference object to allow the creation of arrays which are not NumPy
        arrays. If an array-like passed in as `like` supports the
        `__array_function__` protocol, the result will be defined by it. In
        this case it ensures the creation of an array object compatible with
        that passed in via this argument.

    Returns
    -------
    tri : ndarray of shape (N, M)
        Array with its lower triangle filled with ones and zero elsewhere;
        in other words ``T[i,j] == 1`` for ``j <= i + k``, 0 otherwise.

    See Also
    --------
    numpy.tri

    Notes
    -----
    `like` argument is currently not supported

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    # TODO: add support for `like` (see issue #418)
    if like is not None:
        raise ValueError("like parameter is currently not supported")

    if M is None:
        M = N

    out = ones((N, M), dtype=dtype)
    return tril(out, k)


@add_boilerplate("m")
def trilu(m: ndarray, k: int, lower: bool) -> ndarray:
    if m.ndim < 1:
        raise TypeError("Array must be at least 1-D")
    shape = m.shape if m.ndim >= 2 else m.shape * 2
    result = ndarray(shape, dtype=m.dtype, inputs=(m,))
    result._thunk.trilu(m._thunk, k, lower)
    return result


def tril(m: ndarray, k: int = 0) -> ndarray:
    """

    Lower triangle of an array.

    Return a copy of an array with elements above the `k`-th diagonal zeroed.

    Parameters
    ----------
    m : array_like
        Input array of shape (M, N).
    k : int, optional
        Diagonal above which to zero elements.  `k = 0` (the default) is the
        main diagonal, `k < 0` is below it and `k > 0` is above.

    Returns
    -------
    tril : ndarray
        Lower triangle of `m`, of same shape and data-type as `m`.

    See Also
    --------
    numpy.tril

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return trilu(m, k, True)


def triu(m: ndarray, k: int = 0) -> ndarray:
    """

    Upper triangle of an array.

    Return a copy of a matrix with the elements below the `k`-th diagonal
    zeroed.

    Please refer to the documentation for `tril` for further details.

    See Also
    --------
    numpy.triu

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return trilu(m, k, False)


#############################
# Array manipulation routines
#############################

# Basic operations


@add_boilerplate("a")
def ndim(a: ndarray) -> int:
    """

    Return the number of dimensions of an array.

    Parameters
    ----------
    a : array_like
        Input array.  If it is not already an ndarray, a conversion is
        attempted.

    Returns
    -------
    number_of_dimensions : int
        The number of dimensions in `a`.  Scalars are zero-dimensional.

    See Also
    --------
    ndarray.ndim : equivalent method
    shape : dimensions of array
    ndarray.shape : dimensions of array

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return 0 if a is None else a.ndim


@add_boilerplate("a")
def shape(a: ndarray) -> NdShape:
    """

    Return the shape of an array.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    shape : tuple[int, ...]
        The elements of the shape tuple give the lengths of the
        corresponding array dimensions.

    See Also
    --------
    numpy.shape

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.shape


# Changing array shape


@add_boilerplate("a")
def ravel(a: ndarray, order: OrderType = "C") -> ndarray:
    """
    Return a contiguous flattened array.

    A 1-D array, containing the elements of the input, is returned.  A copy is
    made only if needed.

    Parameters
    ----------
    a : array_like
        Input array.  The elements in `a` are read in the order specified by
        `order`, and packed as a 1-D array.
    order : ``{'C','F', 'A', 'K'}``, optional
        The elements of `a` are read using this index order. 'C' means
        to index the elements in row-major, C-style order,
        with the last axis index changing fastest, back to the first
        axis index changing slowest.  'F' means to index the elements
        in column-major, Fortran-style order, with the
        first index changing fastest, and the last index changing
        slowest. Note that the 'C' and 'F' options take no account of
        the memory layout of the underlying array, and only refer to
        the order of axis indexing.  'A' means to read the elements in
        Fortran-like index order if `a` is Fortran *contiguous* in
        memory, C-like order otherwise.  'K' means to read the
        elements in the order they occur in memory, except for
        reversing the data when strides are negative.  By default, 'C'
        index order is used.

    Returns
    -------
    y : array_like
        y is an array of the same subtype as `a`, with shape ``(a.size,)``.
        Note that matrices are special cased for backward compatibility, if `a`
        is a matrix, then y is a 1-D ndarray.

    See Also
    --------
    numpy.ravel

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.ravel(order=order)


@add_boilerplate("a")
def reshape(
    a: ndarray, newshape: NdShapeLike, order: OrderType = "C"
) -> ndarray:
    """

    Gives a new shape to an array without changing its data.

    Parameters
    ----------
    a : array_like
        Array to be reshaped.
    newshape : int or tuple[int]
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is
        inferred from the length of the array and remaining dimensions.
    order : ``{'C', 'F', 'A'}``, optional
        Read the elements of `a` using this index order, and place the
        elements into the reshaped array using this index order.  'C'
        means to read / write the elements using C-like index order,
        with the last axis index changing fastest, back to the first
        axis index changing slowest. 'F' means to read / write the
        elements using Fortran-like index order, with the first index
        changing fastest, and the last index changing slowest. Note that
        the 'C' and 'F' options take no account of the memory layout of
        the underlying array, and only refer to the order of indexing.
        'A' means to read / write the elements in Fortran-like index
        order if `a` is Fortran *contiguous* in memory, C-like order
        otherwise.

    Returns
    -------
    reshaped_array : ndarray
        This will be a new view object if possible; otherwise, it will
        be a copy.  Note there is no guarantee of the *memory layout* (C- or
        Fortran- contiguous) of the returned array.

    See Also
    --------
    numpy.reshape

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.reshape(newshape, order=order)


# Transpose-like operations


@add_boilerplate("a")
def swapaxes(a: ndarray, axis1: int, axis2: int) -> ndarray:
    """

    Interchange two axes of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    axis1 : int
        First axis.
    axis2 : int
        Second axis.

    Returns
    -------
    a_swapped : ndarray
        If `a` is an ndarray, then a view of `a` is returned; otherwise a new
        array is created.

    See Also
    --------
    numpy.swapaxes

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.swapaxes(axis1, axis2)


@add_boilerplate("a")
def transpose(a: ndarray, axes: Optional[list[int]] = None) -> ndarray:
    """

    Permute the dimensions of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    axes : list[int], optional
        By default, reverse the dimensions, otherwise permute the axes
        according to the values given.

    Returns
    -------
    p : ndarray
        `a` with its axes permuted.  A view is returned whenever
        possible.

    See Also
    --------
    numpy.transpose

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.transpose(axes=axes)


@add_boilerplate("a")
def moveaxis(
    a: ndarray, source: Sequence[int], destination: Sequence[int]
) -> ndarray:
    """
    Move axes of an array to new positions.
    Other axes remain in their original order.

    Parameters
    ----------
    a : ndarray
        The array whose axes should be reordered.
    source : int or Sequence[int]
        Original positions of the axes to move. These must be unique.
    destination : int or Sequence[int]
        Destination positions for each of the original axes. These must also be
        unique.

    Returns
    -------
    result : ndarray
        Array with moved axes. This array is a view of the input array.

    See Also
    --------
    numpy.moveaxis

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    source = normalize_axis_tuple(source, a.ndim, "source")
    destination = normalize_axis_tuple(destination, a.ndim, "destination")
    if len(source) != len(destination):
        raise ValueError(
            "`source` and `destination` arguments must have the same number "
            "of elements"
        )
    order = [n for n in range(a.ndim) if n not in source]
    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)
    return a.transpose(order)


# Changing number of dimensions


def _reshape_recur(ndim: int, arr: ndarray) -> tuple[int, ...]:
    if arr.ndim < ndim:
        cur_shape: tuple[int, ...] = _reshape_recur(ndim - 1, arr)
        if ndim == 2:
            cur_shape = (1,) + cur_shape
        else:
            cur_shape = cur_shape + (1,)
    else:
        cur_shape = arr.shape
    return cur_shape


def _atleast_nd(
    ndim: int, arys: Sequence[ndarray]
) -> Union[list[ndarray], ndarray]:
    inputs = list(convert_to_cunumeric_ndarray(arr) for arr in arys)
    # 'reshape' change the shape of arrays
    # only when arr.shape != _reshape_recur(ndim,arr)
    result = list(arr.reshape(_reshape_recur(ndim, arr)) for arr in inputs)
    # if the number of arrys in `arys` is 1, the return value is a single array
    if len(result) == 1:
        return result[0]
    return result


def atleast_1d(*arys: ndarray) -> Union[list[ndarray], ndarray]:
    """

    Convert inputs to arrays with at least one dimension.
    Scalar inputs are converted to 1-dimensional arrays,
    whilst higher-dimensional inputs are preserved.

    Parameters
    ----------
    *arys : array_like
        One or more input arrays.

    Returns
    -------
    ret : ndarray
        An array, or list of arrays, each with a.ndim >= 1.
        Copies are made only if necessary.

    See Also
    --------
    numpy.atleast_1d

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return _atleast_nd(1, arys)


def atleast_2d(*arys: ndarray) -> Union[list[ndarray], ndarray]:
    """

    View inputs as arrays with at least two dimensions.

    Parameters
    ----------
    *arys : array_like
        One or more array-like sequences.
        Non-array inputs are converted to arrays.
        Arrays that already have two or more dimensions are preserved.

    Returns
    -------
    res, res2, … : ndarray
        An array, or list of arrays, each with a.ndim >= 2.
        Copies are avoided where possible, and
        views with two or more dimensions are returned.

    See Also
    --------
    numpy.atleast_2d

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return _atleast_nd(2, arys)


def atleast_3d(*arys: ndarray) -> Union[list[ndarray], ndarray]:
    """

    View inputs as arrays with at least three dimensions.

    Parameters
    ----------
    *arys : array_like
        One or more array-like sequences.
        Non-array inputs are converted to arrays.
        Arrays that already have three or more dimensions are preserved.

    Returns
    -------
    res, res2, … : ndarray
        An array, or list of arrays, each with a.ndim >= 3.
        Copies are avoided where possible, and
        views with three or more dimensions are returned.
        For example, a 1-D array of shape (N,) becomes
        a view of shape (1, N, 1),  and a 2-D array of shape (M, N)
        becomes a view of shape (M, N, 1).

    See Also
    --------
    numpy.atleast_3d

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return _atleast_nd(3, arys)


@add_boilerplate("a")
def squeeze(a: ndarray, axis: Optional[NdShapeLike] = None) -> ndarray:
    """

    Remove single-dimensional entries from the shape of an array.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple[int], optional
        Selects a subset of the single-dimensional entries in the
        shape. If an axis is selected with shape entry greater than
        one, an error is raised.

    Returns
    -------
    squeezed : ndarray
        The input array, but with all or a subset of the
        dimensions of length 1 removed. This is always `a` itself
        or a view into `a`.

    Raises
    ------
    ValueError
        If `axis` is not None, and an axis being squeezed is not of length 1

    See Also
    --------
    numpy.squeeze

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.squeeze(axis=axis)


# Joining arrays


class ArrayInfo:
    def __init__(
        self, ndim: int, shape: NdShape, dtype: np.dtype[Any]
    ) -> None:
        self.ndim = ndim
        self.shape = shape
        self.dtype = dtype


def convert_to_array_form(indices: Sequence[int]) -> str:
    return "".join(f"[{coord}]" for coord in indices)


def check_list_depth(arr: Any, prefix: NdShape = (0,)) -> int:
    if not isinstance(arr, list):
        return 0
    elif len(arr) == 0:
        raise ValueError(
            f"List at arrays{convert_to_array_form(prefix)} cannot be empty"
        )

    depths = list(
        check_list_depth(each, prefix + (idx,)) for idx, each in enumerate(arr)
    )

    if len(set(depths)) != 1:  # this should be one
        # If we're here elements don't have the same depth
        first_depth = depths[0]
        for idx, other_depth in enumerate(depths[1:]):
            if other_depth != first_depth:
                raise ValueError(
                    "List depths are mismatched. First element was at depth "
                    f"{first_depth}, but there is an element at"
                    f" depth {other_depth}, "
                    f"arrays{convert_to_array_form(prefix+(idx+1,))}"
                )

    return depths[0] + 1


def check_shape_with_axis(
    inputs: list[ndarray],
    func_name: str,
    axis: int,
) -> None:
    ndim = inputs[0].ndim
    shape = inputs[0].shape

    axis = normalize_axis_index(axis, ndim)
    if ndim >= 1:
        if _builtin_any(
            shape[:axis] != inp.shape[:axis]
            or shape[axis + 1 :] != inp.shape[axis + 1 :]
            for inp in inputs
        ):
            raise ValueError(
                f"All arguments to {func_name} "
                "must have the same "
                "dimension size in all dimensions "
                "except the target axis"
            )
    return


def check_shape_dtype_without_axis(
    inputs: Sequence[ndarray],
    func_name: str,
    dtype: Optional[npt.DTypeLike] = None,
    casting: CastingKind = "same_kind",
) -> tuple[list[ndarray], ArrayInfo]:
    if len(inputs) == 0:
        raise ValueError("need at least one array to concatenate")

    inputs = list(convert_to_cunumeric_ndarray(inp) for inp in inputs)
    ndim = inputs[0].ndim
    shape = inputs[0].shape

    if _builtin_any(ndim != inp.ndim for inp in inputs):
        raise ValueError(
            f"All arguments to {func_name} "
            "must have the same number of dimensions"
        )

    # Cast arrays with the passed arguments (dtype, casting)
    if dtype is None:
        dtype = np.find_common_type((inp.dtype for inp in inputs), [])
    else:
        dtype = np.dtype(dtype)

    converted = list(inp.astype(dtype, casting=casting) for inp in inputs)
    return converted, ArrayInfo(ndim, shape, dtype)


def _block_collect_slices(
    arr: Union[ndarray, Sequence[ndarray]], cur_depth: int, depth: int
) -> tuple[list[Any], list[tuple[slice, ...]], Sequence[ndarray]]:
    # collects slices for each array in `arr`
    # the outcome will be slices on every dimension of the output array
    # for each array in `arr`
    if cur_depth < depth:
        sublist_results = list(
            _block_collect_slices(each, cur_depth + 1, depth) for each in arr
        )
        # 'sublist_results' contains a list of 3-way tuples,
        # for arrays, out_shape of the sublist, and slices
        arrays, outshape_list, slices = zip(*sublist_results)
        max_ndim = _builtin_max(
            1 + (depth - cur_depth), *(len(each) for each in outshape_list)
        )
        outshape_list = list(
            ((1,) * (max_ndim - len(each)) + tuple(each))
            for each in outshape_list
        )
        leading_dim = _builtin_sum(
            each[-1 + (cur_depth - depth)] for each in outshape_list
        )
        # flatten array lists from sublists into a single list
        arrays = list(chain(*arrays))
        # prepares the out_shape of the current list
        out_shape = list(outshape_list[0])
        out_shape[-1 + cur_depth - depth] = leading_dim
        offset = 0
        updated_slices = []
        # update the dimension in each slice for the current axis
        for shape, slice_list in zip(outshape_list, slices):
            cur_dim = shape[-1 + cur_depth - depth]
            updated_slices.append(
                list(
                    (slice(offset, offset + cur_dim),) + each
                    for each in slice_list
                )
            )
            offset += cur_dim
        # flatten lists of slices into a single list
        slices = list(chain(*updated_slices))
    else:
        arrays = list(convert_to_cunumeric_ndarray(inp) for inp in arr)
        common_shape = arrays[0].shape
        if len(arr) > 1:
            arrays, common_info = check_shape_dtype_without_axis(
                arrays, block.__name__
            )
            common_shape = common_info.shape
            check_shape_with_axis(arrays, block.__name__, axis=-1)
        # the initial slices for each arr on arr.shape[-1]
        out_shape, slices, arrays = _collect_outshape_slices(
            arrays, common_shape, axis=-1 + len(common_shape)
        )

    return arrays, out_shape, slices


def _block_slicing(arrays: Sequence[ndarray], depth: int) -> ndarray:
    # collects the final slices of input arrays and assign them at once
    arrays, out_shape, slices = _block_collect_slices(arrays, 1, depth)
    out_array = ndarray(shape=out_shape, inputs=arrays)

    for dest, inp in zip(slices, arrays):
        out_array[(Ellipsis,) + tuple(dest)] = inp

    return out_array


def _collect_outshape_slices(
    inputs: Sequence[ndarray], common_shape: NdShape, axis: int
) -> tuple[list[Any], list[tuple[slice, ...]], Sequence[ndarray]]:
    leading_dim = _builtin_sum(arr.shape[axis] for arr in inputs)
    out_shape = list(common_shape)
    out_shape[axis] = leading_dim
    post_idx = (slice(None),) * len(out_shape[axis + 1 :])
    slices = []
    offset = 0
    # collect slices for arrays in `inputs`
    inputs = list(inp for inp in inputs if inp.size > 0)
    for inp in inputs:
        slices.append((slice(offset, offset + inp.shape[axis]),) + post_idx)
        offset += inp.shape[axis]

    return out_shape, slices, inputs


def _concatenate(
    inputs: Sequence[ndarray],
    common_info: ArrayInfo,
    axis: int = 0,
    out: Optional[ndarray] = None,
    dtype: Optional[npt.DTypeLike] = None,
    casting: CastingKind = "same_kind",
) -> ndarray:
    if axis < 0:
        axis += len(common_info.shape)
    out_shape, slices, inputs = _collect_outshape_slices(
        inputs, common_info.shape, axis
    )

    if out is None:
        out_array = ndarray(
            shape=out_shape, dtype=common_info.dtype, inputs=inputs
        )
    else:
        out = convert_to_cunumeric_ndarray(out)
        if not isinstance(out, ndarray):
            raise TypeError("out should be ndarray")
        elif list(out.shape) != out_shape:
            raise ValueError(
                f"out.shape({out.shape}) is not matched "
                f"to the result shape of concatenation ({out_shape})"
            )
        out_array = out

    for dest, src in zip(slices, inputs):
        out_array[(Ellipsis,) + dest] = src

    return out_array


def append(
    arr: ndarray, values: ndarray, axis: Optional[int] = None
) -> ndarray:
    """

    Append values to the end of an array.

    Parameters
    ----------
    arr :  array_like
        Values are appended to a copy of this array.
    values : array_like
        These values are appended to a copy of arr. It must be of the correct
        shape (the same shape as arr, excluding axis). If axis is not
        specified, values can be any shape and will be flattened before use.
    axis : int, optional
        The axis along which values are appended. If axis is not given, both
        `arr` and `values` are flattened before use.

    Returns
    -------
    res : ndarray
        A copy of arr with values appended to axis.

    See Also
    --------
    numpy.append

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    # Check to see if we can build a new tuple of cuNumeric arrays
    inputs = list(convert_to_cunumeric_ndarray(inp) for inp in [arr, values])
    return concatenate(inputs, axis)


def block(arrays: Sequence[Any]) -> ndarray:
    """
    Assemble an nd-array from nested lists of blocks.

    Blocks in the innermost lists are concatenated (see concatenate)
    along the last dimension (-1), then these are concatenated along
    the second-last dimension (-2), and so on until the outermost
    list is reached.

    Blocks can be of any dimension, but will not be broadcasted using
    the normal rules. Instead, leading axes of size 1 are inserted,
    to make block.ndim the same for all blocks. This is primarily useful
    for working with scalars, and means that code like np.block([v, 1])
    is valid, where v.ndim == 1.

    When the nested list is two levels deep, this allows block matrices
    to be constructed from their components.

    Parameters
    ----------
    arrays : nested list of array_like or scalars
        If passed a single ndarray or scalar (a nested list of depth 0),
        this is returned unmodified (and not copied).

        Elements shapes must match along the appropriate axes (without
        broadcasting), but leading 1s will be prepended to the shape as
        necessary to make the dimensions match.

    Returns
    -------
    block_array : ndarray
        The array assembled from the given blocks.
        The dimensionality of the output is equal to the greatest of: * the
        dimensionality of all the inputs * the depth to which the input list
        is nested

    Raises
    ------
    ValueError
        If list depths are mismatched - for instance, [[a, b], c] is
        illegal, and should be spelt [[a, b], [c]]
        If lists are empty - for instance, [[a, b], []]

    See Also
    --------
    numpy.block

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    # arrays should concatenate from innermost subarrays
    # the 'arrays' should be balanced tree
    # check if the 'arrays' is a balanced tree
    depth = check_list_depth(arrays)

    result = _block_slicing(arrays, depth)
    return result


def concatenate(
    inputs: Sequence[ndarray],
    axis: Union[int, None] = 0,
    out: Optional[ndarray] = None,
    dtype: Optional[npt.DTypeLike] = None,
    casting: CastingKind = "same_kind",
) -> ndarray:
    """

    concatenate((a1, a2, ...), axis=0, out=None, dtype=None,
    casting="same_kind")

    Join a sequence of arrays along an existing axis.

    Parameters
    ----------
    a1, a2, ... : Sequence[array_like]
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).
    axis : int, optional
        The axis along which the arrays will be joined.  If axis is None,
        arrays are flattened before use.  Default is 0.
    out : ndarray, optional
        If provided, the destination to place the result. The shape must be
        correct, matching that of what concatenate would have returned if no
        out argument were specified.
    dtype : str or data-type
        If provided, the destination array will have this dtype. Cannot be
        provided together with `out`.
    casting : ``{'no', 'equiv', 'safe', 'same_kind', 'unsafe'}``, optional
        Controls what kind of data casting may occur. Defaults to 'same_kind'.

    Returns
    -------
    res : ndarray
        The concatenated array.

    See Also
    --------
    numpy.concatenate

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if dtype is not None and out is not None:
        raise TypeError(
            "concatenate() only takes `out` or `dtype` as an argument,"
            "but both were provided."
        )

    if casting not in casting_kinds:
        raise ValueError(
            "casting must be one of 'no', 'equiv', "
            "'safe', 'same_kind', or 'unsafe'"
        )

    # flatten arrays if axis == None and concatenate arrays on the first axis
    if axis is None:
        # Reshape arrays in the `array_list` to handle scalars
        reshaped = _atleast_nd(1, inputs)
        if not isinstance(reshaped, list):
            reshaped = [reshaped]
        inputs = list(inp.ravel() for inp in reshaped)
        axis = 0

    # Check to see if we can build a new tuple of cuNumeric arrays
    cunumeric_inputs, common_info = check_shape_dtype_without_axis(
        inputs, concatenate.__name__, dtype, casting
    )
    check_shape_with_axis(cunumeric_inputs, concatenate.__name__, axis)

    return _concatenate(
        cunumeric_inputs,
        common_info,
        axis,
        out,
        dtype,
        casting,
    )


def stack(
    arrays: Sequence[ndarray], axis: int = 0, out: Optional[ndarray] = None
) -> ndarray:
    """

    Join a sequence of arrays along a new axis.

    The ``axis`` parameter specifies the index of the new axis in the
    dimensions of the result. For example, if ``axis=0`` it will be the first
    dimension and if ``axis=-1`` it will be the last dimension.

    Parameters
    ----------
    arrays : Sequence[array_like]
        Each array must have the same shape.

    axis : int, optional
        The axis in the result array along which the input arrays are stacked.

    out : ndarray, optional
        If provided, the destination to place the result. The shape must be
        correct, matching that of what stack would have returned if no
        out argument were specified.

    Returns
    -------
    stacked : ndarray
        The stacked array has one more dimension than the input arrays.

    See Also
    --------
    numpy.stack

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if type(axis) is not int:
        raise TypeError("The target axis should be an integer")

    arrays, common_info = check_shape_dtype_without_axis(
        arrays, stack.__name__
    )
    shapes = {inp.shape for inp in arrays}
    if len(shapes) != 1:
        raise ValueError("all input arrays must have the same shape for stack")

    axis = normalize_axis_index(axis, common_info.ndim + 1)
    shape = common_info.shape[:axis] + (1,) + common_info.shape[axis:]
    arrays = [arr.reshape(shape) for arr in arrays]
    common_info.shape = tuple(shape)
    return _concatenate(arrays, common_info, axis, out=out)


def vstack(tup: Sequence[ndarray]) -> ndarray:
    """

    Stack arrays in sequence vertically (row wise).

    This is equivalent to concatenation along the first axis after 1-D arrays
    of shape `(N,)` have been reshaped to `(1,N)`. Rebuilds arrays divided by
    `vsplit`.

    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    Parameters
    ----------
    tup : Sequence[ndarray]
        The arrays must have the same shape along all but the first axis.
        1-D arrays must have the same length.

    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays, will be at least 2-D.

    See Also
    --------
    numpy.vstack

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    # Reshape arrays in the `array_list` if needed before concatenation
    reshaped = _atleast_nd(2, tup)
    if not isinstance(reshaped, list):
        reshaped = [reshaped]
    tup, common_info = check_shape_dtype_without_axis(
        reshaped, vstack.__name__
    )
    check_shape_with_axis(tup, vstack.__name__, 0)
    return _concatenate(
        tup,
        common_info,
        axis=0,
        dtype=common_info.dtype,
    )


def hstack(tup: Sequence[ndarray]) -> ndarray:
    """

    Stack arrays in sequence horizontally (column wise).

    This is equivalent to concatenation along the second axis, except for 1-D
    arrays where it concatenates along the first axis. Rebuilds arrays divided
    by `hsplit`.

    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    Parameters
    ----------
    tup : Sequence[ndarray]
        The arrays must have the same shape along all but the second axis,
        except 1-D arrays which can be any length.

    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays.

    See Also
    --------
    numpy.hstack

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    # Reshape arrays in the `array_list` to handle scalars
    reshaped = _atleast_nd(1, tup)
    if not isinstance(reshaped, list):
        reshaped = [reshaped]

    tup, common_info = check_shape_dtype_without_axis(
        reshaped, hstack.__name__
    )
    check_shape_with_axis(
        tup, hstack.__name__, axis=(0 if common_info.ndim == 1 else 1)
    )
    # When ndim == 1, hstack concatenates arrays along the first axis
    return _concatenate(
        tup,
        common_info,
        axis=(0 if common_info.ndim == 1 else 1),
        dtype=common_info.dtype,
    )


def dstack(tup: Sequence[ndarray]) -> ndarray:
    """

    Stack arrays in sequence depth wise (along third axis).

    This is equivalent to concatenation along the third axis after 2-D arrays
    of shape `(M,N)` have been reshaped to `(M,N,1)` and 1-D arrays of shape
    `(N,)` have been reshaped to `(1,N,1)`. Rebuilds arrays divided by
    `dsplit`.

    This function makes most sense for arrays with up to 3 dimensions. For
    instance, for pixel-data with a height (first axis), width (second axis),
    and r/g/b channels (third axis). The functions `concatenate`, `stack` and
    `block` provide more general stacking and concatenation operations.

    Parameters
    ----------
    tup : Sequence[ndarray]
        The arrays must have the same shape along all but the third axis.
        1-D or 2-D arrays must have the same shape.

    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays, will be at least 3-D.

    See Also
    --------
    numpy.dstack

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    # Reshape arrays to (1,N,1) for ndim ==1 or (M,N,1) for ndim == 2:
    reshaped = _atleast_nd(3, tup)
    if not isinstance(reshaped, list):
        reshaped = [reshaped]
    tup, common_info = check_shape_dtype_without_axis(
        reshaped, dstack.__name__
    )
    check_shape_with_axis(tup, dstack.__name__, 2)
    return _concatenate(
        tup,
        common_info,
        axis=2,
        dtype=common_info.dtype,
    )


def column_stack(tup: Sequence[ndarray]) -> ndarray:
    """

    Stack 1-D arrays as columns into a 2-D array.

    Take a sequence of 1-D arrays and stack them as columns
    to make a single 2-D array. 2-D arrays are stacked as-is,
    just like with `hstack`.  1-D arrays are turned into 2-D columns
    first.

    Parameters
    ----------
    tup : Sequence[ndarray]
        1-D or 2-D arrays to stack. All of them must have the same
        first dimension.

    Returns
    -------
    stacked : ndarray
        The 2-D array formed by stacking the given arrays.

    See Also
    --------
    numpy.column_stack

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    # Reshape arrays in the `array_list` to handle scalars
    reshaped = _atleast_nd(1, tup)
    if not isinstance(reshaped, list):
        reshaped = [reshaped]

    tup, common_info = check_shape_dtype_without_axis(
        reshaped, column_stack.__name__
    )

    if common_info.ndim == 1:
        tup = list(inp.reshape((inp.shape[0], 1)) for inp in tup)
        common_info.shape = tup[0].shape
    check_shape_with_axis(tup, column_stack.__name__, 1)
    return _concatenate(
        tup,
        common_info,
        axis=1,
        dtype=common_info.dtype,
    )


row_stack = vstack


# Splitting arrays


def split(
    a: ndarray, indices: Union[int, ndarray], axis: int = 0
) -> list[ndarray]:
    """

    Split an array into multiple sub-arrays as views into `ary`.

    Parameters
    ----------
    ary : ndarray
        Array to be divided into sub-arrays.
    indices_or_sections : int or ndarray
        If `indices_or_sections` is an integer, N, the array will be divided
        into N equal arrays along `axis`.  If such a split is not possible,
        an error is raised.

        If `indices_or_sections` is a 1-D array of sorted integers, the entries
        indicate where along `axis` the array is split.  For example,
        ``[2, 3]`` would, for ``axis=0``, result in

          - ary[:2]
          - ary[2:3]
          - ary[3:]

        If an index exceeds the dimension of the array along `axis`,
        an empty sub-array is returned correspondingly.
    axis : int, optional
        The axis along which to split, default is 0.

    Returns
    -------
    sub-arrays : list[ndarray]
        A list of sub-arrays as views into `ary`.

    Raises
    ------
    ValueError
        If `indices_or_sections` is given as an integer, but
        a split does not result in equal division.

    See Also
    --------
    numpy.split

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return array_split(a, indices, axis, equal=True)


def array_split(
    a: ndarray,
    indices: Union[int, tuple[int], ndarray, npt.NDArray[Any]],
    axis: int = 0,
    equal: bool = False,
) -> list[ndarray]:
    """

    Split an array into multiple sub-arrays.

    Please refer to the ``split`` documentation.  The only difference
    between these functions is that ``array_split`` allows
    `indices_or_sections` to be an integer that does *not* equally
    divide the axis. For an array of length l that should be split
    into n sections, it returns l % n sub-arrays of size l//n + 1
    and the rest of size l//n.

    See Also
    --------
    numpy.array_split

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    array = convert_to_cunumeric_ndarray(a)
    split_pts = []
    if axis >= array.ndim:
        raise ValueError(
            f"array({array.shape}) has less dimensions than axis({axis})"
        )

    if isinstance(indices, int):
        if indices <= 0:
            raise ValueError("number sections must be larger than 0.")
        res = array.shape[axis] % indices
        if equal and res != 0:
            raise ValueError("array split does not result in an equal divison")

        len_subarr = array.shape[axis] // indices
        end_idx = array.shape[axis]
        first_idx = len_subarr

        # the requested # of subarray is larger than the size of array
        # -> size of 1 subarrays + empty subarrays
        if len_subarr == 0:
            len_subarr = 1
            first_idx = len_subarr
            end_idx = indices
        else:
            if res != 0:
                # The first 'res' groups have len_subarr+1 elements
                split_pts = list(
                    range(
                        len_subarr + 1, (len_subarr + 1) * res, len_subarr + 1
                    )
                )
                first_idx = (len_subarr + 1) * res
        split_pts.extend(range(first_idx, end_idx + 1, len_subarr))

    elif isinstance(indices, (list, tuple)) or (
        isinstance(indices, (ndarray, np.ndarray)) and indices.dtype == int
    ):
        split_pts = list(indices)
        # adding the size of the target dimension.
        # This helps create dummy or last subarray correctly
        split_pts.append(array.shape[axis])

    else:
        raise ValueError("Integer or array for split should be provided")

    result = []
    start_idx = 0
    end_idx = 0
    out_shape = []
    in_shape: list[Union[int, slice]] = []

    for i in range(array.ndim):
        if i != axis:
            in_shape.append(slice(array.shape[i]))
            out_shape.append(array.shape[i])
        else:
            in_shape.append(1)
            out_shape.append(1)

    for pts in split_pts:
        if type(pts) is not int:
            raise ValueError(
                "Split points in the passed `indices` should be integer"
            )
        end_idx = pts
        # For a split point, which is larger than the dimension for splitting,
        # The last non-empty subarray should be copied from
        # array[last_elem:array.shape[axis]]
        if pts > array.shape[axis]:
            end_idx = array.shape[axis]
        out_shape[axis] = (end_idx - start_idx) + 1
        in_shape[axis] = slice(start_idx, end_idx)
        new_subarray = None
        if start_idx < array.shape[axis] and start_idx < end_idx:
            new_subarray = array[tuple(in_shape)].view()
        else:
            out_shape[axis] = 0
            new_subarray = ndarray(tuple(out_shape), dtype=array.dtype)
        result.append(new_subarray)
        start_idx = pts

    return result


def dsplit(a: ndarray, indices: Union[int, ndarray]) -> list[ndarray]:
    """

    Split array into multiple sub-arrays along the 3rd axis (depth).

    Please refer to the `split` documentation.  `dsplit` is equivalent
    to `split` with ``axis=2``, the array is always split along the third
    axis provided the array dimension is greater than or equal to 3.

    See Also
    --------
    numpy.dsplit

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return split(a, indices, axis=2)


def hsplit(a: ndarray, indices: Union[int, ndarray]) -> list[ndarray]:
    """

    Split an array into multiple sub-arrays horizontally (column-wise).

    Please refer to the `split` documentation.  `hsplit` is equivalent
    to `split` with ``axis=1``, the array is always split along the second
    axis regardless of the array dimension.

    See Also
    --------
    numpy.hsplit

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return split(a, indices, axis=1)


def vsplit(a: ndarray, indices: Union[int, ndarray]) -> list[ndarray]:
    """

    Split an array into multiple sub-arrays vertically (row-wise).

    Please refer to the ``split`` documentation.  ``vsplit`` is equivalent
    to ``split`` with `axis=0` (default), the array is always split along the
    first axis regardless of the array dimension.

    See Also
    --------
    numpy.vsplit

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return split(a, indices, axis=0)


# Tiling arrays


@add_boilerplate("A")
def tile(
    A: ndarray, reps: Union[int, Sequence[int], npt.NDArray[np.int_]]
) -> ndarray:
    """
    Construct an array by repeating A the number of times given by reps.

    If `reps` has length ``d``, the result will have dimension of ``max(d,
    A.ndim)``.

    If ``A.ndim < d``, `A` is promoted to be d-dimensional by prepending new
    axes. So a shape (3,) array is promoted to (1, 3) for 2-D replication,
    or shape (1, 1, 3) for 3-D replication. If this is not the desired
    behavior, promote `A` to d-dimensions manually before calling this
    function.

    If ``A.ndim > d``, `reps` is promoted to `A`.ndim by pre-pending 1's to it.
    Thus for an `A` of shape (2, 3, 4, 5), a `reps` of (2, 2) is treated as
    (1, 1, 2, 2).

    Parameters
    ----------
    A : array_like
        The input array.
    reps : 1d array_like
        The number of repetitions of `A` along each axis.

    Returns
    -------
    c : ndarray
        The tiled output array.

    See Also
    --------
    numpy.tile

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    computed_reps: tuple[int, ...]
    if isinstance(reps, int):
        computed_reps = (reps,)
    else:
        if np.ndim(reps) > 1:
            raise TypeError("`reps` must be a 1d sequence")
        computed_reps = tuple(reps)
    # Figure out the shape of the destination array
    out_dims = _builtin_max(A.ndim, len(computed_reps))
    # Prepend ones until the dimensions match
    while len(computed_reps) < out_dims:
        computed_reps = (1,) + computed_reps
    out_shape: NdShape = ()
    # Prepend dimensions if necessary
    for dim in range(out_dims - A.ndim):
        out_shape += (computed_reps[dim],)
    offset = len(out_shape)
    for dim in range(A.ndim):
        out_shape += (A.shape[dim] * computed_reps[offset + dim],)
    assert len(out_shape) == out_dims
    result = ndarray(out_shape, dtype=A.dtype, inputs=(A,))
    result._thunk.tile(A._thunk, computed_reps)
    return result


def repeat(a: ndarray, repeats: Any, axis: Optional[int] = None) -> ndarray:
    """
    Repeat elements of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    repeats : int or ndarray[int]
        The number of repetitions for each element. repeats is
        broadcasted to fit the shape of the given axis.
    axis : int, optional
        The axis along which to repeat values. By default, use the
        flattened input array, and return a flat output array.

    Returns
    -------
    repeated_array : ndarray
        Output array which has the same shape as a, except along the
        given axis.

    Notes
    -----
    Currently, repeat operations supports only 1D arrays

    See Also
    --------
    numpy.repeat

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    if repeats is None:
        raise TypeError(
            "int() argument must be a string, a bytes-like object or a number,"
            " not 'NoneType'"
        )

    if np.ndim(repeats) > 1:
        raise ValueError("`repeats` should be scalar or 1D array")

    # axes should be integer type
    if axis is not None and not isinstance(axis, int):
        raise TypeError("Axis should be of integer type")

    # when array is a scalar
    if np.ndim(a) == 0:
        if axis is not None and axis != 0 and axis != -1:
            raise np.AxisError(
                f"axis {axis} is out of bounds for array of dimension 0"
            )
        if np.ndim(repeats) == 0:
            if not isinstance(repeats, int):
                runtime.warn(
                    "converting repeats to an integer type",
                    category=UserWarning,
                )
            repeats = np.int64(repeats)
            return full((repeats,), cast(Union[int, float], a))
        elif np.ndim(repeats) == 1 and len(repeats) == 1:
            if not isinstance(repeats, int):
                runtime.warn(
                    "converting repeats to an integer type",
                    category=UserWarning,
                )
            repeats = np.int64(repeats)
            return full((repeats[0],), cast(Union[int, float], a))
        else:
            raise ValueError(
                "`repeat` with a scalar parameter `a` is only "
                "implemented for scalar values of the parameter `repeats`."
            )

    # array is an array
    array = convert_to_cunumeric_ndarray(a)
    if np.ndim(repeats) == 1:
        repeats = convert_to_cunumeric_ndarray(repeats)

    # if no axes specified, flatten array
    if axis is None:
        array = array.ravel()
        axis = 0

    axis_int: int = normalize_axis_index(axis, array.ndim)

    # If repeats is on a zero sized axis_int, then return the array.
    if array.shape[axis_int] == 0:
        return array.copy()

    if np.ndim(repeats) == 1:
        if repeats.shape[0] == 1 and repeats.shape[0] != array.shape[axis_int]:
            repeats = repeats[0]

    # repeats is a scalar.
    if np.ndim(repeats) == 0:
        # repeats is 0
        if repeats == 0:
            empty_shape = list(array.shape)
            empty_shape[axis_int] = 0
            return ndarray(shape=tuple(empty_shape), dtype=array.dtype)
        # repeats should be integer type
        if not isinstance(repeats, int):
            runtime.warn(
                "converting repeats to an integer type",
                category=UserWarning,
            )
        result = array._thunk.repeat(
            repeats=np.int64(repeats),
            axis=axis_int,
            scalar_repeats=True,
        )
    # repeats is an array
    else:
        # repeats should be integer type
        repeats = repeats._warn_and_convert(np.int64)
        if repeats.shape[0] != array.shape[axis_int]:
            raise ValueError("incorrect shape of repeats array")
        result = array._thunk.repeat(
            repeats=repeats._thunk, axis=axis_int, scalar_repeats=False
        )
    return ndarray(shape=result.shape, thunk=result)


# Rearranging elements


@add_boilerplate("m")
def flip(m: ndarray, axis: Optional[NdShapeLike] = None) -> ndarray:
    """
    Reverse the order of elements in an array along the given axis.

    The shape of the array is preserved, but the elements are reordered.

    Parameters
    ----------
    m : array_like
        Input array.
    axis : None or int or tuple[int], optional
         Axis or axes along which to flip over. The default, axis=None, will
         flip over all of the axes of the input array.  If axis is negative it
         counts from the last to the first axis.

         If axis is a tuple of ints, flipping is performed on all of the axes
         specified in the tuple.

    Returns
    -------
    out : array_like
        A view of `m` with the entries of axis reversed.  Since a view is
        returned, this operation is done in constant time.

    See Also
    --------
    numpy.flip

    Availability
    --------
    Single GPU, Single CPU
    """
    return m.flip(axis=axis)


###################
# Binary operations
###################

# Elementwise bit operations


###################
# Indexing routines
###################

# Generating index arrays


@add_boilerplate("arr", "mask", "vals")
def place(arr: ndarray, mask: ndarray, vals: ndarray) -> None:
    """
    Change elements of an array based on conditional and input values.

    Parameters
    ----------
    arr : array_like
        Array to put data into.
    mask : array_like
        Mask array. Must have the same size as `arr`.
    vals : 1-D sequence
        Values to put into `arr`. Only the first N elements are used,
        where N is the number of True values in mask. If vals is smaller
        than N, it will be repeated, and if elements of a are to be masked,
        this sequence must be non-empty.

    See Also
    --------
    numpy.copyto, numpy.put, numpy.take, numpy.extract

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if arr.size == 0:
        return

    if mask.size != arr.size:
        raise ValueError("arr array and condition array must be of same size")

    if vals.ndim != 1:
        raise ValueError("vals array has to be 1-dimensional")

    if mask.shape != arr.shape:
        mask_reshape = reshape(mask, arr.shape)
    else:
        mask_reshape = mask

    num_values = int(count_nonzero(mask_reshape))
    if num_values == 0:
        return

    if vals.size == 0:
        raise ValueError("vals array cannot be empty")

    if num_values != vals.size:
        reps = (num_values + vals.size - 1) // vals.size
        vals_resized = tile(A=vals, reps=reps) if reps > 1 else vals
        vals_resized = vals_resized[:num_values]
    else:
        vals_resized = vals

    if mask_reshape.dtype == bool:
        arr._thunk.set_item(mask_reshape._thunk, vals_resized._thunk)
    else:
        bool_mask = mask_reshape.astype(bool)
        arr._thunk.set_item(bool_mask._thunk, vals_resized._thunk)


@add_boilerplate("condition", "arr")
def extract(condition: ndarray, arr: ndarray) -> ndarray:
    """

    Return the elements of an array that satisfy some condition.

    Parameters
    ----------
    condition : array_like
        An array whose nonzero or True entries indicate the elements
        of `arr` to extract.
    arr : array_like
        Input array of the same size as `condition`.

    Returns
    -------
    result : ndarray
        Rank 1 array of values from arr where `condition` is True.

    See Also
    --------
    numpy.extract

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if condition.size != arr.size:
        raise ValueError("arr array and condition array must be of same size")

    if condition.shape != arr.shape:
        condition_reshape = reshape(condition, arr.shape)
    else:
        condition_reshape = condition

    if condition_reshape.dtype == bool:
        thunk = arr._thunk.get_item(condition_reshape._thunk)
    else:
        bool_condition = condition_reshape.astype(bool)
        thunk = arr._thunk.get_item(bool_condition._thunk)

    return ndarray(shape=thunk.shape, thunk=thunk)


@add_boilerplate("a")
def nonzero(a: ndarray) -> tuple[ndarray, ...]:
    """

    Return the indices of the elements that are non-zero.

    Returns a tuple of arrays, one for each dimension of `a`,
    containing the indices of the non-zero elements in that
    dimension.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    tuple_of_arrays : tuple
        Indices of elements that are non-zero.

    See Also
    --------
    numpy.nonzero

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.nonzero()


@add_boilerplate("a")
def flatnonzero(a: ndarray) -> ndarray:
    """

    Return indices that are non-zero in the flattened version of a.

    This is equivalent to `np.nonzero(np.ravel(a))[0]`.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    res : ndarray
        Output array, containing the indices of the elements of
        `a.ravel()` that are non-zero.

    See Also
    --------
    numpy.flatnonzero

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return nonzero(ravel(a))[0]


@add_boilerplate("a", "x", "y")
def where(
    a: ndarray, x: Optional[ndarray] = None, y: Optional[ndarray] = None
) -> Union[ndarray, tuple[ndarray, ...]]:
    """
    where(condition, [x, y])

    Return elements chosen from `x` or `y` depending on `condition`.

    Parameters
    ----------
    condition : array_like, bool
        Where True, yield `x`, otherwise yield `y`.
    x, y : array_like
        Values from which to choose. `x`, `y` and `condition` need to be
        broadcastable to some shape.

    Returns
    -------
    out : ndarray
        An array with elements from `x` where `condition` is True, and elements
        from `y` elsewhere.

    See Also
    --------
    numpy.where

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if x is None or y is None:
        if x is not None or y is not None:
            raise ValueError(
                "both 'x' and 'y' parameters must be specified together for"
                " 'where'"
            )
        return nonzero(a)
    return ndarray._perform_where(a, x, y)


@add_boilerplate("a")
def argwhere(a: ndarray) -> ndarray:
    """
    argwhere(a)

    Find the indices of array elements that are non-zero, grouped by element.

    Parameters
    ----------
    a : array_like
        Input data.

    Returns
    -------
    index_array : ndarray
        Indices of elements that are non-zero. Indices are grouped by element.
        This array will have shape (N, a.ndim) where N is the number of
        non-zero items.

    See Also
    --------
    numpy.argwhere

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    thunk = a._thunk.argwhere()
    return ndarray(shape=thunk.shape, thunk=thunk)


# Indexing-like operations
def indices(
    dimensions: Sequence[int], dtype: npt.DTypeLike = int, sparse: bool = False
) -> Union[ndarray, tuple[ndarray, ...]]:
    """
    Return an array representing the indices of a grid.
    Compute an array where the subarrays contain index values 0, 1, ...
    varying only along the corresponding axis.

    Parameters
    ----------
    dimensions : Sequence[int]
        The shape of the grid.
    dtype : data-type, optional
        Data type of the result.
    sparse : bool, optional
        Return a sparse representation of the grid instead of a dense
        representation. Default is False.

    Returns
    -------
    grid : ndarray or Tuple[ndarray, ...]
        If sparse is False returns one array of grid indices,
        ``grid.shape = (len(dimensions),) + tuple(dimensions)``.
        If sparse is True returns a tuple of arrays, with
        ``grid[i].shape = (1, ..., 1, dimensions[i], 1, ..., 1)`` with
        dimensions[i] in the ith place

    See Also
    --------
    numpy.indices

    Notes
    -----
    The output shape in the dense case is obtained by prepending the number
    of dimensions in front of the tuple of dimensions, i.e. if `dimensions`
    is a tuple ``(r0, ..., rN-1)`` of length ``N``, the output shape is
    ``(N, r0, ..., rN-1)``.
    The subarrays ``grid[k]`` contains the N-D array of indices along the
    ``k-th`` axis. Explicitly:

        grid[k, i0, i1, ..., iN-1] = ik

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    # implementation of indices routine is adapted from NumPy
    dimensions = tuple(dimensions)
    N = len(dimensions)
    shape = (1,) * N
    if sparse:
        res_tuple: tuple[ndarray, ...] = ()
        for i, dim in enumerate(dimensions):
            idx = arange(dim, dtype=dtype).reshape(
                shape[:i] + (dim,) + shape[i + 1 :]
            )
            res_tuple += (idx,)
        return res_tuple
    else:
        out_shape = (N,) + dimensions
        res_array: ndarray = empty(out_shape, dtype=dtype)
        for i, dim in enumerate(dimensions):
            idx = arange(dim, dtype=dtype).reshape(
                shape[:i] + (dim,) + shape[i + 1 :]
            )
            res_array[i] = idx
        return res_array


def mask_indices(
    n: int, mask_func: Callable[[ndarray, int], ndarray], k: int = 0
) -> tuple[ndarray, ...]:
    """
    Return the indices to access (n, n) arrays, given a masking function.

    Assume `mask_func` is a function that, for a square array a of size
    ``(n, n)`` with a possible offset argument `k`, when called as
    ``mask_func(a, k)`` returns a new array with zeros in certain locations
    (functions like :func:`cunumeric.triu` or :func:`cunumeric.tril`
    do precisely this). Then this function returns the indices where
    the non-zero values would be located.

    Parameters
    ----------
    n : int
        The returned indices will be valid to access arrays of shape (n, n).
    mask_func : callable
        A function whose call signature is similar to that of
        :func:`cunumeric.triu`, :func:`cunumeric.tril`.
        That is, ``mask_func(x, k)`` returns a boolean array, shaped like `x`.
        `k` is an optional argument to the function.
    k : scalar
        An optional argument which is passed through to `mask_func`. Functions
        like :func:`cunumeric.triu`, :func:`cunumeric,tril`
        take a second argument that is interpreted as an offset.

    Returns
    -------
    indices : tuple of arrays.
        The `n` arrays of indices corresponding to the locations where
        ``mask_func(np.ones((n, n)), k)`` is True.

    See Also
    --------
    numpy.mask_indices

    Notes
    -----
    WARNING: `mask_indices` expects `mask_function` to call cuNumeric functions
    for good performance. In case non-cuNumeric functions are called by
    `mask_function`, cuNumeric will have to materialize all data on the host
    which might result in running out of system memory.

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    # this implementation is based on the Cupy
    a = ones((n, n), dtype=bool)
    if not is_implemented(mask_func):
        runtime.warn(
            "Calling non-cuNumeric functions in mask_func can result in bad "
            "performance",
            category=UserWarning,
        )
    return mask_func(a, k).nonzero()


def diag_indices(n: int, ndim: int = 2) -> tuple[ndarray, ...]:
    """
    Return the indices to access the main diagonal of an array.

    This returns a tuple of indices that can be used to access the main
    diagonal of an array a with a.ndim >= 2 dimensions and
    shape (n, n, …, n). For a.ndim = 2 this is the usual diagonal,
    for a.ndim > 2 this is the set of indices to
    access a[i, i, ..., i] for i = [0..n-1].

    Parameters
    ----------
    n : int
        The size, along each dimension, of the arrays for which the
        returned indices can be used.
    ndim : int, optional
        The number of dimensions.

    See Also
    --------
    numpy.diag_indices

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    idx = arange(n, dtype=int)
    return (idx,) * ndim


@add_boilerplate("arr")
def diag_indices_from(arr: ndarray) -> tuple[ndarray, ...]:
    """
    Return the indices to access the main diagonal of an n-dimensional array.

    See diag_indices for full details.

    Parameters
    ----------
    arr : array_like
        at least 2-D

    See Also
    --------
    numpy.diag_indices_from, numpy.diag_indices

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if not arr.ndim >= 2:
        raise ValueError("input array must be at least 2-d")
    # For more than d=2, the strided formula is only valid for arrays with
    # all dimensions equal, so we check first.
    for i in range(1, arr.ndim):
        if arr.shape[i] != arr.shape[0]:
            raise ValueError("All dimensions of input must be of equal length")

    return diag_indices(arr.shape[0], arr.ndim)


def tril_indices(
    n: int, k: int = 0, m: Optional[int] = None
) -> tuple[ndarray, ...]:
    """
    Return the indices for the lower-triangle of an (n, m) array.

    Parameters
    ----------
    n : int
        The row dimension of the arrays for which the returned
        indices will be valid.
    k : int, optional
        Diagonal offset (see :func:`cunumeric.tril` for details).
    m : int, optional
        The column dimension of the arrays for which the returned
        indices will be valid.
        By default `m` is taken equal to `n`.

    Returns
    -------
    inds : tuple of arrays
        The indices for the lower-triangle. The returned tuple contains two
        arrays, each with the indices along one dimension of the array.

    See also
    --------
    numpy.tril_indices

    Notes
    -----

    Availability
    ------------
    Multiple GPUs, Multiple CPUs
    """

    tri_ = tri(n, m, k=k, dtype=bool)
    return nonzero(tri_)


@add_boilerplate("arr")
def tril_indices_from(arr: ndarray, k: int = 0) -> tuple[ndarray, ...]:
    """
    Return the indices for the lower-triangle of arr.

    See :func:`cunumeric.tril_indices` for full details.

    Parameters
    ----------
    arr : array_like
        The indices will be valid for arrays whose dimensions are
        the same as arr.
    k : int, optional
        Diagonal offset (see :func:`cunumeric.tril` for details).

    Returns
    -------
    inds : tuple of arrays
        The indices for the lower-triangle. The returned tuple contains two
        arrays, each with the indices along one dimension of the array.

    See Also
    --------
    numpy.tril_indices_from

    Notes
    -----

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    # this implementation is taken from numpy
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    return tril_indices(arr.shape[-2], k=k, m=arr.shape[-1])


def triu_indices(
    n: int, k: int = 0, m: Optional[int] = None
) -> tuple[ndarray, ...]:
    """
    Return the indices for the upper-triangle of an (n, m) array.

    Parameters
    ----------
    n : int
        The size of the arrays for which the returned indices will
        be valid.
    k : int, optional
        Diagonal offset (see :func:`cunumeric.triu` for details).
    m : int, optional
        The column dimension of the arrays for which the returned
        arrays will be valid.
        By default `m` is taken equal to `n`.

    Returns
    -------
    inds : tuple of arrays
        The indices for the upper-triangle. The returned tuple contains two
        arrays, each with the indices along one dimension of the array.

    See also
    --------
    numpy.triu_indices

    Notes
    -----

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    tri_ = ~tri(n, m, k=k - 1, dtype=bool)
    return nonzero(tri_)


@add_boilerplate("arr")
def triu_indices_from(arr: ndarray, k: int = 0) -> tuple[ndarray, ...]:
    """
    Return the indices for the upper-triangle of arr.

    See :func:`cunumeric.triu_indices` for full details.

    Parameters
    ----------
    arr : ndarray, shape(N, N)
        The indices will be valid for arrays whose dimensions are
        the same as arr.
    k : int, optional
        Diagonal offset (see :func:`cunumeric.triu` for details).

    Returns
    -------
    inds : tuple of arrays
        The indices for the upper-triangle. The returned tuple contains two
        arrays, each with the indices along one dimension of the array.

    See Also
    --------
    numpy.triu_indices_from

    Notes
    -----

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    # this implementation is taken from numpy
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    return triu_indices(arr.shape[-2], k=k, m=arr.shape[-1])


@add_boilerplate("a")
def take(
    a: ndarray,
    indices: ndarray,
    axis: Optional[int] = None,
    out: Optional[ndarray] = None,
    mode: BoundsMode = "raise",
) -> ndarray:
    """
    Take elements from an array along an axis.
    When axis is not None, this function does the same thing as “fancy”
    indexing (indexing arrays using arrays); however, it can be easier
    to use if you need elements along a given axis. A call such as
    `np.take(arr, indices, axis=3)` is equivalent to `arr[:,:,:,indices,...]`.

    Parameters
    ----------
    a : array_like `(Ni…, M, Nk…)`
        The source array.
    indices : array_like `(Nj…)`
        The indices of the values to extract.
        Also allow scalars for indices.
    axis : int, optional
        The axis over which to select values. By default, the flattened input
        array is used.
    out : ndarray, optional `(Ni…, Nj…, Nk…)`
        If provided, the result will be placed in this array. It should be of
        the appropriate shape and dtype.
    mode : ``{'raise', 'wrap', 'clip'}``, optional
        Specifies how out-of-bounds indices will behave.
        'raise' - raise an error (default)
        'wrap' - wrap around
        'clip' - clip to the range
        'clip' mode means that all indices that are too large are replaced by
        the index that addresses the last element along that axis.
        Note that this disables indexing with negative numbers.

    Returns
    -------
    out : ndarray `(Ni…, Nj…, Nk…)`
        The returned array has the same type as a.

    Raises
    ------

    See Also
    --------
    numpy.take

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.take(indices=indices, axis=axis, out=out, mode=mode)


def _fill_fancy_index_for_along_axis_routines(
    a_shape: NdShape, axis: int, indices: ndarray
) -> tuple[ndarray, ...]:
    # the logic below is base on the cupy implementation of
    # the *_along_axis routines
    ndim = len(a_shape)
    fancy_index = []
    for i, n in enumerate(a_shape):
        if i == axis:
            fancy_index.append(indices)
        else:
            ind_shape = (1,) * i + (-1,) + (1,) * (ndim - i - 1)
            fancy_index.append(arange(n).reshape(ind_shape))
    return tuple(fancy_index)


@add_boilerplate("a", "indices")
def take_along_axis(
    a: ndarray, indices: ndarray, axis: Union[int, None]
) -> ndarray:
    """
    Take values from the input array by matching 1d index and data slices.

    This iterates over matching 1d slices oriented along the specified axis in
    the index and data arrays, and uses the former to look up values in the
    latter. These slices can be different lengths.

    Functions returning an index along an axis, like
    :func:`cunumeric.argsort` and :func:`cunumeric.argpartition`,
    produce suitable indices for this function.

    Parameters
    ----------
    arr : ndarray (Ni..., M, Nk...)
        Source array
    indices : ndarray (Ni..., J, Nk...)
        Indices to take along each 1d slice of `arr`. This must match the
        dimension of arr, but dimensions Ni and Nj only need to broadcast
        against `arr`.
    axis : int
        The axis to take 1d slices along. If axis is None, the input array is
        treated as if it had first been flattened to 1d, for consistency with
        :func:`cunumeric.sort` and :func:`cunumeric.argsort`.

    Returns
    -------
    out: ndarray (Ni..., J, Nk...)
        The indexed result. It is going to be a view to `arr` for most cases,
        except the case when `axis=Null` and `arr.ndim>1`.

    See Also
    --------
    numpy.take_along_axis

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("`indices` must be an integer array")

    computed_axis = 0
    if axis is None:
        if indices.ndim != 1:
            raise ValueError("indices must be 1D if axis=None")
        if a.ndim > 1:
            a = a.ravel()
    else:
        computed_axis = normalize_axis_index(axis, a.ndim)

    if a.ndim != indices.ndim:
        raise ValueError(
            "`indices` and `a` must have the same number of dimensions"
        )
    return a[
        _fill_fancy_index_for_along_axis_routines(
            a.shape, computed_axis, indices
        )
    ]


@add_boilerplate("a", "indices", "values")
def put_along_axis(
    a: ndarray, indices: ndarray, values: ndarray, axis: Union[int, None]
) -> None:
    """
    Put values into the destination array by matching 1d index and data slices.

    This iterates over matching 1d slices oriented along the specified axis in
    the index and data arrays, and uses the former to place values into the
    latter. These slices can be different lengths.

    Functions returning an index along an axis, like :func:`cunumeric.argsort`
    and :func:`cunumeric.argpartition`, produce suitable indices for
    this function.

    Parameters
    ----------
    a : ndarray (Ni..., M, Nk...)
        Destination array.
    indices : ndarray (Ni..., J, Nk...)
        Indices to change along each 1d slice of `arr`. This must match the
        dimension of arr, but dimensions in Ni and Nj may be 1 to broadcast
        against `arr`.
    values : array_like (Ni..., J, Nk...)
        values to insert at those indices. Its shape and dimension are
        broadcast to match that of `indices`.
    axis : int
        The axis to take 1d slices along. If axis is None, the destination
        array is treated as if a flattened 1d view had been created of it.
        `axis=None` case is currently supported only for 1D input arrays.

    Note
    ----
    Having duplicate entries in `indices` will result in undefined behavior
    since operation performs asynchronous update of the `arr` entries.

    See Also
    --------
    numpy.put_along_axis

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """

    if a.size == 0:
        return

    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("`indices` must be an integer array")

    computed_axis = 0
    if axis is None:
        if indices.ndim != 1:
            raise ValueError("indices must be 1D if axis=None")
        if a.ndim > 1:
            # TODO call a=a.flat when flat is implemented
            raise ValueError("a.ndim>1 case is not supported when axis=None")
        if (indices.size == 0) or (values.size == 0):
            return
        if values.shape != indices.shape:
            values = values._wrap(indices.size)
    else:
        computed_axis = normalize_axis_index(axis, a.ndim)

    if a.ndim != indices.ndim:
        raise ValueError(
            "`indices` and `a` must have the same number of dimensions"
        )
    ind = _fill_fancy_index_for_along_axis_routines(
        a.shape, computed_axis, indices
    )
    a[ind] = values


@add_boilerplate("a")
def choose(
    a: ndarray,
    choices: Sequence[ndarray],
    out: Optional[ndarray] = None,
    mode: BoundsMode = "raise",
) -> ndarray:
    """
    Construct an array from an index array and a list of arrays to choose from.

    Given an "index" array (`a`) of integers and a sequence of ``n`` arrays
    (`choices`), `a` and each choice array are first broadcast, as necessary,
    to arrays of a common shape; calling these *Ba* and *Bchoices[i], i =
    0,...,n-1* we have that, necessarily, ``Ba.shape == Bchoices[i].shape``
    for each ``i``.  Then, a new array with shape ``Ba.shape`` is created as
    follows:

    * if ``mode='raise'`` (the default), then, first of all, each element of
      ``a`` (and thus ``Ba``) must be in the range ``[0, n-1]``; now, suppose
      that ``i`` (in that range) is the value at the ``(j0, j1, ..., jm)``
      position in ``Ba`` - then the value at the same position in the new array
      is the value in ``Bchoices[i]`` at that same position;

    * if ``mode='wrap'``, values in `a` (and thus `Ba`) may be any (signed)
      integer; modular arithmetic is used to map integers outside the range
      `[0, n-1]` back into that range; and then the new array is constructed
      as above;

    * if ``mode='clip'``, values in `a` (and thus ``Ba``) may be any (signed)
      integer; negative integers are mapped to 0; values greater than ``n-1``
      are mapped to ``n-1``; and then the new array is constructed as above.

    Parameters
    ----------
    a : ndarray[int]
        This array must contain integers in ``[0, n-1]``, where ``n`` is the
        number of choices, unless ``mode=wrap`` or ``mode=clip``, in which
        cases any integers are permissible.
    choices : Sequence[ndarray]
        Choice arrays. `a` and all of the choices must be broadcastable to the
        same shape.  If `choices` is itself an array (not recommended), then
        its outermost dimension (i.e., the one corresponding to
        ``choices.shape[0]``) is taken as defining the "sequence".
    out : ndarray, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype. Note that `out` is always
        buffered if ``mode='raise'``; use other modes for better performance.
    mode : ``{'raise', 'wrap', 'clip'}``, optional
        Specifies how indices outside ``[0, n-1]`` will be treated:

          * 'raise' : an exception is raised (default)
          * 'wrap' : value becomes value mod ``n``
          * 'clip' : values < 0 are mapped to 0, values > n-1 are mapped to n-1

    Returns
    -------
    merged_array : ndarray
        The merged result.

    Raises
    ------
    ValueError: shape mismatch
        If `a` and each choice array are not all broadcastable to the same
        shape.

    See Also
    --------
    numpy.choose

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.choose(choices=choices, out=out, mode=mode)


@add_boilerplate("condition", "a")
def compress(
    condition: ndarray,
    a: ndarray,
    axis: Optional[int] = None,
    out: Optional[ndarray] = None,
) -> ndarray:
    """
    Return selected slices of an array along given axis.

    When working along a given axis, a slice along that axis is returned
    in output for each index where condition evaluates to True.
    When working on a 1-D array, compress is equivalent to numpy.extract.

    Parameters
    ----------
    condition, 1-D array of bools
        Array that selects which entries to return. If `len(c)` is less than
        the size of a along the given axis, then output is truncated to the
        length of the condition array.

    a : array_like
        Array from which to extract a part.

    axis: int, optional
        Axis along which to take slices. If None (default),
        work on the flattened array.

    out : ndarray, optional
        Output array. Its type is preserved and it must be of the right
        shape to hold the output.

    Returns
    -------
    compressed_array : ndarray
        A copy of `a` without the slices along `axis` for which condition
        is false.

    Raises
    ------
    ValueError : dimension mismatch
        If condition is not 1D array
    ValueError : shape mismatch
        If condition contains entries that are out of bounds of array
    ValueError : shape mismatch
        If output array has a wrong shape

    See Also
    --------
    numpy.compress, numpy.extract

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    return a.compress(condition, axis=axis, out=out)


@add_boilerplate("a")
def diagonal(
    a: ndarray,
    offset: int = 0,
    axis1: Optional[int] = None,
    axis2: Optional[int] = None,
    extract: bool = True,
    axes: Optional[tuple[int, int]] = None,
) -> ndarray:
    """
    diagonal(a: ndarray, offset=0, axis1=None, axis2=None)

    Return specified diagonals.

    If `a` is 2-D, returns the diagonal of `a` with the given offset,
    i.e., the collection of elements of the form ``a[i, i+offset]``.  If
    `a` has more than two dimensions, then the axes specified by `axis1`
    and `axis2` are used to determine the 2-D sub-array whose diagonal is
    returned.  The shape of the resulting array can be determined by
    removing `axis1` and `axis2` and appending an index to the right equal
    to the size of the resulting diagonals.

    Parameters
    ----------
    a : array_like
        Array from which the diagonals are taken.
    offset : int, optional
        Offset of the diagonal from the main diagonal.  Can be positive or
        negative.  Defaults to main diagonal (0).
    axis1 : int, optional
        Axis to be used as the first axis of the 2-D sub-arrays from which
        the diagonals should be taken.  Defaults to first axis (0).
    axis2 : int, optional
        Axis to be used as the second axis of the 2-D sub-arrays from
        which the diagonals should be taken. Defaults to second axis (1).

    Returns
    -------
    array_of_diagonals : ndarray
        If `a` is 2-D, then a 1-D array containing the diagonal and of the
        same type as `a` is returned unless `a` is a `matrix`, in which case
        a 1-D array rather than a (2-D) `matrix` is returned in order to
        maintain backward compatibility.

        If ``a.ndim > 2``, then the dimensions specified by `axis1` and `axis2`
        are removed, and a new axis inserted at the end corresponding to the
        diagonal.

    Raises
    ------
    ValueError
        If the dimension of `a` is less than 2.

    Notes
    -----
    Unlike NumPy's, the cuNumeric implementation always returns a copy

    See Also
    --------
    numpy.diagonal

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    return a.diagonal(
        offset=offset, axis1=axis1, axis2=axis2, extract=extract, axes=axes
    )


@add_boilerplate("a", "indices", "values")
def put(
    a: ndarray, indices: ndarray, values: ndarray, mode: str = "raise"
) -> None:
    """
    Replaces specified elements of an array with given values.
    The indexing works as if the target array is first flattened.

    Parameters
    ----------
    a : array_like
        Array to put data into
    indices : array_like
        Target indices, interpreted as integers.
        WARNING: In case there are repeated entries in the
        indices array, Legate doesn't guarantee the order in
        which values are updated.

    values : array_like
        Values to place in `a` at target indices. If values array is shorter
        than indices, it will be repeated as necessary.
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices will behave.
        'raise' : raise an error.
        'wrap' : wrap around.
        'clip' : clip to the range.

    See Also
    --------
    numpy.put

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    a.put(indices=indices, values=values, mode=mode)


@add_boilerplate("a", "mask", "values")
def putmask(a: ndarray, mask: ndarray, values: ndarray) -> None:
    """
    putmask(a, mask, values)
    Changes elements of an array based on conditional and input values.
    Sets ``a.flat[n] = values[n]`` for each n where ``mask.flat[n]==True``.
    If `values` is not the same size as `a` and `mask` then it will repeat.
    This gives behavior different from ``a[mask] = values``.

    Parameters
    ----------
    a : ndarray
        Target array.
    mask : array_like
        Boolean mask array. It has to be the same shape as `a`.
    values : array_like
        Values to put into `a` where `mask` is True. If `values` is smaller
        than `a` it will be repeated.

    See Also
    --------
    numpy.putmask

    Availability
    ------------
    Multiple GPUs, Multiple CPUs
    """
    if not a.shape == mask.shape:
        raise ValueError("mask and data must be the same size")

    mask = mask._warn_and_convert(np.dtype(bool))

    if a.dtype != values.dtype:
        values = values._warn_and_convert(a.dtype)

    try:
        np.broadcast_shapes(values.shape, a.shape)
    except ValueError:
        values = values._wrap(a.size)
        values = values.reshape(a.shape)

    a._thunk.putmask(mask._thunk, values._thunk)


@add_boilerplate("a", "val")
def fill_diagonal(a: ndarray, val: ndarray, wrap: bool = False) -> None:
    """
    Fill the main diagonal of the given array of any dimensionality.

    For an array a with a.ndim >= 2, the diagonal is the list of locations with
    indices a[i, ..., i] all identical. This function modifies the input
    array in-place, it does not return a value.

    Parameters
    ----------

    a : array, at least 2-D.
        Array whose diagonal is to be filled, it gets modified in-place.
    val : scalar or array_like
        Value(s) to write on the diagonal. If val is scalar, the value is
        written along the diagonal.
        If array-like, the flattened val is written along
        the diagonal, repeating if necessary to fill all diagonal entries.
    wrap : bool
        If true, the diagonal "wraps" after N columns, for tall 2d matrices.

    Raises
    ------
    ValueError
        If the dimension of `a` is less than 2.

    Notes
    -----

    See Also
    --------
    numpy.fill_diagonal

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """
    if val.size == 0 or a.size == 0:
        return

    if a.ndim < 2:
        raise ValueError("array must be at least 2-d")

    n = _builtin_min(a.shape)

    if a.ndim > 2:
        for s in a.shape:
            if s != n:
                raise ValueError(
                    "All dimensions of input must be of equal length"
                )

    len_val = n

    if a.ndim == 2 and wrap and a.shape[0] > a.shape[1]:
        len_val = a.shape[0] - (a.shape[0] // (a.shape[1] + 1))

    if (val.size != len_val and val.ndim > 0) or val.ndim > 1:
        val = val._wrap(len_val)

    if a.ndim == 2 and wrap and a.shape[0] > a.shape[1]:
        idx0_tmp = arange(a.shape[1], dtype=int)
        idx0 = idx0_tmp.copy()
        while idx0.size < len_val:
            idx0_tmp = idx0_tmp + (a.shape[1] + 1)
            idx0 = hstack((idx0, idx0_tmp))
        idx0 = idx0[0:len_val]
        idx1 = arange(len_val, dtype=int) % a.shape[1]
        a[idx0, idx1] = val
    else:
        idx = arange(n, dtype=int)
        indices = (idx,) * a.ndim

        a[indices] = val


################
# Linear algebra
################

# Matrix and vector products


@add_boilerplate("a", "b")
def inner(a: ndarray, b: ndarray, out: Optional[ndarray] = None) -> ndarray:
    """
    Inner product of two arrays.

    Ordinary inner product of vectors for 1-D arrays (without complex
    conjugation), in higher dimensions a sum product over the last axes.

    Parameters
    ----------
    a, b : array_like
    out : ndarray, optional
        Output argument. This must have the exact shape that would be returned
        if it was not present. If its dtype is not what would be expected from
        this operation, then the result will be (unsafely) cast to `out`.

    Returns
    -------
    output : ndarray
        If `a` and `b` are both
        scalars or both 1-D arrays then a scalar is returned; otherwise
        an array is returned.
        ``output.shape = (*a.shape[:-1], *b.shape[:-1])``
        If `out` is given, then it is returned.

    Notes
    -----
    The cuNumeric implementation is a little more liberal than NumPy in terms
    of allowed broadcasting, e.g. ``inner(ones((1,)), ones((4,)))`` is allowed.

    See Also
    --------
    numpy.inner

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if a.ndim == 0 or b.ndim == 0:
        return multiply(a, b, out=out)
    (a_modes, b_modes, out_modes) = inner_modes(a.ndim, b.ndim)
    return _contract(
        a_modes,
        b_modes,
        out_modes,
        a,
        b,
        out=out,
        casting="unsafe",
    )


@add_boilerplate("a", "b")
def dot(a: ndarray, b: ndarray, out: Optional[ndarray] = None) -> ndarray:
    """
    Dot product of two arrays. Specifically,

    - If both `a` and `b` are 1-D arrays, it is inner product of vectors
      (without complex conjugation).

    - If both `a` and `b` are 2-D arrays, it is matrix multiplication,
      but using ``a @ b`` is preferred.

    - If either `a` or `b` is 0-D (scalar), it is equivalent to
      :func:`multiply` and using ``cunumeric.multiply(a, b)`` or ``a * b`` is
      preferred.

    - If `a` is an N-D array and `b` is a 1-D array, it is a sum product over
      the last axis of `a` and `b`.

    - If `a` is an N-D array and `b` is an M-D array (where ``M>=2``), it is a
      sum product over the last axis of `a` and the second-to-last axis of
      `b`::

        dot(a: ndarray, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

    Parameters
    ----------
    a : array_like
        First argument.
    b : array_like
        Second argument.
    out : ndarray, optional
        Output argument. This must have the exact shape and dtype that would be
        returned if it was not present.

    Returns
    -------
    output : ndarray
        Returns the dot product of `a` and `b`. If `out` is given, then it is
        returned.

    Notes
    -----
    The cuNumeric implementation is a little more liberal than NumPy in terms
    of allowed broadcasting, e.g. ``dot(ones((3,1)), ones((4,5)))`` is allowed.

    Except for the inner-product case, only floating-point types are supported.

    See Also
    --------
    numpy.dot

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.dot(b, out=out)


@add_boilerplate("a", "b")
def matmul(
    a: ndarray,
    b: ndarray,
    /,
    out: Optional[ndarray] = None,
    *,
    casting: CastingKind = "same_kind",
    dtype: Optional[np.dtype[Any]] = None,
) -> ndarray:
    """
    Matrix product of two arrays.

    Parameters
    ----------
    a, b : array_like
        Input arrays, scalars not allowed.
    out : ndarray, optional
        A location into which the result is stored. If provided, it must have
        a shape that matches the signature `(n,k),(k,m)->(n,m)`.
    casting : ``{'no', 'equiv', 'safe', 'same_kind', 'unsafe'}``, optional
        Controls what kind of data casting may occur.

          * 'no' means the data types should not be cast at all.
          * 'equiv' means only byte-order changes are allowed.
          * 'safe' means only casts which can preserve values are allowed.
          * 'same_kind' means only safe casts or casts within a kind,
            like float64 to float32, are allowed.
          * 'unsafe' means any data conversions may be done.

        Default is 'same_kind'.
    dtype : data-type, optional
        If provided, forces the calculation to use the data type specified.
        Note that you may have to also give a more liberal `casting`
        parameter to allow the conversions. Default is None.

    Returns
    -------
    output : ndarray
        The matrix product of the inputs.
        This is a scalar only when both a, b are 1-d vectors.
        If `out` is given, then it is returned.

    Notes
    -----
    The behavior depends on the arguments in the following way.

    - If both arguments are 2-D they are multiplied like conventional
      matrices.
    - If either argument is N-D, N > 2, it is treated as a stack of
      matrices residing in the last two indexes and broadcast accordingly.
    - If the first argument is 1-D, it is promoted to a matrix by
      prepending a 1 to its dimensions. After matrix multiplication
      the prepended 1 is removed.
    - If the second argument is 1-D, it is promoted to a matrix by
      appending a 1 to its dimensions. After matrix multiplication
      the appended 1 is removed.

    ``matmul`` differs from ``dot`` in two important ways:

    - Multiplication by scalars is not allowed, use ``*`` instead.
    - Stacks of matrices are broadcast together as if the matrices
      were elements, respecting the signature ``(n,k),(k,m)->(n,m)``:

      >>> a = ones([9, 5, 7, 4])
      >>> c = ones([9, 5, 4, 3])
      >>> dot(a: ndarray, c).shape
      (9, 5, 7, 9, 5, 3)
      >>> matmul(a: ndarray, c).shape
      (9, 5, 7, 3)
      >>> # n is 7, k is 4, m is 3

    The cuNumeric implementation is a little more liberal than NumPy in terms
    of allowed broadcasting, e.g. ``matmul(ones((3,1)), ones((4,5)))`` is
    allowed.

    Only floating-point types are supported.

    See Also
    --------
    numpy.matmul

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if a.ndim == 0 or b.ndim == 0:
        raise ValueError("Scalars not allowed in matmul")
    (a_modes, b_modes, out_modes) = matmul_modes(a.ndim, b.ndim)
    return _contract(
        a_modes,
        b_modes,
        out_modes,
        a,
        b,
        out=out,
        casting=casting,
        dtype=dtype,
    )


@add_boilerplate("a", "b")
def vdot(a: ndarray, b: ndarray, out: Optional[ndarray] = None) -> ndarray:
    """
    Return the dot product of two vectors.

    The vdot(`a`, `b`) function handles complex numbers differently than
    dot(`a`, `b`).  If the first argument is complex the complex conjugate
    of the first argument is used for the calculation of the dot product.

    Note that `vdot` handles multidimensional arrays differently than `dot`:
    it does *not* perform a matrix product, but flattens input arguments
    to 1-D vectors first. Consequently, it should only be used for vectors.

    Parameters
    ----------
    a : array_like
        If `a` is complex the complex conjugate is taken before calculation
        of the dot product.
    b : array_like
        Second argument to the dot product.
    out : ndarray, optional
        Output argument. This must have the exact shape that would be returned
        if it was not present. If its dtype is not what would be expected from
        this operation, then the result will be (unsafely) cast to `out`.

    Returns
    -------
    output : ndarray
        Dot product of `a` and `b`. If `out` is given, then it is returned.

    Notes
    -----
    The cuNumeric implementation is a little more liberal than NumPy in terms
    of allowed broadcasting, e.g. ``vdot(ones((1,)), ones((4,)))`` is allowed.

    See Also
    --------
    numpy.vdot

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return inner(a.ravel().conj(), b.ravel(), out=out)


@add_boilerplate("a", "b")
def outer(a: ndarray, b: ndarray, out: Optional[ndarray] = None) -> ndarray:
    """
    Compute the outer product of two vectors.

    Given two vectors, ``a = [a0, a1, ..., aM]`` and ``b = [b0, b1, ..., bN]``,
    the outer product is::

      [[a0*b0  a0*b1 ... a0*bN ]
       [a1*b0    .
       [ ...          .
       [aM*b0            aM*bN ]]

    Parameters
    ----------
    a : (M,) array_like
        First input vector. Input is flattened if not already 1-dimensional.
    b : (N,) array_like
        Second input vector. Input is flattened if not already 1-dimensional.
    out : (M, N) ndarray, optional
        A location where the result is stored. If its dtype is not what would
        be expected from this operation, then the result will be (unsafely)
        cast to `out`.

    Returns
    -------
    output : (M, N) ndarray
        ``output[i, j] = a[i] * b[j]``
        If `out` is given, then it is returned.

    See Also
    --------
    numpy.outer

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return multiply(
        a.ravel()[:, np.newaxis], b.ravel()[np.newaxis, :], out=out
    )


@add_boilerplate("a", "b")
def tensordot(
    a: ndarray,
    b: ndarray,
    axes: AxesPairLike = 2,
    out: Optional[ndarray] = None,
) -> ndarray:
    """
    Compute tensor dot product along specified axes.

    Given two tensors, `a` and `b`, and an array_like object containing
    two array_like objects, ``(a_axes, b_axes)``, sum the products of
    `a`'s and `b`'s elements (components) over the axes specified by
    ``a_axes`` and ``b_axes``. The third argument can be a single non-negative
    integer_like scalar, ``N``; if it is such, then the last ``N`` dimensions
    of `a` and the first ``N`` dimensions of `b` are summed over.

    Parameters
    ----------
    a, b : array_like
        Tensors to "dot".

    axes : int or array_like
        * integer_like
          If an int N, sum over the last N axes of `a` and the first N axes
          of `b` in order.
        * (2,) array_like
          Or, a list of axes to be summed over, first sequence applying to `a`,
          second to `b`. Both elements array_like must be of the same length.
    out : ndarray, optional
        Output argument. This must have the exact shape that would be returned
        if it was not present. If its dtype is not what would be expected from
        this operation, then the result will be (unsafely) cast to `out`.

    Returns
    -------
    output : ndarray
        The tensor dot product of the inputs. If `out` is given, then it is
        returned.

    Notes
    -----
    The cuNumeric implementation is a little more liberal than NumPy in terms
    of allowed broadcasting, e.g. ``tensordot(ones((3,1)), ones((1,4)))`` is
    allowed.

    Except for the inner-product case, only floating-point types are supported.

    See Also
    --------
    numpy.tensordot

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    (a_modes, b_modes, out_modes) = tensordot_modes(a.ndim, b.ndim, axes)
    return _contract(
        a_modes,
        b_modes,
        out_modes,
        a,
        b,
        out=out,
        casting="unsafe",
    )


# Trivial multi-tensor contraction strategy: contract in input order
class NullOptimizer(oe.paths.PathOptimizer):  # type: ignore [misc,no-any-unimported] # noqa
    def __call__(
        self,
        inputs: list[set[str]],
        outputs: set[str],
        size_dict: dict[str, int],
        memory_limit: Union[int, None] = None,
    ) -> list[tuple[int, int]]:
        return [(0, 1)] + [(0, -1)] * (len(inputs) - 2)


def _maybe_cast_input(
    arr: ndarray, to_dtype: np.dtype[Any], casting: CastingKind
) -> ndarray:
    if arr.dtype == to_dtype:
        return arr
    if not np.can_cast(arr.dtype, to_dtype, casting=casting):
        raise TypeError(
            f"Cannot cast input array of type {arr.dtype} to {to_dtype} with "
            f"casting rule '{casting}'"
        )
    return arr.astype(to_dtype)


# Generalized tensor contraction
def _contract(
    a_modes: list[str],
    b_modes: list[str],
    out_modes: list[str],
    a: ndarray,
    b: Optional[ndarray] = None,
    out: Optional[ndarray] = None,
    casting: CastingKind = "same_kind",
    dtype: Optional[np.dtype[Any]] = None,
) -> ndarray:
    # Sanity checks
    if len(a_modes) != a.ndim:
        raise ValueError(
            f"Expected {len(a_modes)}-d input array but got {a.ndim}-d"
        )
    if b is None:
        if len(b_modes) != 0:
            raise ValueError("Missing input array")
    elif len(b_modes) != b.ndim:
        raise ValueError(
            f"Expected {len(b_modes)}-d input array but got {b.ndim}-d"
        )
    if out is not None and len(out_modes) != out.ndim:
        raise ValueError(
            f"Expected {len(out_modes)}-d output array but got {out.ndim}-d"
        )
    if len(set(out_modes)) != len(out_modes):
        raise ValueError("Duplicate mode labels on output")
    if len(set(out_modes) - set(a_modes) - set(b_modes)) > 0:
        raise ValueError("Unknown mode labels on output")

    # Handle types
    makes_view = b is None and len(a_modes) == len(out_modes)
    if dtype is not None and not makes_view:
        c_dtype = dtype
    elif out is not None:
        c_dtype = out.dtype
    elif b is None:
        c_dtype = a.dtype
    else:
        c_dtype = ndarray.find_common_type(a, b)
    a = _maybe_cast_input(a, c_dtype, casting)
    if b is not None:
        b = _maybe_cast_input(b, c_dtype, casting)
    out_dtype = out.dtype if out is not None else c_dtype

    # Handle duplicate modes on inputs
    c_a_modes = Counter(a_modes)
    for mode, count in c_a_modes.items():
        if count > 1:
            axes = [i for (i, m) in enumerate(a_modes) if m == mode]
            a = a._diag_helper(axes=axes)
            # diagonal is stored on last axis
            a_modes = [m for m in a_modes if m != mode] + [mode]
    c_b_modes = Counter(b_modes)
    for mode, count in c_b_modes.items():
        if count > 1:
            axes = [i for (i, m) in enumerate(b_modes) if m == mode]
            b = b._diag_helper(axes=axes)  # type: ignore [union-attr]
            # diagonal is stored on last axis
            b_modes = [m for m in b_modes if m != mode] + [mode]

    # Drop modes corresponding to singleton dimensions. This handles cases of
    # broadcasting.
    for dim in reversed(range(a.ndim)):
        if a.shape[dim] == 1:
            a = a.squeeze(dim)
            a_modes.pop(dim)
    if b is not None:
        for dim in reversed(range(b.ndim)):
            if b.shape[dim] == 1:
                b = b.squeeze(dim)
                b_modes.pop(dim)

    # Sum-out modes appearing on one argument, and missing from the result
    # TODO: If we supported sum on multiple axes we could do the full sum in a
    # single operation, and avoid intermediates.
    for dim, mode in reversed(list(enumerate(a_modes))):
        if mode not in b_modes and mode not in out_modes:
            a_modes.pop(dim)
            a = a.sum(axis=dim)

    for dim, mode in reversed(list(enumerate(b_modes))):
        if mode not in a_modes and mode not in out_modes:
            b_modes.pop(dim)
            b = b.sum(axis=dim)  # type: ignore [union-attr]

    # Compute extent per mode. No need to consider broadcasting at this stage,
    # since it has been handled above.
    mode2extent: dict[str, int] = {}
    for mode, extent in chain(
        zip(a_modes, a.shape), zip(b_modes, b.shape) if b is not None else []
    ):
        prev_extent = mode2extent.get(mode)
        if prev_extent is not None and extent != prev_extent:
            raise ValueError(
                f"Incompatible sizes between matched dimensions: {extent} vs "
                f"{prev_extent}"
            )
        mode2extent[mode] = extent

    # Any modes appearing only on the result must have originally been present
    # on one of the operands, but got dropped by the broadcast-handling code.
    out_shape = (
        out.shape
        if out is not None
        else tuple(mode2extent.get(mode, 1) for mode in out_modes)
    )
    c_modes = []
    c_shape: NdShape = ()
    c_bloated_shape: NdShape = ()
    for mode, extent in zip(out_modes, out_shape):
        if mode not in a_modes and mode not in b_modes:
            c_bloated_shape += (1,)
        else:
            assert extent > 1
            c_modes.append(mode)
            c_shape += (extent,)
            c_bloated_shape += (extent,)

    # Verify output array has the right shape (input arrays can be broadcasted
    # up to match the output, but not the other way around). There should be no
    # unknown or singleton modes on the result at this point.
    for mode, extent in zip(c_modes, c_shape):
        prev_extent = mode2extent[mode]
        assert prev_extent != 1
        if extent != prev_extent:
            raise ValueError("Wrong shape on output array")

    # Test for fallback to unary case
    if b is not None:
        if len(a_modes) == 0:
            a = a * b
            a_modes = b_modes
            b = None
            b_modes = []
        elif len(b_modes) == 0:
            a = a * b
            b = None

    if b is None:
        # Unary contraction case
        assert len(a_modes) == len(c_modes) and set(a_modes) == set(c_modes)
        if len(a_modes) == 0:
            # NumPy doesn't return a view in this case
            c = copy(a)
        elif a_modes == c_modes:
            c = a
        else:
            # Shuffle input array according to mode labels
            axes = [a_modes.index(mode) for mode in c_modes]
            assert _builtin_all(ax >= 0 for ax in axes)
            c = a.transpose(axes)

    else:
        # Binary contraction case
        # Create result array, if output array can't be directly targeted
        if out is not None and out_dtype == c_dtype and out_shape == c_shape:
            c = out
        else:
            c = ndarray(
                shape=c_shape,
                dtype=c_dtype,
                inputs=(a, b),
            )
        # Perform operation
        c._thunk.contract(
            c_modes,
            a._thunk,
            a_modes,
            b._thunk,
            b_modes,
            mode2extent,
        )

    # Postprocess result before returning
    if out is c:
        # We already decided above to use the output array directly
        return out
    if out_dtype != c_dtype or out_shape != c_bloated_shape:
        # We need to broadcast the result of the contraction or switch types
        # before returning
        if not np.can_cast(c_dtype, out_dtype, casting=casting):
            raise TypeError(
                f"Cannot cast intermediate result array of type {c_dtype} "
                f"into output array of type {out_dtype} with casting rule "
                f"'{casting}'"
            )
        if out is None:
            out = ndarray(
                shape=out_shape,
                dtype=out_dtype,
                inputs=(c,),
            )
        out[...] = c.reshape(c_bloated_shape)
        return out
    if out_shape != c_shape:
        # We need to add missing dimensions, but they are all of size 1, so
        # we don't need to broadcast
        assert c_bloated_shape == out_shape
        if out is None:
            return c.reshape(out_shape)
        else:
            out[...] = c.reshape(out_shape)
            return out
    if out is not None:
        # The output and result arrays are fully compatible, but we still
        # need to copy
        out[...] = c
        return out
    return c


def einsum(
    expr: str,
    *operands: ndarray,
    out: Optional[ndarray] = None,
    dtype: Optional[np.dtype[Any]] = None,
    casting: CastingKind = "safe",
    optimize: Union[bool, str] = False,
) -> ndarray:
    """
    Evaluates the Einstein summation convention on the operands.

    Using the Einstein summation convention, many common multi-dimensional,
    linear algebraic array operations can be represented in a simple fashion.
    In *implicit* mode `einsum` computes these values.

    In *explicit* mode, `einsum` provides further flexibility to compute
    other array operations that might not be considered classical Einstein
    summation operations, by disabling, or forcing summation over specified
    subscript labels.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation as comma separated list of
        subscript labels. An implicit (classical Einstein summation)
        calculation is performed unless the explicit indicator '->' is
        included as well as subscript labels of the precise output form.
    operands : list[array_like]
        These are the arrays for the operation.
    out : ndarray, optional
        If provided, the calculation is done into this array.
    dtype : data-type, optional
        If provided, forces the calculation to use the data type specified.
        Note that you may have to also give a more liberal `casting`
        parameter to allow the conversions. Default is None.
    casting : ``{'no', 'equiv', 'safe', 'same_kind', 'unsafe'}``, optional
        Controls what kind of data casting may occur.

          * 'no' means the data types should not be cast at all.
          * 'equiv' means only byte-order changes are allowed.
          * 'safe' means only casts which can preserve values are allowed.
          * 'same_kind' means only safe casts or casts within a kind,
            like float64 to float32, are allowed.
          * 'unsafe' means any data conversions may be done.

        Default is 'safe'.
    optimize : ``{False, True, 'greedy', 'optimal'}``, optional
        Controls if intermediate optimization should occur. No optimization
        will occur if False. Uses opt_einsum to find an optimized contraction
        plan if True.

    Returns
    -------
    output : ndarray
        The calculation based on the Einstein summation convention.

    Notes
    -----
    For most expressions, only floating-point types are supported.

    See Also
    --------
    numpy.einsum

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    operands_list = [convert_to_cunumeric_ndarray(op) for op in operands]
    if not optimize:
        optimize = NullOptimizer()
    # This call normalizes the expression (adds the output part if it's
    # missing, expands '...') and checks for some errors (mismatch on number
    # of dimensions between operand and expression, wrong number of operands,
    # unknown modes on output, a mode appearing under two different
    # non-singleton extents).
    computed_operands, contractions = oe.contract_path(
        expr, *operands_list, einsum_call=True, optimize=optimize
    )
    for indices, _, sub_expr, _, _ in contractions:
        assert len(indices) == 1 or len(indices) == 2
        a = computed_operands.pop(indices[0])
        b = computed_operands.pop(indices[1]) if len(indices) == 2 else None
        if b is None:
            m = re.match(r"([a-zA-Z]*)->([a-zA-Z]*)", sub_expr)
            if m is None:
                raise NotImplementedError("Non-alphabetic mode labels")
            a_modes = list(m.group(1))
            b_modes = []
            out_modes = list(m.group(2))
        else:
            m = re.match(r"([a-zA-Z]*),([a-zA-Z]*)->([a-zA-Z]*)", sub_expr)
            if m is None:
                raise NotImplementedError("Non-alphabetic mode labels")
            a_modes = list(m.group(1))
            b_modes = list(m.group(2))
            out_modes = list(m.group(3))
        sub_result = _contract(
            a_modes,
            b_modes,
            out_modes,
            a,
            b,
            out=(out if len(computed_operands) == 0 else None),
            casting=casting,
            dtype=dtype,
        )
        computed_operands.append(sub_result)
    assert len(computed_operands) == 1
    return computed_operands[0]


def einsum_path(
    expr: str,
    *operands: ndarray,
    optimize: Union[bool, list[Any], tuple[Any, ...], str] = "greedy",
) -> tuple[list[Union[str, int]], str]:
    """
    Evaluates the lowest cost contraction order for an einsum expression by
    considering the creation of intermediate arrays.

    Parameters
    ----------
    expr : str
        Specifies the subscripts for summation.
    *operands : Sequence[array_like]
        These are the arrays for the operation.
    optimize : ``{bool, list, tuple, 'greedy', 'optimal'}``
        Choose the type of path. If a tuple is provided, the second argument is
        assumed to be the maximum intermediate size created. If only a single
        argument is provided the largest input or output array size is used
        as a maximum intermediate size.

        * if a list is given that starts with ``einsum_path``, uses this as the
          contraction path
        * if False no optimization is taken
        * if True defaults to the 'greedy' algorithm
        * 'optimal' An algorithm that combinatorially explores all possible
          ways of contracting the listed tensors and chooses the least costly
          path. Scales exponentially with the number of terms in the
          contraction.
        * 'greedy' An algorithm that chooses the best pair contraction
          at each step. Effectively, this algorithm searches the largest inner,
          Hadamard, and then outer products at each step. Scales cubically with
          the number of terms in the contraction. Equivalent to the 'optimal'
          path for most contractions.

        Default is 'greedy'.

    Returns
    -------
    path : list[Tuple[int,...]]
        A list representation of the einsum path.
    string_repr : str
        A printable representation of the einsum path.

    Notes
    -----
    The resulting path indicates which terms of the input contraction should be
    contracted first, the result of this contraction is then appended to the
    end of the contraction list. This list can then be iterated over until all
    intermediate contractions are complete.

    See Also
    --------
    numpy.einsum_path

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    computed_operands = [convert_to_cunumeric_ndarray(op) for op in operands]
    memory_limit = _builtin_max(op.size for op in computed_operands)
    if type(optimize) == tuple:
        if len(optimize) != 2:
            raise ValueError("einsum_path expects optimize tuples of size 2")
        optimize, memory_limit = optimize
    if optimize is True:
        optimize = "greedy"
    elif optimize is False:
        optimize = [tuple(range(len(computed_operands)))]
    elif optimize in ["greedy", "optimal"]:
        pass
    elif (
        type(optimize) == list
        and len(optimize) > 1
        and optimize[0] == "einsum_path"
    ):
        optimize = optimize[1:]
    else:
        raise ValueError(
            f"einsum_path: unexpected value for optimize: {optimize}"
        )
    path, info = oe.contract_path(
        expr, *computed_operands, optimize=optimize, memory_limit=memory_limit
    )
    return ["einsum_path"] + path, info


@add_boilerplate("a")
def trace(
    a: ndarray,
    offset: int = 0,
    axis1: Optional[int] = None,
    axis2: Optional[int] = None,
    dtype: Optional[np.dtype[Any]] = None,
    out: Optional[ndarray] = None,
) -> ndarray:
    """
    Return the sum along diagonals of the array.

    If a is 2-D, the sum along its diagonal with the given offset is
    returned, i.e., the sum of elements a[i,i+offset] for all i.
    If a has more than two dimensions, then the axes specified by axis1
    and axis2 are used to determine the 2-D sub-arrays whose traces
    are returned. The shape of the resulting array is the same as that
    of a with axis1 and axis2 removed.

    Parameters
    ----------
    a : array_like
        Input array, from which the diagonals are taken.
    offset : int, optional
        Offset of the diagonal from the main diagonal. Can be both
        positive and negative. Defaults to 0.
    axis1, axis2 : int, optional
        Axes to be used as the first and second axis of the 2-D sub-arrays
        from which the diagonals should be taken. Defaults are the
        first two axes of a.
    dtype : data-type, optional
        Determines the data-type of the returned array and of the
        accumulator where the elements are summed. If dtype has the value
        None and a is of integer type of precision less than the default
        integer precision, then the default integer precision is used.
        Otherwise, the precision is the same as that of a.

    out : ndarray, optional
        Array into which the output is placed. Its type is preserved and
        it must be of the right shape to hold the output.

    Returns
    -------
    sum_along_diagonals : ndarray
        If a is 2-D, the sum along the diagonal is returned. If a has
        larger dimensions, then an array of sums along diagonals is returned.

    Raises
    ------
    ValueError
        If the dimension of `a` is less than 2.

    See Also
    --------
    numpy.diagonal

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.trace(
        offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=out
    )


#################
# Logic functions
#################

# Truth value testing


@add_boilerplate("a")
def all(
    a: ndarray,
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    out: Optional[ndarray] = None,
    keepdims: bool = False,
    where: bool = True,
) -> ndarray:
    """
    Test whether all array elements along a given axis evaluate to True.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : None or int or tuple[int], optional
        Axis or axes along which a logical AND reduction is performed.
        The default (``axis=None``) is to perform a logical AND over all
        the dimensions of the input array. `axis` may be negative, in
        which case it counts from the last to the first axis.

        If this is a tuple of ints, a reduction is performed on multiple
        axes, instead of a single axis or all the axes as before.
    out : ndarray, optional
        Alternate output array in which to place the result.
        It must have the same shape as the expected output and its
        type is preserved (e.g., if ``dtype(out)`` is float, the result
        will consist of 0.0's and 1.0's). See `ufuncs-output-type` for more
        details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `all` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    all : ndarray, bool
        A new boolean or array is returned unless `out` is specified,
        in which case a reference to `out` is returned.

    See Also
    --------
    numpy.all

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.all(axis=axis, out=out, keepdims=keepdims, where=where)


@add_boilerplate("a")
def any(
    a: ndarray,
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    out: Optional[ndarray] = None,
    keepdims: bool = False,
    where: bool = True,
) -> ndarray:
    """
    Test whether any array element along a given axis evaluates to True.

    Returns single boolean unless `axis` is not ``None``

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : None or int or tuple[int], optional
        Axis or axes along which a logical OR reduction is performed.
        The default (``axis=None``) is to perform a logical OR over all
        the dimensions of the input array. `axis` may be negative, in
        which case it counts from the last to the first axis.

        If this is a tuple of ints, a reduction is performed on multiple
        axes, instead of a single axis or all the axes as before.
    out : ndarray, optional
        Alternate output array in which to place the result.  It must have
        the same shape as the expected output and its type is preserved
        (e.g., if it is of type float, then it will remain so, returning
        1.0 for True and 0.0 for False, regardless of the type of `a`).
        See `ufuncs-output-type` for more details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `any` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    any : bool or ndarray
        A new boolean or `ndarray` is returned unless `out` is specified,
        in which case a reference to `out` is returned.

    See Also
    --------
    numpy.any

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.any(axis=axis, out=out, keepdims=keepdims, where=where)


# Array contents


# Logic operations


# Comparison


@add_boilerplate("a", "b")
def allclose(
    a: ndarray,
    b: ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> ndarray:
    """

    Returns True if two arrays are element-wise equal within a tolerance.

    The tolerance values are positive, typically very small numbers.  The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.

    NaNs are treated as equal if they are in the same place and if
    ``equal_nan=True``.  Infs are treated as equal if they are in the same
    place and of the same sign in both arrays.

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    rtol : float
        The relative tolerance parameter (see Notes).
    atol : float
        The absolute tolerance parameter (see Notes).
    equal_nan : bool
        Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        considered equal to NaN's in `b` in the output array.

    Returns
    -------
    allclose : ndarray scalar
        Returns True if the two arrays are equal within the given
        tolerance; False otherwise.

    Notes
    -----
    If the following equation is element-wise True, then allclose returns
    True.

     absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

    See Also
    --------
    numpy.allclose

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if equal_nan:
        raise NotImplementedError(
            "cuNumeric does not support `equal_nan` yet for allclose"
        )
    args = (np.array(rtol, dtype=np.float64), np.array(atol, dtype=np.float64))
    return ndarray._perform_binary_reduction(
        BinaryOpCode.ISCLOSE,
        a,
        b,
        dtype=np.dtype(bool),
        extra_args=args,
    )


@add_boilerplate("a", "b")
def isclose(
    a: ndarray,
    b: ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> ndarray:
    """

    Returns a boolean array where two arrays are element-wise equal within a
    tolerance.

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    rtol : float
        The relative tolerance parameter (see Notes).
    atol : float
        The absolute tolerance parameter (see Notes).
    equal_nan : bool
        Whether to compare NaN's as equal.  If True, NaN's in `a` will be
        considered equal to NaN's in `b` in the output array.

    Returns
    -------
    y : array_like
        Returns a boolean array of where `a` and `b` are equal within the
        given tolerance. If both `a` and `b` are scalars, returns a single
        boolean value.

    Notes
    -----
    For finite values, isclose uses the following equation to test whether
    two floating point values are equivalent.

     absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

    See Also
    --------
    numpy.isclose

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if equal_nan:
        raise NotImplementedError(
            "cuNumeric does not support `equal_nan` yet for isclose"
        )

    out_shape = np.broadcast_shapes(a.shape, b.shape)
    out = empty(out_shape, dtype=bool)

    common_type = ndarray.find_common_type(a, b)
    a = a.astype(common_type)
    b = b.astype(common_type)

    out._thunk.isclose(a._thunk, b._thunk, rtol, atol, equal_nan)
    return out


@add_boilerplate("a1", "a2")
def array_equal(
    a1: ndarray, a2: ndarray, equal_nan: bool = False
) -> Union[bool, ndarray]:
    """

    True if two arrays have the same shape and elements, False otherwise.

    Parameters
    ----------
    a1, a2 : array_like
        Input arrays.
    equal_nan : bool
        Whether to compare NaN's as equal. If the dtype of a1 and a2 is
        complex, values will be considered equal if either the real or the
        imaginary component of a given value is ``nan``.

    Returns
    -------
    b : ndarray scalar
        Returns True if the arrays are equal.

    See Also
    --------
    numpy.array_equal

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if equal_nan:
        raise NotImplementedError(
            "cuNumeric does not support `equal_nan` yet for `array_equal`"
        )

    if a1.shape != a2.shape:
        return False
    return ndarray._perform_binary_reduction(
        BinaryOpCode.EQUAL, a1, a2, dtype=np.dtype(np.bool_)
    )


########################
# Mathematical functions
########################

# Trigonometric functions


# Hyperbolic functions


# Rounding


# Sums, products, differences


@add_boilerplate("a")
def prod(
    a: ndarray,
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    dtype: Optional[np.dtype[Any]] = None,
    out: Optional[ndarray] = None,
    keepdims: bool = False,
    initial: Optional[Union[int, float]] = None,
    where: bool = True,
) -> ndarray:
    """

    Return the product of array elements over a given axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple[int], optional
        Axis or axes along which a product is performed.  The default,
        axis=None, will calculate the product of all the elements in the
        input array. If axis is negative it counts from the last to the
        first axis.

        If axis is a tuple of ints, a product is performed on all of the
        axes specified in the tuple instead of a single axis or all the
        axes as before.
    dtype : data-type, optional
        The type of the returned array, as well as of the accumulator in
        which the elements are multiplied.  The dtype of `a` is used by
        default unless `a` has an integer dtype of less precision than the
        default platform integer.  In that case, if `a` is signed then the
        platform integer is used while if `a` is unsigned then an unsigned
        integer of the same precision as the platform integer is used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output, but the type of the output
        values will be cast if necessary.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `prod` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    initial : scalar, optional
        The starting value for this product. See `~cunumeric.ufunc.reduce` for
        details.

    where : array_like[bool], optional
        Elements to include in the product. See `~cunumeric.ufunc.reduce` for
        details.

    Returns
    -------
    product_along_axis : ndarray, see `dtype` parameter above.
        An array shaped as `a` but with the specified axis removed.
        Returns a reference to `out` if specified.

    See Also
    --------
    numpy.prod

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return multiply.reduce(
        a,
        axis=axis,
        dtype=dtype,
        out=out,
        keepdims=keepdims,
        initial=initial,
        where=where,
    )


@add_boilerplate("a")
def sum(
    a: ndarray,
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    dtype: Optional[np.dtype[Any]] = None,
    out: Optional[ndarray] = None,
    keepdims: bool = False,
    initial: Optional[Union[int, float]] = None,
    where: bool = True,
) -> ndarray:
    """

    Sum of array elements over a given axis.

    Parameters
    ----------
    a : array_like
        Elements to sum.
    axis : None or int or tuple[int], optional
        Axis or axes along which a sum is performed.  The default,
        axis=None, will sum all of the elements of the input array.  If
        axis is negative it counts from the last to the first axis.

        If axis is a tuple of ints, a sum is performed on all of the axes
        specified in the tuple instead of a single axis or all the axes as
        before.
    dtype : data-type, optional
        The type of the returned array and of the accumulator in which the
        elements are summed.  The dtype of `a` is used by default unless `a`
        has an integer dtype of less precision than the default platform
        integer.  In that case, if `a` is signed then the platform integer
        is used while if `a` is unsigned then an unsigned integer of the
        same precision as the platform integer is used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output, but the type of the output
        values will be cast if necessary.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `sum` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    initial : scalar, optional
        Starting value for the sum. See `~cunumeric.ufunc.reduce` for details.

    where : array_like[bool], optional
        Elements to include in the sum. See `~cunumeric.ufunc.reduce` for
        details.

    Returns
    -------
    sum_along_axis : ndarray
        An array with the same shape as `a`, with the specified
        axis removed.   If `a` is a 0-d array, or if `axis` is None, a scalar
        is returned.  If an output array is specified, a reference to
        `out` is returned.

    See Also
    --------
    numpy.sum

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return add.reduce(
        a,
        axis=axis,
        dtype=dtype,
        out=out,
        keepdims=keepdims,
        initial=initial,
        where=where,
    )


@add_boilerplate("a")
def cumprod(
    a: ndarray,
    axis: Optional[int] = None,
    dtype: Optional[np.dtype[Any]] = None,
    out: Optional[ndarray] = None,
) -> ndarray:
    """
    Return the cumulative product of the elements along a given axis.

    Parameters
    ----------
    a : array_like
        Input array.

    axis : int, optional
        Axis along which the cumulative product is computed. The default (None)
        is to compute the cumprod over the flattened array.

    dtype : dtype, optional
        Type of the returned array and of the accumulator in which the elements
        are multiplied. If dtype is not specified, it defaults to the dtype of
        a, unless a has an integer dtype with a precision less than that of the
        default platform integer. In that case, the default platform integer is
        used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have the
        same shape and buffer length as the expected output but the type will
        be cast if necessary. See Output type determination for more details.

    Returns
    -------
    cumprod : ndarray
        A new array holding the result is returned unless out is specified, in
        which case a reference to out is returned. The result has the same size
        as a, and the same shape as a if axis is not None or a is a 1-d array.

    See Also
    --------
    numpy.cumprod

    Notes
    -----
    CuNumeric's parallel implementation may yield different results from NumPy
    with floating point and complex types. For example, when boundary values
    such as inf occur they may not propagate as expected. Consider the float32
    array ``[3e+37, 1, 100, 0.01]``. NumPy's cumprod will return a result of
    ``[3e+37, 3e+37, inf, inf]``. However, cuNumeric might internally partition
    the array such that partition 0 has ``[3e+37, 1]``  and partition 1 has
    ``[100, 0.01]``, returning the result ``[3e+37, 3e+37, inf, 3e+37]``.

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return ndarray._perform_scan(
        ScanCode.PROD,
        a,
        axis=axis,
        dtype=dtype,
        out=out,
        nan_to_identity=False,
    )


@add_boilerplate("a")
def cumsum(
    a: ndarray,
    axis: Optional[int] = None,
    dtype: Optional[np.dtype[Any]] = None,
    out: Optional[ndarray] = None,
) -> ndarray:
    """
    Return the cumulative sum of the elements along a given axis.

    Parameters
    ----------
    a : array_like
        Input array.

    axis : int, optional
        Axis along which the cumulative sum is computed. The default (None) is
        to compute the cumsum over the flattened array.

    dtype : dtype, optional
        Type of the returned array and of the accumulator in which the elements
        are summed. If dtype is not specified, it defaults to the dtype of a,
        unless a has an integer dtype with a precision less than that of the
        default platform integer. In that case, the default platform integer is
        used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have the
        same shape and buffer length as the expected output but the type will
        be cast if necessary. See Output type determination for more details.

    Returns
    -------
    cumsum : ndarray.
        A new array holding the result is returned unless out is specified, in
        which case a reference to out is returned. The result has the same size
        as a, and the same shape as a if axis is not None or a is a 1-d array.

    See Also
    --------
    numpy.cumsum

    Notes
    -----
    CuNumeric's parallel implementation may yield different results from NumPy
    with floating point and complex types. For example, when boundary values
    such as inf occur they may not propagate as expected. For more explanation
    check cunumeric.cumprod.

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return ndarray._perform_scan(
        ScanCode.SUM, a, axis=axis, dtype=dtype, out=out, nan_to_identity=False
    )


@add_boilerplate("a")
def nancumprod(
    a: ndarray,
    axis: Optional[int] = None,
    dtype: Optional[np.dtype[Any]] = None,
    out: Optional[ndarray] = None,
) -> ndarray:
    """
    Return the cumulative product of the elements along a given axis treating
    Not a Numbers (NaNs) as one. The cumulative product does not change when
    NaNs are encountered and leading NaNs are replaced by ones.

    Ones are returned for slices that are all-NaN or empty.

    Parameters
    ----------
    a : array_like
        Input array.

    axis : int, optional
        Axis along which the cumulative product is computed. The default (None)
        is to compute the nancumprod over the flattened array.

    dtype : dtype, optional
        Type of the returned array and of the accumulator in which the elements
        are multiplied. If dtype is not specified, it defaults to the dtype of
        a, unless a has an integer dtype with a precision less than that of the
        default platform integer. In that case, the default platform integer is
        used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have the
        same shape and buffer length as the expected output but the type will
        be cast if necessary. See Output type determination for more details.

    Returns
    -------
    nancumprod : ndarray.
        A new array holding the result is returned unless out is specified, in
        which case a reference to out is returned. The result has the same size
        as a, and the same shape as a if axis is not None or a is a 1-d array.

    See Also
    --------
    numpy.nancumprod

    Notes
    -----
    CuNumeric's parallel implementation may yield different results from NumPy
    with floating point and complex types. For example, when boundary values
    such as inf occur they may not propagate as expected. For more explanation
    check cunumeric.cumprod.

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return ndarray._perform_scan(
        ScanCode.PROD, a, axis=axis, dtype=dtype, out=out, nan_to_identity=True
    )


@add_boilerplate("a")
def nancumsum(
    a: ndarray,
    axis: Optional[int] = None,
    dtype: Optional[np.dtype[Any]] = None,
    out: Optional[ndarray] = None,
) -> ndarray:
    """
    Return the cumulative sum of the elements along a given axis treating Not a
    Numbers (NaNs) as zero. The cumulative sum does not change when NaNs are
    encountered and leading NaNs are replaced by zeros.

    Zeros are returned for slices that are all-NaN or empty.

    Parameters
    ----------
    a : array_like
        Input array.

    axis : int, optional
        Axis along which the cumulative sum is computed. The default (None) is
        to compute the nancumsum over the flattened array.

    dtype : dtype, optional
        Type of the returned array and of the accumulator in which the elements
        are summed. If dtype is not specified, it defaults to the dtype of a,
        unless a has an integer dtype with a precision less than that of the
        default platform integer. In that case, the default platform integer is
        used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have the
        same shape and buffer length as the expected output but the type will
        be cast if necessary. See Output type determination for more details.

    Returns
    -------
    nancumsum : ndarray.
        A new array holding the result is returned unless out is specified, in
        which case a reference to out is returned. The result has the same size
        as a, and the same shape as a if axis is not None or a is a 1-d array.

    See Also
    --------
    numpy.nancumsum

    Notes
    -----
    CuNumeric's parallel implementation may yield different results from NumPy
    with floating point and complex types. For example, when boundary values
    such as inf occur they may not propagate as expected. For more explanation
    check cunumeric.cumprod.

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return ndarray._perform_scan(
        ScanCode.SUM, a, axis=axis, dtype=dtype, out=out, nan_to_identity=True
    )


@add_boilerplate("a")
def nanargmax(
    a: ndarray,
    axis: Any = None,
    out: Union[ndarray, None] = None,
    *,
    keepdims: bool = False,
) -> ndarray:
    """

    Return the indices of the maximum values in the specified axis ignoring
    NaNs. For all-NaN slices ValueError is raised. Warning: the results cannot
    be trusted if a slice contains only NaNs and -Infs.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        By default, the index corresponds to the flattened array, otherwise
        along the specified axis.
    out : ndarray, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the array.

    Returns
    -------
    index_array : ndarray[int]
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.

    See Also
    --------
    numpy.nanargmax

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    # if the datatype is not floating point, then call argmax
    allowed_types: list[np.dtype[Any]] = [
        np.dtype(np.float16),
        np.dtype(np.float32),
        np.dtype(np.float64),
    ]

    if a.dtype in allowed_types:
        unary_reduction_code = UnaryRedCode.NANARGMAX
    else:
        unary_reduction_code = UnaryRedCode.ARGMAX

    index_array = a._perform_unary_reduction(
        unary_reduction_code,
        a,
        axis=axis,
        out=out,
        keepdims=keepdims,
        res_dtype=np.dtype(np.int64),
    )

    # Error handling: we return identity when the array or a slice
    # contains only NaNs, so raise an error
    identity = np.iinfo(np.int64).min
    if index_array.size == 1:
        if index_array == identity:
            raise ValueError("Array contains only NaNs")
    else:
        if identity in index_array:
            raise ValueError("Array contains only NaNs")

    return index_array


# Exponents and logarithms


# Arithmetic operations


# Handling complex numbers


@add_boilerplate("val")
def real(val: ndarray) -> ndarray:
    """
    Return the real part of the complex argument.

    Parameters
    ----------
    val : array_like
        Input array.

    Returns
    -------
    out : ndarray or scalar
        The real component of the complex argument. If `val` is real, the type
        of `val` is used for the output.  If `val` has complex elements, the
        returned type is float.

    See Also
    --------
    numpy.real

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return val.real


@add_boilerplate("val")
def imag(val: ndarray) -> ndarray:
    """

    Return the imaginary part of the complex argument.

    Parameters
    ----------
    val : array_like
        Input array.

    Returns
    -------
    out : ndarray or scalar
        The imaginary component of the complex argument. If `val` is real,
        the type of `val` is used for the output.  If `val` has complex
        elements, the returned type is float.

    See Also
    --------
    numpy.imag

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return val.imag


# Extrema Finding


@add_boilerplate("a")
def amax(
    a: ndarray,
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    dtype: Optional[np.dtype[Any]] = None,
    out: Optional[ndarray] = None,
    keepdims: bool = False,
    initial: Optional[Union[int, float]] = None,
    where: bool = True,
) -> ndarray:
    """

    Return the maximum of an array or maximum along an axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple[int], optional
        Axis or axes along which to operate.  By default, flattened input is
        used.

        If this is a tuple of ints, the maximum is selected over multiple axes,
        instead of a single axis or all the axes as before.
    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
        See `ufuncs-output-type` for more details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `amax` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    initial : scalar, optional
        The minimum value of an output element. Must be present to allow
        computation on empty slice. See `~cunumeric.ufunc.reduce` for details.

    where : array_like[bool], optional
        Elements to compare for the maximum. See `~cunumeric.ufunc.reduce`
        for details.

    Returns
    -------
    amax : ndarray or scalar
        Maximum of `a`. If `axis` is None, the result is a scalar value.
        If `axis` is given, the result is an array of dimension
        ``a.ndim - 1``.

    See Also
    --------
    numpy.amax

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return maximum.reduce(
        a,
        axis=axis,
        dtype=dtype,
        out=out,
        keepdims=keepdims,
        initial=initial,
        where=where,
    )


max = amax


@add_boilerplate("a")
def amin(
    a: ndarray,
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    dtype: Optional[np.dtype[Any]] = None,
    out: Optional[ndarray] = None,
    keepdims: bool = False,
    initial: Optional[Union[int, float]] = None,
    where: bool = True,
) -> ndarray:
    """

    Return the minimum of an array or minimum along an axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple[int], optional
        Axis or axes along which to operate.  By default, flattened input is
        used.

        If this is a tuple of ints, the minimum is selected over multiple axes,
        instead of a single axis or all the axes as before.
    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
        See `ufuncs-output-type` for more details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `amin` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    initial : scalar, optional
        The maximum value of an output element. Must be present to allow
        computation on empty slice. See `~cunumeric.ufunc.reduce` for details.

    where : array_like[bool], optional
        Elements to compare for the minimum. See `~cunumeric.ufunc.reduce`
        for details.

    Returns
    -------
    amin : ndarray or scalar
        Minimum of `a`. If `axis` is None, the result is a scalar value.
        If `axis` is given, the result is an array of dimension
        ``a.ndim - 1``.

    See Also
    --------
    numpy.amin

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return minimum.reduce(
        a,
        axis=axis,
        dtype=dtype,
        out=out,
        keepdims=keepdims,
        initial=initial,
        where=where,
    )


min = amin

# Miscellaneous


@add_boilerplate("a", "v")
def convolve(a: ndarray, v: ndarray, mode: ConvolveMode = "full") -> ndarray:
    """

    Returns the discrete, linear convolution of two ndarrays.

    If `a` and `v` are both 1-D and `v` is longer than `a`, the two are
    swapped before computation. For N-D cases, the arguments are never swapped.

    Parameters
    ----------
    a : (N,) array_like
        First input ndarray.
    v : (M,) array_like
        Second input ndarray.
    mode : ``{'full', 'valid', 'same'}``, optional
        'same':
          The output is the same size as `a`, centered with respect to
          the 'full' output. (default)

        'full':
          The output is the full discrete linear convolution of the inputs.

        'valid':
          The output consists only of those elements that do not
          rely on the zero-padding. In 'valid' mode, either `a` or `v`
          must be at least as large as the other in every dimension.

    Returns
    -------
    out : ndarray
        Discrete, linear convolution of `a` and `v`.

    See Also
    --------
    numpy.convolve

    Notes
    -----
    The current implementation only supports the 'same' mode.

    Unlike `numpy.convolve`, `cunumeric.convolve` supports N-dimensional
    inputs, but it follows NumPy's behavior for 1-D inputs.

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if mode != "same":
        raise NotImplementedError("Need to implement other convolution modes")

    if a.ndim != v.ndim:
        raise RuntimeError("Arrays should have the same dimensions")
    elif a.ndim > 3:
        raise NotImplementedError(f"{a.ndim}-D arrays are not yet supported")

    if a.ndim == 1 and a.size < v.size:
        v, a = a, v

    if a.dtype != v.dtype:
        v = v.astype(a.dtype)
    out = ndarray(
        shape=a.shape,
        dtype=a.dtype,
        inputs=(a, v),
    )
    a._thunk.convolve(v._thunk, out._thunk, mode)
    return out


@add_boilerplate("a")
def clip(
    a: ndarray,
    a_min: Union[int, float, npt.ArrayLike, None],
    a_max: Union[int, float, npt.ArrayLike, None],
    out: Union[npt.NDArray[Any], ndarray, None] = None,
) -> ndarray:
    """

    Clip (limit) the values in an array.

    Given an interval, values outside the interval are clipped to
    the interval edges.  For example, if an interval of ``[0, 1]``
    is specified, values smaller than 0 become 0, and values larger
    than 1 become 1.

    Parameters
    ----------
    a : array_like
        Array containing elements to clip.
    a_min : scalar or array_like or None
        Minimum value. If None, clipping is not performed on lower
        interval edge. Not more than one of `a_min` and `a_max` may be
        None.
    a_max : scalar or array_like or None
        Maximum value. If None, clipping is not performed on upper
        interval edge. Not more than one of `a_min` and `a_max` may be
        None. If `a_min` or `a_max` are array_like, then the three
        arrays will be broadcasted to match their shapes.
    out : ndarray, optional
        The results will be placed in this array. It may be the input
        array for in-place clipping.  `out` must be of the right shape
        to hold the output.  Its type is preserved.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    clipped_array : ndarray
        An array with the elements of `a`, but where values
        < `a_min` are replaced with `a_min`, and those > `a_max`
        with `a_max`.

    See Also
    --------
    numpy.clip

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.clip(a_min, a_max, out=out)


##################################
# Set routines
##################################


@add_boilerplate("ar")
def unique(
    ar: ndarray,
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    axis: Optional[int] = None,
) -> ndarray:
    """

    Find the unique elements of an array.
    Returns the sorted unique elements of an array. There are three optional
    outputs in addition to the unique elements:
    * the indices of the input array that give the unique values
    * the indices of the unique array that reconstruct the input array
    * the number of times each unique value comes up in the input array

    Parameters
    ----------
    ar : array_like
        Input array. Unless `axis` is specified, this will be flattened if it
        is not already 1-D.
    return_index : bool, optional
        If True, also return the indices of `ar` (along the specified axis,
        if provided, or in the flattened array) that result in the unique
        array.
        Currently not supported.
    return_inverse : bool, optional
        If True, also return the indices of the unique array (for the specified
        axis, if provided) that can be used to reconstruct `ar`.
        Currently not supported.
    return_counts : bool, optional
        If True, also return the number of times each unique item appears
        in `ar`.
        Currently not supported.
    axis : int or None, optional
        The axis to operate on. If None, `ar` will be flattened. If an integer,
        the subarrays indexed by the given axis will be flattened and treated
        as the elements of a 1-D array with the dimension of the given axis,
        see the notes for more details.  Object arrays or structured arrays
        that contain objects are not supported if the `axis` kwarg is used. The
        default is None.
        Currently not supported.

    Returns
    -------
    unique : ndarray
        The sorted unique values.
    unique_indices : ndarray, optional
        The indices of the first occurrences of the unique values in the
        original array. Only provided if `return_index` is True.
    unique_inverse : ndarray, optional
        The indices to reconstruct the original array from the
        unique array. Only provided if `return_inverse` is True.
    unique_counts : ndarray, optional
        The number of times each of the unique values comes up in the
        original array. Only provided if `return_counts` is True.

    See Also
    --------
    numpy.unique

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    Notes
    --------
    Keyword arguments for optional outputs are not yet supported.
    `axis` is also not handled currently.

    """
    if _builtin_any((return_index, return_inverse, return_counts, axis)):
        raise NotImplementedError(
            "Keyword arguments for `unique` are not yet supported"
        )

    return ar.unique()


##################################
# Sorting, searching, and counting
##################################

# Sorting


@add_boilerplate("a")
def argsort(
    a: ndarray,
    axis: int = -1,
    kind: SortType = "quicksort",
    order: Optional[Union[str, list[str]]] = None,
) -> ndarray:
    """

    Returns the indices that would sort an array.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or None, optional
        Axis to sort. By default, the index -1 (the last axis) is used. If
        None, the flattened array is used.
    kind : ``{'quicksort', 'mergesort', 'heapsort', 'stable'}``, optional
        Default is 'quicksort'. The underlying sort algorithm might vary.
        The code basically supports 'stable' or *not* 'stable'.
    order : str or list[str], optional
        Currently not supported

    Returns
    -------
    index_array : ndarray[int]
        Array of indices that sort a along the specified axis. It has the
        same shape as `a.shape` or is flattened in case of `axis` is None.

    See Also
    --------
    numpy.argsort

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    result = ndarray(a.shape, np.int64)
    result._thunk.sort(
        rhs=a._thunk, argsort=True, axis=axis, kind=kind, order=order
    )
    return result


def msort(a: ndarray) -> ndarray:
    """

    Returns a sorted copy of an array sorted along the first axis.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    out : ndarray
        Sorted array with same dtype and shape as `a`.

    See Also
    --------
    numpy.msort

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return sort(a, axis=0)


@add_boilerplate("a")
def searchsorted(
    a: ndarray,
    v: Union[int, float, ndarray],
    side: SortSide = "left",
    sorter: Optional[ndarray] = None,
) -> Union[int, ndarray]:
    """

    Find the indices into a sorted array a such that, if the corresponding
    elements in v were inserted before the indices, the order of a would be
    preserved.

    Parameters
    ----------
    a : 1-D array_like
        Input array. If sorter is None, then it must be sorted in ascending
        order, otherwise sorter must be an array of indices that sort it.
    v : scalar or array_like
        Values to insert into a.
    side : ``{'left', 'right'}``, optional
        If 'left', the index of the first suitable location found is given.
        If 'right', return the last such index. If there is no suitable index,
        return either 0 or N (where N is the length of a).
    sorter : 1-D array_like, optional
        Optional array of integer indices that sort array a into ascending
        order. They are typically the result of argsort.

    Returns
    -------
    indices : int or array_like[int]
        Array of insertion points with the same shape as v, or an integer
        if v is a scalar.

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.searchsorted(v, side, sorter)


@add_boilerplate("a")
def sort(
    a: ndarray,
    axis: int = -1,
    kind: SortType = "quicksort",
    order: Optional[Union[str, list[str]]] = None,
) -> ndarray:
    """

    Returns a sorted copy of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or None, optional
        Axis to sort. By default, the index -1 (the last axis) is used. If
        None, the flattened array is used.
    kind : ``{'quicksort', 'mergesort', 'heapsort', 'stable'}``, optional
        Default is 'quicksort'. The underlying sort algorithm might vary.
        The code basically supports 'stable' or *not* 'stable'.
    order : str or list[str], optional
        Currently not supported

    Returns
    -------
    out : ndarray
        Sorted array with same dtype and shape as `a`. In case `axis` is
        None the result is flattened.


    See Also
    --------
    numpy.sort

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    result = ndarray(a.shape, a.dtype)
    result._thunk.sort(rhs=a._thunk, axis=axis, kind=kind, order=order)
    return result


@add_boilerplate("a")
def sort_complex(a: ndarray) -> ndarray:
    """

    Returns a sorted copy of an array sorted along the last axis. Sorts the
    real part first, the imaginary part second.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    out : ndarray, complex
        Sorted array with same shape as `a`.

    See Also
    --------
    numpy.sort_complex

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    result = sort(a)
    # force complex result upon return
    if np.issubdtype(result.dtype, np.complexfloating):
        return result
    elif (
        np.issubdtype(result.dtype, np.integer) and result.dtype.itemsize <= 2
    ):
        return result.astype(np.complex64, copy=True)
    else:
        return result.astype(np.complex128, copy=True)


# partition


@add_boilerplate("a")
def argpartition(
    a: ndarray,
    kth: Union[int, Sequence[int]],
    axis: int = -1,
    kind: SelectKind = "introselect",
    order: Optional[Union[str, list[str]]] = None,
) -> ndarray:
    """

    Perform an indirect partition along the given axis.

    Parameters
    ----------
    a : array_like
        Input array.
    kth : int or Sequence[int]
    axis : int or None, optional
        Axis to partition. By default, the index -1 (the last axis) is used. If
        None, the flattened array is used.
    kind : ``{'introselect'}``, optional
        Currently not supported.
    order : str or list[str], optional
        Currently not supported.

    Returns
    -------
    out : ndarray[int]
        Array of indices that partitions a along the specified axis. It has the
        same shape as `a.shape` or is flattened in case of `axis` is None.


    Notes
    -----
    The current implementation falls back to `cunumeric.argsort`.

    See Also
    --------
    numpy.argpartition

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    result = ndarray(a.shape, np.int64)
    result._thunk.partition(
        rhs=a._thunk,
        argpartition=True,
        kth=kth,
        axis=axis,
        kind=kind,
        order=order,
    )
    return result


@add_boilerplate("a")
def partition(
    a: ndarray,
    kth: Union[int, Sequence[int]],
    axis: int = -1,
    kind: SelectKind = "introselect",
    order: Optional[Union[str, list[str]]] = None,
) -> ndarray:
    """

    Returns a partitioned copy of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    kth : int or Sequence[int]
    axis : int or None, optional
        Axis to partition. By default, the index -1 (the last axis) is used. If
        None, the flattened array is used.
    kind : ``{'introselect'}``, optional
        Currently not supported.
    order : str or list[str], optional
        Currently not supported.

    Returns
    -------
    out : ndarray
        Partitioned array with same dtype and shape as `a`. In case `axis` is
        None the result is flattened.

    Notes
    -----
    The current implementation falls back to `cunumeric.sort`.

    See Also
    --------
    numpy.partition

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    result = ndarray(a.shape, a.dtype)
    result._thunk.partition(
        rhs=a._thunk, kth=kth, axis=axis, kind=kind, order=order
    )
    return result


# Searching


@add_boilerplate("a")
def argmax(
    a: ndarray,
    axis: Optional[int] = None,
    out: Optional[ndarray] = None,
    *,
    keepdims: bool = False,
) -> ndarray:
    """

    Returns the indices of the maximum values along an axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.
    out : ndarray, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the array.

    Returns
    -------
    index_array : ndarray[int]
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.

    See Also
    --------
    numpy.argmax

    Notes
    -----
    CuNumeric's parallel implementation may yield different results from NumPy
    when the array contains NaN(s).

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.argmax(axis=axis, out=out, keepdims=keepdims)


@add_boilerplate("a")
def argmin(
    a: ndarray,
    axis: Optional[int] = None,
    out: Optional[ndarray] = None,
    *,
    keepdims: bool = False,
) -> ndarray:
    """

    Returns the indices of the minimum values along an axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.
    out : ndarray, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the array.

    Returns
    -------
    index_array : ndarray[int]
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.

    See Also
    --------
    numpy.argmin

    Notes
    -----
    CuNumeric's parallel implementation may yield different results from NumPy
    when the array contains NaN(s).

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.argmin(axis=axis, out=out, keepdims=keepdims)


# Counting


@add_boilerplate("a")
def count_nonzero(
    a: ndarray, axis: Optional[Union[int, tuple[int, ...]]] = None
) -> Union[int, ndarray]:
    """

    Counts the number of non-zero values in the array ``a``.

    Parameters
    ----------
    a : array_like
        The array for which to count non-zeros.
    axis : int or tuple, optional
        Axis or tuple of axes along which to count non-zeros.
        Default is None, meaning that non-zeros will be counted
        along a flattened version of ``a``.

    Returns
    -------
    count : int or ndarray[int]
        Number of non-zero values in the array along a given axis.
        Otherwise, the total number of non-zero values in the array
        is returned.

    See Also
    --------
    numpy.count_nonzero

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if a.size == 0:
        return 0
    return ndarray._perform_unary_reduction(
        UnaryRedCode.COUNT_NONZERO,
        a,
        res_dtype=np.dtype(np.uint64),
        axis=axis,
    )


############
# Statistics
############

# Averages and variances


@add_boilerplate("a")
def mean(
    a: ndarray,
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    dtype: Optional[np.dtype[Any]] = None,
    out: Optional[ndarray] = None,
    keepdims: bool = False,
) -> ndarray:
    """

    Compute the arithmetic mean along the specified axis.

    Returns the average of the array elements.  The average is taken over
    the flattened array by default, otherwise over the specified axis.
    `float64` intermediate and return values are used for integer inputs.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
    axis : None or int or tuple[int], optional
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.

        If this is a tuple of ints, a mean is performed over multiple axes,
        instead of a single axis or all the axes as before.
    dtype : data-type, optional
        Type to use in computing the mean.  For integer inputs, the default
        is `float64`; for floating point inputs, it is the same as the
        input dtype.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.
        See `ufuncs-output-type` for more details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `mean` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    m : ndarray
        If `out=None`, returns a new array of the same dtype a above
        containing the mean values, otherwise a reference to the output
        array is returned.

    See Also
    --------
    numpy.mean

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.mean(axis=axis, dtype=dtype, out=out, keepdims=keepdims)


# Histograms


@add_boilerplate("x", "weights")
def bincount(
    x: ndarray, weights: Optional[ndarray] = None, minlength: int = 0
) -> ndarray:
    """
    bincount(x, weights=None, minlength=0)

    Count number of occurrences of each value in array of non-negative ints.

    The number of bins (of size 1) is one larger than the largest value in
    `x`. If `minlength` is specified, there will be at least this number
    of bins in the output array (though it will be longer if necessary,
    depending on the contents of `x`).
    Each bin gives the number of occurrences of its index value in `x`.
    If `weights` is specified the input array is weighted by it, i.e. if a
    value ``n`` is found at position ``i``, ``out[n] += weight[i]`` instead
    of ``out[n] += 1``.

    Parameters
    ----------
    x : array_like
        1-D input array of non-negative ints.
    weights : array_like, optional
        Weights, array of the same shape as `x`.
    minlength : int, optional
        A minimum number of bins for the output array.

    Returns
    -------
    out : ndarray[int]
        The result of binning the input array.
        The length of `out` is equal to ``cunumeric.amax(x)+1``.

    Raises
    ------
    ValueError
        If the input is not 1-dimensional, or contains elements with negative
        values, or if `minlength` is negative.
    TypeError
        If the type of the input is float or complex.

    See Also
    --------
    numpy.bincount

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if x.ndim != 1:
        raise ValueError("the input array must be 1-dimensional")
    if weights is not None:
        if weights.shape != x.shape:
            raise ValueError("weights array must be same shape for bincount")
        if weights.dtype.kind == "c":
            raise ValueError("weights must be convertible to float64")
        # Make sure the weights are float64
        weights = weights.astype(np.float64)
    if x.dtype.kind != "i":
        raise TypeError("input array for bincount must be integer type")
    if minlength < 0:
        raise ValueError("'minlength' must not be negative")
    # Note that the following are non-blocking operations,
    # though passing their results to `int` is blocking
    max_val, min_val = amax(x), amin(x)
    if int(min_val) < 0:
        raise ValueError("the input array must have no negative elements")
    minlength = _builtin_max(minlength, int(max_val) + 1)
    if x.size == 1:
        # Handle the special case of 0-D array
        if weights is None:
            out = zeros((minlength,), dtype=np.dtype(np.int64))
            out[x[0]] = 1
        else:
            out = zeros((minlength,), dtype=weights.dtype)
            index = x[0]
            out[index] = weights[index]
    else:
        # Normal case of bincount
        if weights is None:
            out = ndarray(
                (minlength,),
                dtype=np.dtype(np.int64),
                inputs=(x, weights),
            )
            out._thunk.bincount(x._thunk)
        else:
            out = ndarray(
                (minlength,),
                dtype=weights.dtype,
                inputs=(x, weights),
            )
            out._thunk.bincount(x._thunk, weights=weights._thunk)
    return out
