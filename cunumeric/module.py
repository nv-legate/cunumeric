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

import math
import re
from collections import Counter
from functools import wraps
from inspect import signature
from itertools import chain
from typing import Optional, Set

import numpy as np
import opt_einsum as oe
from cunumeric._ufunc.comparison import maximum, minimum
from cunumeric._ufunc.floating import floor
from cunumeric._ufunc.math import add, multiply

from .array import (
    convert_to_cunumeric_ndarray,
    convert_to_predicate_ndarray,
    ndarray,
)
from .config import BinaryOpCode, UnaryRedCode
from .runtime import runtime
from .utils import inner_modes, matmul_modes, tensordot_modes

_builtin_abs = abs
_builtin_all = all
_builtin_any = any
_builtin_max = max
_builtin_min = min
_builtin_sum = sum


def add_boilerplate(*array_params: str):
    """
    Adds required boilerplate to the wrapped module-level ndarray function.

    Every time the wrapped function is called, this wrapper will:
    * Convert all specified array-like parameters, plus the special "out"
      parameter (if present), to cuNumeric ndarrays.
    * Convert the special "where" parameter (if present) to a valid predicate.
    * Handle the case of scalar cuNumeric ndarrays, by forwarding the operation
      to the equivalent `()`-shape numpy array (if the operation exists on base
      numpy).

    NOTE: Assumes that no parameters are mutated besides `out`.
    """
    keys: Set[str] = set(array_params)

    def decorator(func):
        assert not hasattr(
            func, "__wrapped__"
        ), "this decorator must be the innermost"

        # For each parameter specified by name, also consider the case where
        # it's passed as a positional parameter.
        indices: Set[int] = set()
        all_formals: Set[str] = set()
        where_idx: Optional[int] = None
        out_idx: Optional[int] = None
        for (idx, param) in enumerate(signature(func).parameters):
            all_formals.add(param)
            if param == "where":
                where_idx = idx
            elif param == "out":
                out_idx = idx
            elif param in keys:
                indices.add(idx)
        assert len(keys - all_formals) == 0, "unkonwn parameter(s)"

        @wraps(func)
        def wrapper(*args, **kwargs):
            assert (where_idx is None or len(args) <= where_idx) and (
                out_idx is None or len(args) <= out_idx
            ), "'where' and 'out' should be passed as keyword arguments"

            # Convert relevant arguments to cuNumeric ndarrays
            args = tuple(
                convert_to_cunumeric_ndarray(arg)
                if idx in indices and arg is not None
                else arg
                for (idx, arg) in enumerate(args)
            )
            for (k, v) in kwargs.items():
                if v is None:
                    continue
                elif k == "where":
                    kwargs[k] = convert_to_predicate_ndarray(v)
                elif k == "out":
                    kwargs[k] = convert_to_cunumeric_ndarray(v, share=True)
                elif k in keys:
                    kwargs[k] = convert_to_cunumeric_ndarray(v)

            # Handle the case where all array-like parameters are scalar, by
            # performing the operation on the equivalent scalar numpy arrays.
            # NOTE: This implicitly blocks on the contents of these arrays.
            if (
                hasattr(np, func.__name__)
                and _builtin_all(
                    isinstance(args[idx], ndarray) and args[idx]._thunk.scalar
                    for idx in indices
                )
                and _builtin_all(
                    isinstance(v, ndarray) and v._thunk.scalar
                    for v in (kwargs.get("where", None),)
                )
            ):
                out = None
                if "out" in kwargs:
                    out = kwargs["out"]
                    del kwargs["out"]
                args = tuple(
                    arg._thunk.__numpy_array__()
                    if (idx in indices) and isinstance(arg, ndarray)
                    else arg
                    for (idx, arg) in enumerate(args)
                )
                for (k, v) in kwargs.items():
                    if (k in keys or k == "where") and isinstance(v, ndarray):
                        kwargs[k] = v._thunk.__numpy_array__()
                result = convert_to_cunumeric_ndarray(
                    getattr(np, func.__name__)(*args, **kwargs)
                )
                if out is not None:
                    if out._thunk.dtype != result._thunk.dtype:
                        out._thunk.convert(result._thunk, warn=False)
                    else:
                        out._thunk.copy(result._thunk)
                    result = out
                return result

            return func(*args, **kwargs)

        return wrapper

    return decorator


#########################
# Array creation routines
#########################

# From shape or value


def empty(shape, dtype=np.float64):
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
def empty_like(a, dtype=None):
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
    shape = a.shape
    if dtype is not None:
        dtype = np.dtype(dtype)
    else:
        dtype = a.dtype
    return ndarray(shape, dtype=dtype, inputs=(a,))


def eye(N, M=None, k=0, dtype=np.float64):
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
    result = ndarray((N, M), dtype)
    result._thunk.eye(k)
    return result


def identity(n, dtype=float):
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


def ones(shape, dtype=np.float64):
    """

    Return a new array of given shape and type, filled with ones.

    Parameters
    ----------
    shape : int or Sequence[int]
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


def ones_like(a, dtype=None):
    """

    Return an array of ones with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of the
        returned array.
    dtype : data-type, optional
        Overrides the data type of the result.

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
    return full_like(a, 1, dtype=usedtype)


def zeros(shape, dtype=np.float64):
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


def zeros_like(a, dtype=None):
    """

    Return an array of zeros with the same shape and type as a given array.

    Parameters
    ----------
    a : array_like
        The shape and data-type of `a` define these same attributes of
        the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.

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
    return full_like(a, 0, dtype=usedtype)


def full(shape, value, dtype=None):
    """

    Return a new array of given shape and type, filled with `fill_value`.

    Parameters
    ----------
    shape : int or Sequence[int]
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


def full_like(a, value, dtype=None):
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
    result = empty_like(a, dtype=dtype)
    val = np.array(value, dtype=result.dtype)
    result._thunk.fill(val)
    return result


# From existing data


def array(obj, dtype=None, copy=True, order="K", subok=False, ndmin=0):
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


def asarray(a, dtype=None):
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
def copy(a):
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


def arange(start, stop=None, step=1, dtype=None):
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
    start,
    stop,
    num=50,
    endpoint=True,
    retstep=False,
    dtype=None,
    axis=0,
):
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
    step : float, optional
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

    dt = np.result_type(start, stop, float(num))
    if dtype is None:
        dtype = dt

    delta = stop - start
    y = arange(0, num, dtype=dt)

    # Reshape these arrays into dimensions that allow them to broadcast
    if delta.ndim > 0:
        if axis is None or axis == 0:
            # First dimension
            y = y.reshape((-1,) + (1,) * delta.ndim)
            # Nothing else needs to be reshaped here because
            # they should all broadcast correctly with y
            if endpoint and num > 1:
                out = -1
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
        out = -1
    # else delta is a scalar so start must be also
    # therefore it will trivially broadcast correctly

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
        if delta.nim == 0:
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
def diag(v, k=0):
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
    elif v.ndim > 2:
        raise ValueError("diag requires 1- or 2-D array, use diagonal instead")


@add_boilerplate("m")
def trilu(m, k, lower):
    if m.ndim < 1:
        raise TypeError("Array must be at least 1-D")
    shape = m.shape if m.ndim >= 2 else m.shape * 2
    result = ndarray(shape, dtype=m.dtype, inputs=(m,))
    result._thunk.trilu(m._thunk, k, lower)
    return result


def tril(m, k=0):
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


def triu(m, k=0):
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
def shape(a):
    """

    Return the shape of an array.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    shape : tuple[int]
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
def ravel(a, order="C"):
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
def reshape(a, newshape, order="C"):
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
def swapaxes(a, axis1, axis2):
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
def transpose(a, axes=None):
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


# Changing number of dimensions


@add_boilerplate("a")
def squeeze(a, axis=None):
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
    return a.squeeze(a, axis=axis)


# Joining arrays


class ArrayInfo:
    def __init__(self, ndim, shape, dtype):
        self.ndim = ndim
        self.shape = shape
        self.dtype = dtype


def convert_to_array_form(indices):
    return "".join(f"[{coord}]" for coord in indices)


def check_list_depth(arr, prefix=(0,)):
    if not isinstance(arr, list):
        return 0
    elif len(arr) == 0:
        raise ValueError(
            f"List at arrays{convert_to_array_form(prefix)} cannot be empty"
        )

    depths = [
        check_list_depth(each, prefix + (idx,)) for idx, each in enumerate(arr)
    ]
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


def check_shape_dtype(
    inputs, func_name, axis, dtype=None, casting="same_kind"
):
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
    if ndim > 1 and _builtin_any(
        shape[1:axis] != inp.shape[1:axis]
        and shape[axis + 1 :] != inp.shape[axis + 1 :]
        for inp in inputs
    ):
        raise ValueError(
            f"All arguments to {func_name} "
            "must have the same "
            "dimension size in all dimensions "
            "except the target axis"
        )

    # Cast arrays with the passed arguments (dtype, casting)
    if dtype is None:
        dtype = np.find_common_type((inp.dtype for inp in inputs), [])
    else:
        dtype = np.dtype(dtype)

    converted = list(inp.astype(dtype, casting=casting) for inp in inputs)
    return converted, ArrayInfo(ndim, shape, dtype)


def _block(arr, cur_depth, depth):
    if cur_depth < depth:
        inputs = list(_block(each, cur_depth + 1, depth) for each in arr)
    else:
        inputs = list(convert_to_cunumeric_ndarray(inp) for inp in arr)

    # this reshape of elements could be replaced
    # w/ np.atleast_*d when they're implemented
    # Computes the maximum number of dimensions for the concatenation
    max_ndim = _builtin_max(
        1 + (depth - cur_depth), *(inp.ndim for inp in inputs)
    )
    # Append leading 1's to make elements to have the same 'ndim'
    reshaped = list(
        inp.reshape((1,) * (max_ndim - inp.ndim) + inp.shape)
        if max_ndim > inp.ndim
        else inp
        for inp in inputs
    )
    return concatenate(reshaped, axis=-1 + (cur_depth - depth))


def _concatenate(
    inputs,
    axis=0,
    out=None,
    dtype=None,
    casting="same_kind",
    common_info=None,
):
    if axis < 0:
        axis += len(common_info.shape)
    leading_dim = _builtin_sum(arr.shape[axis] for arr in inputs)
    out_shape = list(common_info.shape)
    out_shape[axis] = leading_dim

    out_array = ndarray(
        shape=out_shape, dtype=common_info.dtype, inputs=inputs
    )

    # Copy the values over from the inputs
    offset = 0
    idx_arr = []
    for i in range(0, axis):
        idx_arr.append(slice(out_shape[i]))

    idx_arr.append(0)

    for i in range(axis + 1, common_info.ndim):
        idx_arr.append(slice(out_shape[i]))

    for inp in inputs:
        if inp.size > 0:
            idx_arr[axis] = slice(offset, offset + inp.shape[axis])
            out_array[tuple(idx_arr)] = inp
            offset += inp.shape[axis]
    return out_array


def append(arr, values, axis=None):
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


def block(arrays):
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

    result = _block(arrays, 1, depth)
    return result


def concatenate(inputs, axis=0, out=None, dtype=None, casting="same_kind"):
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
    # flatten arrays if axis == None and concatenate arrays on the first axis
    if axis is None:
        inputs = list(inp.ravel() for inp in inputs)
        axis = 0

    # Check to see if we can build a new tuple of cuNumeric arrays
    cunumeric_inputs, common_info = check_shape_dtype(
        inputs, concatenate.__name__, axis, dtype, casting
    )

    return _concatenate(
        cunumeric_inputs, axis, out, dtype, casting, common_info
    )


def stack(arrays, axis=0, out=None):
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
        raise ValueError("The target axis should be an integer")

    arrays, common_info = check_shape_dtype(arrays, stack.__name__, axis)

    if axis > common_info.ndim:
        raise ValueError(
            "The target axis should be smaller or"
            " equal to the number of dimensions"
            " of input arrays"
        )

    shape = list(common_info.shape)
    shape.insert(axis, 1)
    arrays = [arr.reshape(shape) for arr in arrays]
    common_info.shape = shape
    return _concatenate(arrays, axis, out=out, common_info=common_info)


def vstack(tup):
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
    inputs = list(convert_to_cunumeric_ndarray(inp) for inp in tup)
    reshaped = list(
        inp.reshape([1, inp.shape[0]]) if inp.ndim == 1 else inp
        for inp in inputs
    )
    tup, common_info = check_shape_dtype(reshaped, vstack.__name__, 0)
    common_info.shape = tup[0].shape

    return _concatenate(
        tup,
        axis=0,
        dtype=common_info.dtype,
        common_info=common_info,
    )


def hstack(tup):
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
    tup, common_info = check_shape_dtype(tup, hstack.__name__, 1)
    # When ndim == 1, hstack concatenates arrays along the first axis
    return _concatenate(
        tup,
        axis=(0 if common_info.ndim == 1 else 1),
        dtype=common_info.dtype,
        common_info=common_info,
    )


def dstack(tup):
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
    reshaped = []
    inputs = list(convert_to_cunumeric_ndarray(inp) for inp in tup)
    for arr in inputs:
        if arr.ndim == 1:
            arr = arr.reshape((1,) + arr.shape + (1,))
        elif arr.ndim == 2:
            arr = arr.reshape(arr.shape + (1,))
        reshaped.append(arr)
    tup, common_info = check_shape_dtype(reshaped, dstack.__name__, 2)

    return _concatenate(
        tup,
        axis=2,
        dtype=common_info.dtype,
        common_info=common_info,
    )


def column_stack(tup):
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
    tup, common_info = check_shape_dtype(tup, column_stack.__name__, 1)
    # When ndim == 1, hstack concatenates arrays along the first axis
    if common_info.ndim == 1:
        tup = list(inp.reshape([inp.shape[0], 1]) for inp in tup)
        common_info.shape = tup[0].shape
    return _concatenate(
        tup,
        axis=1,
        dtype=common_info.dtype,
        common_info=common_info,
    )


row_stack = vstack


# Splitting arrays


def split(a, indices, axis=0):
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


def array_split(a, indices, axis=0, equal=False):
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
    dtype = type(indices)
    split_pts = []
    if axis >= array.ndim:
        raise ValueError(
            f"array({array.shape}) has less dimensions than axis({axis})"
        )

    if dtype == int:
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
    elif (
        (dtype == np.ndarray and indices.dtype == int)
        or dtype == list
        or dtype == tuple
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
    in_shape = []

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


def dsplit(a, indices):
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


def hsplit(a, indices):
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


def vsplit(a, indices):
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
def tile(A, reps):
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
    reps : array_like
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
    if not hasattr(reps, "__len__"):
        reps = (reps,)
    # Figure out the shape of the destination array
    out_dims = A.ndim if A.ndim > len(reps) else len(reps)
    # Prepend ones until the dimensions match
    while len(reps) < out_dims:
        reps = (1,) + reps
    out_shape = ()
    # Prepend dimensions if necessary
    for dim in range(out_dims - A.ndim):
        out_shape += (reps[dim],)
    offset = len(out_shape)
    for dim in range(A.ndim):
        out_shape += (A.shape[dim] * reps[offset + dim],)
    assert len(out_shape) == out_dims
    result = ndarray(out_shape, dtype=A.dtype, inputs=(A,))
    result._thunk.tile(A._thunk, reps)
    return result


def repeat(a, repeats, axis=None):
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

    # when array is a scalar
    if np.ndim(a) == 0:
        if np.ndim(repeats) == 0:
            return full((repeats,), a)
        else:
            raise ValueError(
                "`repeat` with a scalar parameter `a` is only "
                "implemented for scalar values of the parameter `repeats`."
            )
    if np.ndim(repeats) > 1:
        raise ValueError("`repeats` should be scalar or 1D array")

    # array is an array
    array = convert_to_cunumeric_ndarray(a)
    if np.ndim(repeats) == 1:
        repeats = convert_to_cunumeric_ndarray(repeats)

    # if no axes specified, flatten array
    if axis is None:
        array = array.ravel()
        axis = 0

    # axes should be integer type
    if not isinstance(axis, int):
        raise TypeError("Axis should be integer type")
    axis = np.int32(axis)

    if axis >= array.ndim:
        return ValueError("axis exceeds dimension of the input array")

    # If repeats is on a zero sized axis, then return the array.
    if array.shape[axis] == 0:
        return array.copy()

    if np.ndim(repeats) == 1:
        if repeats.shape[0] == 1 and repeats.shape[0] != array.shape[axis]:
            repeats = repeats[0]

    # repeats is a scalar.
    if np.ndim(repeats) == 0:
        # repeats is 0
        if repeats == 0:
            empty_shape = list(array.shape)
            empty_shape[axis] = 0
            empty_shape = tuple(empty_shape)
            return ndarray(shape=empty_shape, dtype=array.dtype)
        # repeats should be integer type
        if not isinstance(repeats, int):
            runtime.warn(
                "converting repeats to an integer type",
                category=UserWarning,
            )
        repeats = np.int64(repeats)
        result = array._thunk.repeat(
            repeats=repeats,
            axis=axis,
            scalar_repeats=True,
        )
    # repeats is an array
    else:
        # repeats should be integer type
        if repeats.dtype != np.int64:
            runtime.warn(
                "converting repeats to an integer type",
                category=RuntimeWarning,
            )
        repeats = repeats.astype(np.int64)
        if repeats.shape[0] != array.shape[axis]:
            return ValueError("incorrect shape of repeats array")
        result = array._thunk.repeat(
            repeats=repeats._thunk, axis=axis, scalar_repeats=False
        )
    return ndarray(shape=result.shape, thunk=result)


# Rearranging elements


@add_boilerplate("m")
def flip(m, axis=None):
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


@add_boilerplate("a")
def nonzero(a):
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


@add_boilerplate("a", "x", "y")
def where(a, x=None, y=None):
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


# Indexing-like operations


@add_boilerplate("a")
def choose(a, choices, out=None, mode="raise"):
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


@add_boilerplate("a")
def diagonal(a, offset=0, axis1=None, axis2=None, extract=True, axes=None):
    """
    diagonal(a, offset=0, axis1=None, axis2=None)

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


################
# Linear algebra
################

# Matrix and vector products


@add_boilerplate("a", "b")
def inner(a, b, out=None):
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
    return _contract(a_modes, b_modes, out_modes, a, b, out=out)


@add_boilerplate("a", "b")
def dot(a, b, out=None):
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

        dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

    Parameters
    ----------
    a : array_like
        First argument.
    b : array_like
        Second argument.
    out : ndarray, optional
        Output argument. This must have the exact shape that would be returned
        if it was not present. If its dtype is not what would be expected from
        this operation, then the result will be (unsafely) cast to `out`.

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
def matmul(a, b, out=None):
    """
    Matrix product of two arrays.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays, scalars not allowed.
    out : ndarray, optional
        A location into which the result is stored. If provided, it must have
        a shape that matches the signature `(n,k),(k,m)->(n,m)`. If its dtype
        is not what would be expected from this operation, then the result will
        be (unsafely) cast to `out`.

    Returns
    -------
    output : ndarray
        The matrix product of the inputs.
        This is a scalar only when both x1, x2 are 1-d vectors.
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
      >>> dot(a, c).shape
      (9, 5, 7, 9, 5, 3)
      >>> matmul(a, c).shape
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
    return _contract(a_modes, b_modes, out_modes, a, b, out=out)


@add_boilerplate("a", "b")
def vdot(a, b, out=None):
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
def outer(a, b, out=None):
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
def tensordot(a, b, axes=2, out=None):
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
    return _contract(a_modes, b_modes, out_modes, a, b, out=out)


# Trivial multi-tensor contraction strategy: contract in input order
class NullOptimizer(oe.paths.PathOptimizer):
    def __call__(self, inputs, output, size_dict, memory_limit=None):
        return [(0, 1)] + [(0, -1)] * (len(inputs) - 2)


# Generalized tensor contraction
@add_boilerplate("a", "b")
def _contract(
    a_modes,
    b_modes,
    out_modes,
    a,
    b=None,
    out=None,
):
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

    # Handle duplicate modes on inputs
    c_a_modes = Counter(a_modes)
    for (mode, count) in c_a_modes.items():
        if count > 1:
            axes = [i for (i, m) in enumerate(a_modes) if m == mode]
            a = a._diag_helper(axes=axes)
            # diagonal is stored on last axis
            a_modes = [m for m in a_modes if m != mode] + [mode]
    c_b_modes = Counter(b_modes)
    for (mode, count) in c_b_modes.items():
        if count > 1:
            axes = [i for (i, m) in enumerate(b_modes) if m == mode]
            b = b._diag_helper(axes=axes)
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
    for (dim, mode) in reversed(list(enumerate(a_modes))):
        if mode not in b_modes and mode not in out_modes:
            a_modes.pop(dim)
            a = a.sum(axis=dim)
    for (dim, mode) in reversed(list(enumerate(b_modes))):
        if mode not in a_modes and mode not in out_modes:
            b_modes.pop(dim)
            b = b.sum(axis=dim)

    # Compute extent per mode. No need to consider broadcasting at this stage,
    # since it has been handled above.
    mode2extent = {}
    for (mode, extent) in chain(
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
    c_shape = ()
    c_bloated_shape = ()
    for (mode, extent) in zip(out_modes, out_shape):
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
    for (mode, extent) in zip(c_modes, c_shape):
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

    # Handle types
    c_dtype = ndarray.find_common_type(a, b) if b is not None else a.dtype
    out_dtype = out.dtype if out is not None else c_dtype

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
        # Check for type conversion on the way in
        if a.dtype != c.dtype:
            temp = ndarray(
                shape=a.shape,
                dtype=c.dtype,
                inputs=(a,),
            )
            temp._thunk.convert(a._thunk)
            a = temp
        if b.dtype != c.dtype:
            temp = ndarray(
                shape=b.shape,
                dtype=c.dtype,
                inputs=(b,),
            )
            temp._thunk.convert(b._thunk)
            b = temp
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
        if out is None:
            out = zeros(out_shape, out_dtype)
        out[...] = c.reshape(c_bloated_shape)
        return out
    if out is None and out_shape != c_shape:
        # We need to add missing dimensions, but they are all of size 1, so
        # we don't need to broadcast
        assert c_bloated_shape == out_shape
        return c.reshape(out_shape)
    if out is not None:
        # The output and result arrays are fully compatible, but we still
        # need to copy
        out[...] = c
        return out
    return c


def einsum(expr, *operands, out=None, optimize=False):
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
    if not optimize:
        optimize = NullOptimizer()
    # This call normalizes the expression (adds the output part if it's
    # missing, expands '...') and checks for some errors (mismatch on number
    # of dimensions between operand and expression, wrong number of operands,
    # unknown modes on output, a mode appearing under two different
    # non-singleton extents).
    operands, contractions = oe.contract_path(
        expr, *operands, einsum_call=True, optimize=optimize
    )
    for (indices, _, sub_expr, _, _) in contractions:
        assert len(indices) == 1 or len(indices) == 2
        a = operands.pop(indices[0])
        b = operands.pop(indices[1]) if len(indices) == 2 else None
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
            out=(out if len(operands) == 0 else None),
        )
        operands.append(sub_result)
    assert len(operands) == 1
    return operands[0]


@add_boilerplate("a")
def trace(a, offset=0, axis1=None, axis2=None, dtype=None, out=None):
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
def all(a, axis=None, out=None, keepdims=False, where=True):
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
def any(a, axis=None, out=None, keepdims=False, where=True):
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
def allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
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
    allclose : bool
        Returns True if the two arrays are equal within the given
        tolerance; False otherwise.

    See Also
    --------
    numpy.allclose

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if equal_nan:
        raise NotImplementedError(
            "cuNumeric does not support equal NaN yet for allclose"
        )
    args = (np.array(rtol, dtype=np.float64), np.array(atol, dtype=np.float64))
    return ndarray._perform_binary_reduction(
        BinaryOpCode.ALLCLOSE,
        a,
        b,
        dtype=np.dtype(bool),
        extra_args=args,
    )


@add_boilerplate("a", "b")
def array_equal(a, b):
    """

    True if two arrays have the same shape and elements, False otherwise.

    Parameters
    ----------
    a1, a2 : array_like
        Input arrays.

    Returns
    -------
    b : bool
        Returns True if the arrays are equal.

    See Also
    --------
    numpy.array_equal

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if a.shape != b.shape:
        return False
    return ndarray._perform_binary_reduction(
        BinaryOpCode.EQUAL, a, b, dtype=np.dtype(np.bool_)
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
    a,
    axis=None,
    dtype=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
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
    a,
    axis=None,
    dtype=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
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


# Exponents and logarithms


# Arithmetic operations


# Handling complex numbers


@add_boilerplate("val")
def real(val):
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
def imag(val):
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
    a,
    axis=None,
    dtype=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
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
    a,
    axis=None,
    dtype=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
):
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
def convolve(a, v, mode="full"):
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
def clip(a, a_min, a_max, out=None):
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
    ar,
    return_index=False,
    return_inverse=False,
    return_counts=False,
    axis=None,
):
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
    if any((return_index, return_inverse, return_counts, axis)):
        raise NotImplementedError(
            "Keyword arguments for `unique` are not yet supported"
        )

    return ar.unique()


##################################
# Sorting, searching, and counting
##################################

# Sorting


@add_boilerplate("a")
def argsort(a, axis=-1, kind="quicksort", order=None):
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

    Notes
    -----
    The current implementation has only limited support for distributed data.
    Distributed 1-D or flattened data will be broadcasted.

    See Also
    --------
    numpy.argsort

    Availability
    --------
    Multiple GPUs, Single CPU
    """

    result = ndarray(a.shape, np.int64)
    result._thunk.sort(
        rhs=a._thunk, argsort=True, axis=axis, kind=kind, order=order
    )
    return result


def msort(a):
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

    Notes
    -----
    The current implementation has only limited support for distributed data.
    Distributed 1-D  data will be broadcasted.

    See Also
    --------
    numpy.msort

    Availability
    --------
    Multiple GPUs, Single CPU
    """
    return sort(a, axis=0)


@add_boilerplate("a")
def sort(a, axis=-1, kind="quicksort", order=None):
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

    Notes
    -----
    The current implementation has only limited support for distributed data.
    Distributed 1-D or flattened data will be broadcasted.

    See Also
    --------
    numpy.sort

    Availability
    --------
    Multiple GPUs, Single CPU
    """
    result = ndarray(a.shape, a.dtype)
    result._thunk.sort(rhs=a._thunk, axis=axis, kind=kind, order=order)
    return result


@add_boilerplate("a")
def sort_complex(a):
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

    Notes
    -----
    The current implementation has only limited support for distributed data.
    Distributed 1-D data will be broadcasted.

    See Also
    --------
    numpy.sort_complex

    Availability
    --------
    Multiple GPUs, Single CPU
    """

    result = sort(a)
    # force complex result upon return
    if np.issubdtype(result.dtype, np.complexfloating):
        return result
    else:
        return result.astype(np.complex64, copy=True)


# partition


@add_boilerplate("a")
def argpartition(a, kth, axis=-1, kind="introselect", order=None):
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
    Multiple GPUs, Single CPU
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
def partition(a, kth, axis=-1, kind="introselect", order=None):
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
    Multiple GPUs, Single CPU
    """
    result = ndarray(a.shape, a.dtype)
    result._thunk.partition(
        rhs=a._thunk, kth=kth, axis=axis, kind=kind, order=order
    )
    return result


# Searching


@add_boilerplate("a")
def argmax(a, axis=None, out=None):
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

    Returns
    -------
    index_array : ndarray[int]
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.

    See Also
    --------
    numpy.argmax

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if out is not None:
        if out.dtype != np.int64:
            raise ValueError("output array must have int64 dtype")
    return a.argmax(axis=axis, out=out)


@add_boilerplate("a")
def argmin(a, axis=None, out=None):
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

    Returns
    -------
    index_array : ndarray[int]
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.

    See Also
    --------
    numpy.argmin

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    if out is not None:
        if out is not None and out.dtype != np.int64:
            raise ValueError("output array must have int64 dtype")
    return a.argmin(axis=axis, out=out)


# Counting


@add_boilerplate("a")
def count_nonzero(a, axis=None):
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
        axis=axis,
        dtype=np.dtype(np.uint64),
        check_types=False,
    )


############
# Statistics
############

# Averages and variances


@add_boilerplate("a")
def mean(a, axis=None, dtype=None, out=None, keepdims=False):
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


@add_boilerplate("a", "weights")
def bincount(a, weights=None, minlength=0):
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
    if weights is not None:
        if weights.shape != a.shape:
            raise ValueError("weights array must be same shape for bincount")
        if weights.dtype.kind == "c":
            raise ValueError("weights must be convertible to float64")
        # Make sure the weights are float64
        weights = weights.astype(np.float64)
    if a.dtype.kind != "i" and a.dtype.kind != "u":
        raise TypeError("input array for bincount must be integer type")
    # If nobody told us the size then compute it
    if minlength <= 0:
        minlength = int(amax(a)) + 1
    if a.size == 1:
        # Handle the special case of 0-D array
        if weights is None:
            out = zeros((minlength,), dtype=np.dtype(np.int64))
            out[a[0]] = 1
        else:
            out = zeros((minlength,), dtype=weights.dtype)
            index = a[0]
            out[index] = weights[index]
    else:
        # Normal case of bincount
        if weights is None:
            out = ndarray(
                (minlength,),
                dtype=np.dtype(np.int64),
                inputs=(a, weights),
            )
            out._thunk.bincount(a._thunk)
        else:
            out = ndarray(
                (minlength,),
                dtype=weights.dtype,
                inputs=(a, weights),
            )
            out._thunk.bincount(a._thunk, weights=weights._thunk)
    return out
