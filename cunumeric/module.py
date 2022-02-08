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

from .array import ndarray
from .config import BinaryOpCode, UnaryOpCode, UnaryRedCode
from .runtime import runtime

_builtin_abs = abs
_builtin_all = all
_builtin_any = any
_builtin_max = max
_builtin_min = min


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
                ndarray.convert_to_cunumeric_ndarray(arg)
                if idx in indices and arg is not None
                else arg
                for (idx, arg) in enumerate(args)
            )
            for (k, v) in kwargs.items():
                if v is None:
                    continue
                elif k == "where":
                    kwargs[k] = ndarray.convert_to_predicate_ndarray(v)
                elif k == "out":
                    kwargs[k] = ndarray.convert_to_cunumeric_ndarray(
                        v, share=True
                    )
                elif k in keys:
                    kwargs[k] = ndarray.convert_to_cunumeric_ndarray(v)

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
                result = ndarray.convert_to_cunumeric_ndarray(
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


def _output_float_dtype(input):
    # Floats keep their floating point kind, otherwise switch to float64
    if input.dtype.kind in ("f", "c"):
        return input.dtype
    else:
        return np.dtype(np.float64)


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
    shape : int or tuple of int
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
    GPU, CPU
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
    GPU, CPU
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
    I : ndarray of shape (N,M)
      An array where all elements are equal to zero, except for the `k`-th
      diagonal, whose values are equal to one.

    See Also
    --------
    numpy.eye

    Availability
    --------
    GPU, CPU
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
    GPU, CPU
    """
    return eye(N=n, M=n, dtype=dtype)


def ones(shape, dtype=np.float64):
    """

    Return a new array of given shape and type, filled with ones.

    Parameters
    ----------
    shape : int or sequence of ints
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
    GPU, CPU
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
    GPU, CPU
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
    shape : int or tuple of ints
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
    GPU, CPU
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
    GPU, CPU
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
    shape : int or sequence of ints
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
    GPU, CPU
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
    GPU, CPU
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
    order : {'K', 'A', 'C', 'F'}, optional
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
    GPU, CPU
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
    GPU, CPU
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
    GPU, CPU
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
    start : number, optional
        Start of interval.  The interval includes this value.  The default
        start value is 0.
    stop : number
        End of interval.  The interval does not include this value, except
        in some cases where `step` is not an integer and floating point
        round-off affects the length of `out`.
    step : number, optional
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.  The default
        step size is 1.  If `step` is specified as a position argument,
        `start` must also be given.
    dtype : dtype
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
    GPU, CPU
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
    dtype : dtype, optional
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
    GPU, CPU
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
    GPU, CPU
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
    m : array_like, shape (M, N)
        Input array.
    k : int, optional
        Diagonal above which to zero elements.  `k = 0` (the default) is the
        main diagonal, `k < 0` is below it and `k > 0` is above.

    Returns
    -------
    tril : ndarray, shape (M, N)
        Lower triangle of `m`, of same shape and data-type as `m`.

    See Also
    --------
    numpy.tril

    Availability
    --------
    GPU, CPU
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
    GPU, CPU
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
    shape : tuple of ints
        The elements of the shape tuple give the lengths of the
        corresponding array dimensions.

    See Also
    --------
    numpy.shape

    Availability
    --------
    GPU, CPU
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
    order : {'C','F', 'A', 'K'}, optional
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
    GPU, CPU
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
    newshape : int or tuple of ints
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is
        inferred from the length of the array and remaining dimensions.
    order : {'C', 'F', 'A'}, optional
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
    GPU, CPU
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
    GPU, CPU
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
    axes : list of ints, optional
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
    GPU, CPU
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
    axis : None or int or tuple of ints, optional
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
    GPU, CPU
    """
    return a.squeeze(a, axis=axis)


# Joining arrays


class ArrayInfo:
    def __init__(self, ndim, shape, dtype):
        self.ndim = ndim
        self.shape = shape
        self.dtype = dtype


def check_shape_dtype(inputs, func_name, dtype=None, casting="same_kind"):
    if len(inputs) == 0:
        raise ValueError("need at least one array to concatenate")

    inputs = list(ndarray.convert_to_cunumeric_ndarray(inp) for inp in inputs)
    ndim = inputs[0].ndim
    shape = inputs[0].shape

    if _builtin_any(ndim != inp.ndim for inp in inputs):
        raise ValueError(
            f"All arguments to {func_name} "
            "must have the same number of dimensions"
        )
    if ndim > 1 and _builtin_any(shape[1:] != inp.shape[1:] for inp in inputs):
        raise ValueError(
            f"All arguments to {func_name}"
            "must have the same "
            "dimension size in all dimensions "
            "except the target axis"
        )

    # Cast arrays with the passed arguments (dtype, casting)
    if dtype is None:
        dtype = np.min_scalar_type(inputs)
    else:
        dtype = np.dtype(dtype)

    converted = list(inp.astype(dtype, casting=casting) for inp in inputs)
    return converted, ArrayInfo(ndim, shape, dtype)


def _concatenate(
    inputs,
    axis=0,
    out=None,
    dtype=None,
    casting="same_kind",
    common_info=None,
):
    # Check to see if we can build a new tuple of cuNumeric arrays
    leading_dim = 0
    cunumeric_inputs = inputs

    leading_dim = common_info.shape[axis] * len(cunumeric_inputs)

    out_shape = list(common_info.shape)
    out_shape[axis] = leading_dim

    out_array = ndarray(
        shape=out_shape, dtype=common_info.dtype, inputs=cunumeric_inputs
    )

    # Copy the values over from the inputs
    offset = 0
    idx_arr = []
    for i in range(0, axis):
        idx_arr.append(slice(out_shape[i]))

    idx_arr.append(0)

    for i in range(axis + 1, common_info.ndim):
        idx_arr.append(slice(out_shape[i]))

    for inp in cunumeric_inputs:
        idx_arr[axis] = slice(offset, offset + inp.shape[axis])
        out_array[tuple(idx_arr)] = inp
        offset += inp.shape[axis]
    return out_array


def concatenate(inputs, axis=0, out=None, dtype=None, casting="same_kind"):
    """

    concatenate((a1, a2, ...), axis=0, out=None, dtype=None,
    casting="same_kind")

    Join a sequence of arrays along an existing axis.

    Parameters
    ----------
    a1, a2, ... : sequence of array_like
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).
    axis : int, optional
        The axis along which the arrays will be joined.  If axis is None,
        arrays are flattened before use.  Default is 0.
    out : ndarray, optional
        If provided, the destination to place the result. The shape must be
        correct, matching that of what concatenate would have returned if no
        out argument were specified.
    dtype : str or dtype
        If provided, the destination array will have this dtype. Cannot be
        provided together with `out`.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
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
    GPU, CPU
    """
    # Check to see if we can build a new tuple of cuNumeric arrays
    cunumeric_inputs, common_info = check_shape_dtype(
        inputs, concatenate.__name__, dtype, casting
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
    arrays : sequence of array_like
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
    GPU, CPU
    """
    fname = stack.__name__
    arrays, common_info = check_shape_dtype(arrays, fname)
    if axis > common_info.ndim:
        raise ValueError(
            "The target axis should be smaller or"
            " equal to the number of dimensions"
            " of input arrays"
        )
    else:
        shape = list(common_info.shape)
        shape.insert(axis, 1)
        for i, arr in enumerate(arrays):
            arrays[i] = arr.reshape(shape)
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
    tup : sequence of ndarrays
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
    GPU, CPU
    """
    fname = vstack.__name__
    # Reshape arrays in the `tup` if needed before concatenation
    tup, common_info = check_shape_dtype(tup, fname)
    if common_info.ndim == 1:
        for i, arr in enumerate(tup):
            tup[i] = arr.reshape([1, arr.shape[0]])
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
    tup : sequence of ndarrays
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
    GPU, CPU
    """
    fname = hstack.__name__
    tup, common_info = check_shape_dtype(tup, fname)
    if (
        common_info.ndim == 1
    ):  # When ndim == 1, hstack concatenates arrays along the first axis
        return _concatenate(
            tup,
            axis=0,
            dtype=common_info.dtype,
            common_info=common_info,
        )
    else:
        return _concatenate(
            tup,
            axis=1,
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
    tup : sequence of arrays
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
    GPU, CPU
    """
    fname = dstack.__name__
    tup, common_info = check_shape_dtype(tup, fname)
    # Reshape arrays to (1,N,1) for ndim ==1 or (M,N,1) for ndim == 2:
    if common_info.ndim <= 2:
        shape = list(tup[0].shape)
        if common_info.ndim == 1:
            shape.insert(0, 1)
        shape.append(1)
        common_info.shape = shape
        for i, arr in enumerate(tup):
            tup[i] = arr.reshape(shape)
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
    tup : sequence of 1-D or 2-D arrays.
        Arrays to stack. All of them must have the same first dimension.

    Returns
    -------
    stacked : 2-D array
        The array formed by stacking the given arrays.

    See Also
    --------
    numpy.column_stack

    Availability
    --------
    GPU, CPU
    """
    return hstack(tup)


row_stack = vstack


# Splitting arrays


def split(a, indices, axis=0):
    """

    Split an array into multiple sub-arrays as views into `ary`.

    Parameters
    ----------
    ary : ndarray
        Array to be divided into sub-arrays.
    indices_or_sections : int or 1-D array
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
    sub-arrays : list of ndarrays
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
    GPU, CPU
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
    GPU, CPU
    """
    array = ndarray.convert_to_cunumeric_ndarray(a)
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
    GPU, CPU
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
    GPU, CPU
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
    GPU, CPU
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
    GPU, CPU
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
    axis : None or int or tuple of ints, optional
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
    Single GPU/CPU only
    """
    return m.flip(axis=axis)


###################
# Binary operations
###################

# Elementwise bit operations


@add_boilerplate("a")
def invert(a, out=None, where=True, dtype=None):
    """
    Compute bit-wise inversion, or bit-wise NOT, element-wise.

    Computes the bit-wise NOT of the underlying binary representation of
    the integers in the input arrays. This ufunc implements the C/Python
    operator ``~``.

    Parameters
    ----------
    x : array_like
        Only integer and boolean types are handled.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : ndarray or scalar
        Result.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    numpy.invert

    Availability
    --------
    GPU, CPU
    """
    if a.dtype.type == np.bool_:
        # Boolean values are special, just do negatiion
        return ndarray.perform_unary_op(
            UnaryOpCode.LOGICAL_NOT,
            a,
            dst=out,
            dtype=dtype,
            out_dtype=np.dtype(np.bool_),
            where=where,
        )
    else:
        return ndarray.perform_unary_op(
            UnaryOpCode.INVERT, a, dst=out, dtype=dtype, where=where
        )


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
    GPU, CPU
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
    GPU, CPU
    """
    if x is None or y is None:
        if x is not None or y is not None:
            raise ValueError(
                "both 'x' and 'y' parameters must be specified together for"
                " 'where'"
            )
        return nonzero(a)
    return ndarray.perform_where(a, x, y)


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
    a : int array
        This array must contain integers in ``[0, n-1]``, where ``n`` is the
        number of choices, unless ``mode=wrap`` or ``mode=clip``, in which
        cases any integers are permissible.
    choices : sequence of arrays
        Choice arrays. `a` and all of the choices must be broadcastable to the
        same shape.  If `choices` is itself an array (not recommended), then
        its outermost dimension (i.e., the one corresponding to
        ``choices.shape[0]``) is taken as defining the "sequence".
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype. Note that `out` is always
        buffered if ``mode='raise'``; use other modes for better performance.
    mode : {'raise' (default), 'wrap', 'clip'}, optional
        Specifies how indices outside ``[0, n-1]`` will be treated:

          * 'raise' : an exception is raised
          * 'wrap' : value becomes value mod ``n``
          * 'clip' : values < 0 are mapped to 0, values > n-1 are mapped to n-1

    Returns
    -------
    merged_array : array
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
    GPU, CPU
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
    GPU, CPU

    """
    return a.diagonal(
        offset=offset, axis1=axis1, axis2=axis2, extract=extract, axes=axes
    )


################
# Linear algebra
################

# Matrix and vector products


@add_boilerplate("a", "b")
def dot(a, b, out=None):
    """

    Dot product of two arrays. Specifically,

    - If both `a` and `b` are 1-D arrays, it is inner product of vectors
      (without complex conjugation).

    - If both `a` and `b` are 2-D arrays, it is matrix multiplication,
      but using :func:`matmul` or ``a @ b`` is preferred.

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
        Output argument. This must have the exact kind that would be returned
        if it was not used. In particular, it must have the right type, must be
        C-contiguous, and its dtype must be the dtype that would be returned
        for `dot(a,b)`. This is a performance feature. Therefore, if these
        conditions are not met, an exception is raised, instead of attempting
        to be flexible.

    Returns
    -------
    output : ndarray
        Returns the dot product of `a` and `b`.  If `a` and `b` are both
        scalars or both 1-D arrays then a scalar is returned; otherwise
        an array is returned.
        If `out` is given, then it is returned.

    Raises
    ------
    ValueError
        If the last dimension of `a` is not the same size as
        the second-to-last dimension of `b`.

    Notes
    -----
    The current implementation only supports 1-D or 2-D input arrays.

    See Also
    --------
    numpy.dot

    Availability
    --------
    GPU, CPU
    """
    return a.dot(b, out=out)


def tensordot(a, b, axes=2):
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

    axes : int or (2,) array_like
        * integer_like
          If an int N, sum over the last N axes of `a` and the first N axes
          of `b` in order. The sizes of the corresponding axes must match.
        * (2,) array_like
          Or, a list of axes to be summed over, first sequence applying to `a`,
          second to `b`. Both elements array_like must be of the same length.

    Returns
    -------
    output : ndarray
        The tensor dot product of the input.

    Notes
    -----
    The current implementation inherits the limitation of `dot`; i.e., it only
    supports 1-D or 2-D input arrays.

    See Also
    --------
    numpy.tensordot

    Availability
    --------
    GPU, CPU
    """
    # This is the exact same code as the canonical numpy.
    # See https://github.com/numpy/numpy/blob/v1.21.0/numpy/core/numeric.py#L943-L1133. # noqa:  E501
    try:
        iter(axes)
    except Exception:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(0, axes))
    else:
        axes_a, axes_b = axes
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1

    as_ = a.shape
    nda = a.ndim
    bs = b.shape
    ndb = b.ndim
    equal = True
    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        raise ValueError("shape-mismatch for sum")

    # Move the axes to sum over to the end of "a"
    # and to the front of "b"
    notin = [k for k in range(nda) if k not in axes_a]
    newaxes_a = notin + axes_a
    N2 = 1
    for axis in axes_a:
        N2 *= as_[axis]
    newshape_a = (int(np.multiply.reduce([as_[ax] for ax in notin])), N2)
    olda = [as_[axis] for axis in notin]

    notin = [k for k in range(ndb) if k not in axes_b]
    newaxes_b = axes_b + notin
    N2 = 1
    for axis in axes_b:
        N2 *= bs[axis]
    newshape_b = (N2, int(np.multiply.reduce([bs[ax] for ax in notin])))
    oldb = [bs[axis] for axis in notin]

    at = a.transpose(newaxes_a).reshape(newshape_a)
    bt = b.transpose(newaxes_b).reshape(newshape_b)
    res = dot(at, bt)
    return res.reshape(olda + oldb)


# Trivial multi-tensor contraction strategy: contract in input order
class NullOptimizer(oe.paths.PathOptimizer):
    def __call__(self, inputs, output, size_dict, memory_limit=None):
        return [(0, 1)] + [(0, -1)] * (len(inputs) - 2)


# Generalized tensor contraction
@add_boilerplate("a", "b")
def _contract(expr, a, b=None, out=None):
    # Parse modes out of contraction expression (assuming expression has been
    # normalized already by contract_path)
    if b is None:
        m = re.match(r"([a-zA-Z]*)->([a-zA-Z]*)", expr)
        assert m is not None
        a_modes = list(m.group(1))
        b_modes = []
        out_modes = list(m.group(2))
    else:
        m = re.match(r"([a-zA-Z]*),([a-zA-Z]*)->([a-zA-Z]*)", expr)
        assert m is not None
        a_modes = list(m.group(1))
        b_modes = list(m.group(2))
        out_modes = list(m.group(3))

    # Sanity checks
    if out is not None and len(out_modes) != out.ndim:
        raise ValueError(
            f"Expected {len(out_modes)}-d output array but got {out.ndim}-d"
        )
    if len(set(out_modes)) != len(out_modes):
        raise ValueError("Duplicate mode labels on output tensor")

    # Handle duplicate modes on inputs
    c_a_modes = Counter(a_modes)
    for (mode, count) in c_a_modes.items():
        if count > 1:
            axes = [i for (i, m) in enumerate(a_modes) if m == mode]
            a = a.diag_helper(axes=axes)
            # diagonal is stored on last axis
            a_modes = [m for m in a_modes if m != mode] + [mode]
    c_b_modes = Counter(b_modes)
    for (mode, count) in c_b_modes.items():
        if count > 1:
            axes = [i for (i, m) in enumerate(b_modes) if m == mode]
            b = b.diag_helper(axes=axes)
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
        # This should have already been checked by contract_path
        assert prev_extent is None or extent == prev_extent
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
    if b is None or c_dtype in [
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ]:
        # Can support this type directly
        pass
    elif np.can_cast(c_dtype, np.float64):
        # Will have to go through a supported type, and cast the result back to
        # the input type (or the type of the provided output array)
        c_dtype = np.float64
    else:
        raise TypeError(f"Unsupported type: {c_dtype}")

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
    operands : list of array_like
        These are the arrays for the operation.
    out : ndarray, optional
        If provided, the calculation is done into this array.
    optimize : {False, True, 'greedy', 'optimal'}, optional
        Controls if intermediate optimization should occur. No optimization
        will occur if False. Uses opt_einsum to find an optimized contraction
        plan if True.

    Returns
    -------
    output : ndarray
        The calculation based on the Einstein summation convention.

    See Also
    --------
    numpy.einsum

    Availability
    --------
    GPU, CPU
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
        sub_opers = [operands.pop(i) for i in indices]
        if len(operands) == 0:  # last iteration
            sub_result = _contract(sub_expr, *sub_opers, out=out)
        else:
            sub_result = _contract(sub_expr, *sub_opers)
        operands.append(sub_result)
    assert len(operands) == 1
    return operands[0]


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
    axis : None or int or tuple of ints, optional
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
    GPU, CPU
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
    axis : None or int or tuple of ints, optional
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
    GPU, CPU
    """
    return a.any(axis=axis, out=out, keepdims=keepdims, where=where)


# Array contents


@add_boilerplate("a")
def isinf(a, out=None, where=True, dtype=None, **kwargs):
    """
    Test element-wise for positive or negative infinity.

    Returns a boolean array of the same shape as `x`, True where ``x ==
    +/-inf``, otherwise False.

    Parameters
    ----------
    x : array_like
        Input values
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : bool (scalar) or boolean ndarray
        True where ``x`` is positive or negative infinity, false otherwise.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    numpy.isinf

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_unary_op(
        UnaryOpCode.ISINF,
        a,
        dst=out,
        dtype=dtype,
        where=where,
        out_dtype=np.dtype(np.bool_),
    )


@add_boilerplate("a")
def isnan(a, out=None, where=True, dtype=None, **kwargs):
    """
    Test element-wise for NaN and return result as a boolean array.

    Parameters
    ----------
    x : array_like
        Input array.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : ndarray or bool
        True where ``x`` is NaN, false otherwise.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    numpy.isnan

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_unary_op(
        UnaryOpCode.ISNAN,
        a,
        dst=out,
        dtype=dtype,
        where=where,
        out_dtype=np.dtype(np.bool_),
    )


# Logic operations


@add_boilerplate("a", "b")
def logical_and(a, b, out=None, where=True, dtype=np.dtype(np.bool), **kwargs):
    """
    Compute the truth value of x1 AND x2 element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays. If ``x1.shape != x2.shape``, they must be broadcastable
        to a common shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : ndarray or bool
        Boolean result of the logical AND operation applied to the elements
        of `x1` and `x2`; the shape is determined by broadcasting.
        This is a scalar if both `x1` and `x2` are scalars.

    See Also
    --------
    numpy.logical_and

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_binary_op(
        BinaryOpCode.LOGICAL_AND,
        a,
        b,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
        where=where,
    )


@add_boilerplate("a", "b")
def logical_or(a, b, out=None, where=True, dtype=np.dtype(np.bool), **kwargs):
    """
    Compute the truth value of x1 OR x2 element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Logical OR is applied to the elements of `x1` and `x2`.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : ndarray or bool
        Boolean result of the logical OR operation applied to the elements
        of `x1` and `x2`; the shape is determined by broadcasting.
        This is a scalar if both `x1` and `x2` are scalars.

    See Also
    --------
    numpy.logical_or

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_binary_op(
        BinaryOpCode.LOGICAL_OR,
        a,
        b,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
        where=where,
    )


@add_boilerplate("a")
def logical_not(a, out=None, where=True, dtype=None, **kwargs):
    """
    Compute the truth value of NOT x element-wise.

    Parameters
    ----------
    x : array_like
        Logical NOT is applied to the elements of `x`.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : bool or ndarray of bool
        Boolean result with the same shape as `x` of the NOT operation
        on elements of `x`.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    numpy.logical_not

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_unary_op(
        UnaryOpCode.LOGICAL_NOT,
        a,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=np.dtype(np.bool_),
    )


@add_boilerplate("a", "b")
def logical_xor(a, b, out=None, where=True, dtype=np.dtype(np.bool), **kwargs):
    """
    Compute the truth value of x1 XOR x2, element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Logical XOR is applied to the elements of `x1` and `x2`. If ``x1.shape
        != x2.shape``, they must be broadcastable to a common shape (which
        becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : bool or ndarray of bool
        Boolean result of the logical XOR operation applied to the elements
        of `x1` and `x2`; the shape is determined by broadcasting.
        This is a scalar if both `x1` and `x2` are scalars.

    See Also
    --------
    numpy.logical_xor

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_binary_op(
        BinaryOpCode.LOGICAL_XOR,
        a,
        b,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
        where=where,
    )


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
    GPU, CPU
    """
    if equal_nan:
        raise NotImplementedError(
            "cuNumeric does not support equal NaN yet for allclose"
        )
    args = (np.array(rtol, dtype=np.float64), np.array(atol, dtype=np.float64))
    return ndarray.perform_binary_reduction(
        BinaryOpCode.ALLCLOSE,
        a,
        b,
        dtype=np.dtype(np.bool),
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
    GPU, CPU
    """
    if a.shape != b.shape:
        return False
    return ndarray.perform_binary_reduction(
        BinaryOpCode.EQUAL, a, b, dtype=np.dtype(np.bool_)
    )


@add_boilerplate("a", "b")
def greater(a, b, out=None, where=True, dtype=np.dtype(np.bool), **kwargs):
    """
    Return the truth value of (x1 > x2) element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays. If ``x1.shape != x2.shape``, they must be broadcastable
        to a common shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : ndarray or scalar
        Output array, element-wise comparison of `x1` and `x2`.
        Typically of type bool, unless ``dtype=object`` is passed.
        This is a scalar if both `x1` and `x2` are scalars.

        See Also
        --------
        numpy.greater

        Availability
        --------
        GPU, CPU
    """
    return ndarray.perform_binary_op(
        BinaryOpCode.GREATER,
        a,
        b,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
        where=where,
    )


@add_boilerplate("a", "b")
def greater_equal(
    a, b, out=None, where=True, dtype=np.dtype(np.bool), **kwargs
):
    """
    Return the truth value of (x1 >= x2) element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays. If ``x1.shape != x2.shape``, they must be broadcastable
        to a common shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : bool or ndarray of bool
        Output array, element-wise comparison of `x1` and `x2`.
        Typically of type bool, unless ``dtype=object`` is passed.
        This is a scalar if both `x1` and `x2` are scalars.

    See Also
    --------
    numpy.greater_equal

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_binary_op(
        BinaryOpCode.GREATER_EQUAL,
        a,
        b,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
        where=where,
    )


@add_boilerplate("a", "b")
def less(a, b, out=None, where=True, dtype=np.dtype(np.bool), **kwargs):
    """
    Return the truth value of (x1 < x2) element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays. If ``x1.shape != x2.shape``, they must be broadcastable
        to a common shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : ndarray or scalar
        Output array, element-wise comparison of `x1` and `x2`.
        Typically of type bool, unless ``dtype=object`` is passed.
        This is a scalar if both `x1` and `x2` are scalars.

    See Also
    --------
    numpy.less

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_binary_op(
        BinaryOpCode.LESS,
        a,
        b,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
        where=where,
    )


@add_boilerplate("a", "b")
def less_equal(a, b, out=None, where=True, dtype=np.dtype(np.bool), **kwargs):
    """
    Return the truth value of (x1 =< x2) element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays. If ``x1.shape != x2.shape``, they must be broadcastable
        to a common shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : ndarray or scalar
        Output array, element-wise comparison of `x1` and `x2`.
        Typically of type bool, unless ``dtype=object`` is passed.
        This is a scalar if both `x1` and `x2` are scalars.

    See Also
    --------
    numpy.less_equal

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_binary_op(
        BinaryOpCode.LESS_EQUAL,
        a,
        b,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
        where=where,
    )


@add_boilerplate("a", "b")
def equal(a, b, out=None, where=True, dtype=np.dtype(np.bool), **kwargs):
    """
    Return (x1 == x2) element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays. If ``x1.shape != x2.shape``, they must be broadcastable
        to a common shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : ndarray or scalar
        Output array, element-wise comparison of `x1` and `x2`.
        Typically of type bool, unless ``dtype=object`` is passed.
        This is a scalar if both `x1` and `x2` are scalars.

    See Also
    --------
    numpy.equal

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_binary_op(
        BinaryOpCode.EQUAL,
        a,
        b,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
        where=where,
    )


@add_boilerplate("a", "b")
def not_equal(a, b, out=None, where=True, dtype=np.dtype(np.bool), **kwargs):
    """
    Return (x1 != x2) element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays.  If ``x1.shape != x2.shape``, they must be broadcastable
        to a common shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : ndarray or scalar
        Output array, element-wise comparison of `x1` and `x2`.
        Typically of type bool, unless ``dtype=object`` is passed.
        This is a scalar if both `x1` and `x2` are scalars.

    See Also
    --------
    numpy.not_equal

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_binary_op(
        BinaryOpCode.NOT_EQUAL,
        a,
        b,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
        where=where,
    )


########################
# Mathematical functions
########################

# Trigonometric functions


@add_boilerplate("a")
def sin(a, out=None, where=True, dtype=None, **kwargs):
    """
    Trigonometric sine, element-wise.

    Parameters
    ----------
    x : array_like
        Angle, in radians (:math:`2 \\pi` rad equals 360 degrees).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : array_like
        The sine of each element of x.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    numpy.sin

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_unary_op(
        UnaryOpCode.SIN,
        a,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=_output_float_dtype(a),
    )


@add_boilerplate("a")
def cos(a, out=None, where=True, dtype=None, **kwargs):
    """
    Cosine element-wise.

    Parameters
    ----------
    x : array_like
        Input array in radians.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : ndarray
        The corresponding cosine values.
        This is a scalar if `x` is a scalar.

    Notes
    -----
    If `out` is provided, the function writes the result into it,
    and returns a reference to `out`.  (See

    See Also
    --------
    numpy.cos

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_unary_op(
        UnaryOpCode.COS,
        a,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=_output_float_dtype(a),
    )


@add_boilerplate("a")
def tan(a, out=None, where=True, dtype=None, **kwargs):
    """
    Compute tangent element-wise.

    Parameters
    ----------
    x : array_like
        Input array.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : ndarray
        The corresponding tangent values.
        This is a scalar if `x` is a scalar.

    Notes
    -----
    If `out` is provided, the function writes the result into it,
    and returns a reference to `out`.  (See

    See Also
    --------
    numpy.tan

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_unary_op(
        UnaryOpCode.TAN,
        a,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=_output_float_dtype(a),
    )


@add_boilerplate("a")
def arcsin(a, out=None, where=True, dtype=None, **kwargs):
    """
    Inverse sine, element-wise.

    Parameters
    ----------
    x : array_like
        `y`-coordinate on the unit circle.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    angle : ndarray
        The inverse sine of each element in `x`, in radians and in the
        closed interval ``[-pi/2, pi/2]``.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    numpy.arcsin

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_unary_op(
        UnaryOpCode.ARCSIN,
        a,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=_output_float_dtype(a),
    )


@add_boilerplate("a")
def arccos(a, out=None, where=True, dtype=None, **kwargs):
    """
    Trigonometric inverse cosine, element-wise.

    The inverse of `cos` so that, if ``y = cos(x)``, then ``x = arccos(y)``.

    Parameters
    ----------
    x : array_like
        `x`-coordinate on the unit circle.
        For real arguments, the domain is [-1, 1].
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    angle : ndarray
        The angle of the ray intersecting the unit circle at the given
        `x`-coordinate in radians [0, pi].
        This is a scalar if `x` is a scalar.

    See Also
    --------
    numpy.arccos

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_unary_op(
        UnaryOpCode.ARCCOS,
        a,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=_output_float_dtype(a),
    )


@add_boilerplate("a")
def arctan(a, out=None, where=True, dtype=None, **kwargs):
    """
    Trigonometric inverse tangent, element-wise.

    The inverse of tan, so that if ``y = tan(x)`` then ``x = arctan(y)``.

    Parameters
    ----------
    x : array_like
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : ndarray or scalar
        Out has the same shape as `x`.  Its real part is in
        ``[-pi/2, pi/2]`` (``arctan(+/-inf)`` returns ``+/-pi/2``).
        This is a scalar if `x` is a scalar.

    See Also
    --------
    numpy.arctan

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_unary_op(
        UnaryOpCode.ARCTAN,
        a,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=_output_float_dtype(a),
    )


# Hyperbolic functions


@add_boilerplate("a")
def tanh(a, out=None, where=True, dtype=None, **kwargs):
    """
    Compute hyperbolic tangent element-wise.

    Parameters
    ----------
    x : array_like
        Input array.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : ndarray
        The corresponding hyperbolic tangent values.
        This is a scalar if `x` is a scalar.

    Notes
    -----
    If `out` is provided, the function writes the result into it,
    and returns a reference to `out`.  (See

    See Also
    --------
    numpy.tanh

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_unary_op(
        UnaryOpCode.TANH,
        a,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=_output_float_dtype(a),
    )


# Rounding


@add_boilerplate("a")
def rint(a, out=None, where=True, dtype=None, **kwargs):
    """
    Round elements of the array to the nearest integer.

    Parameters
    ----------
    x : array_like
        Input array.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : ndarray or scalar
        Output array is same shape and type as `x`.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    numpy.rint

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_unary_op(
        UnaryOpCode.RINT,
        a,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=_output_float_dtype(a),
    )


@add_boilerplate("a")
def floor(a, out=None, where=True, dtype=None, **kwargs):
    """
    Return the floor of the input, element-wise.

    The floor of the scalar `x` is the largest integer `i`, such that
    `i <= x`.  It is often denoted as :math:`\\lfloor x \\rfloor`.

    Parameters
    ----------
    x : array_like
        Input data.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : ndarray or scalar
        The floor of each element in `x`.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    numpy.floor

    Availability
    --------
    GPU, CPU
    """
    # If this is an integer array then there is nothing to do for floor
    if a.dtype.kind in ("i", "u", "b"):
        return a
    return ndarray.perform_unary_op(
        UnaryOpCode.FLOOR, a, dst=out, dtype=dtype, where=where
    )


@add_boilerplate("a")
def ceil(a, out=None, where=True, dtype=None, **kwargs):
    """
    Return the ceiling of the input, element-wise.

    The ceil of the scalar `x` is the smallest integer `i`, such that
    `i >= x`.  It is often denoted as :math:`\\lceil x \\rceil`.

    Parameters
    ----------
    x : array_like
        Input data.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : ndarray or scalar
        The ceiling of each element in `x`, with `float` dtype.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    numpy.ceil

    Availability
    --------
    GPU, CPU
    """
    # If this is an integer array then there is nothing to do for ceil
    if a.dtype.kind in ("i", "u", "b"):
        return a
    return ndarray.perform_unary_op(
        UnaryOpCode.CEIL, a, dst=out, dtype=dtype, where=where
    )


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
    axis : None or int or tuple of ints, optional
        Axis or axes along which a product is performed.  The default,
        axis=None, will calculate the product of all the elements in the
        input array. If axis is negative it counts from the last to the
        first axis.

        If axis is a tuple of ints, a product is performed on all of the
        axes specified in the tuple instead of a single axis or all the
        axes as before.
    dtype : dtype, optional
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

    where : array_like of bool, optional
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
    GPU, CPU
    """
    return a.prod(
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
    axis : None or int or tuple of ints, optional
        Axis or axes along which a sum is performed.  The default,
        axis=None, will sum all of the elements of the input array.  If
        axis is negative it counts from the last to the first axis.

        If axis is a tuple of ints, a sum is performed on all of the axes
        specified in the tuple instead of a single axis or all the axes as
        before.
    dtype : dtype, optional
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

    where : array_like of bool, optional
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
    GPU, CPU
    """
    return a.sum(
        axis=axis,
        dtype=dtype,
        out=out,
        keepdims=keepdims,
        initial=initial,
        where=where,
    )


# Exponents and logarithms


@add_boilerplate("a")
def exp(a, out=None, where=True, dtype=None, **kwargs):
    """
    Calculate the exponential of all elements in the input array.

    Parameters
    ----------
    x : array_like
        Input values.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : ndarray or scalar
        Output array, element-wise exponential of `x`.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    numpy.exp

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_unary_op(
        UnaryOpCode.EXP,
        a,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=_output_float_dtype(a),
    )


@add_boilerplate("a")
def exp2(a, out=None, where=True, dtype=None, **kwargs):
    """
    Calculate `2**p` for all `p` in the input array.

    Parameters
    ----------
    x : array_like
        Input values.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : ndarray or scalar
        Element-wise 2 to the power `x`.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    numpy.exp2

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_unary_op(
        UnaryOpCode.EXP2,
        a,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=_output_float_dtype(a),
    )


@add_boilerplate("a")
def log(a, out=None, where=True, dtype=None, **kwargs):
    """
    Natural logarithm, element-wise.

    The natural logarithm `log` is the inverse of the exponential function,
    so that `log(exp(x)) = x`. The natural logarithm is logarithm in base
    `e`.

    Parameters
    ----------
    x : array_like
        Input value.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : ndarray
        The natural logarithm of `x`, element-wise.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    numpy.log

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_unary_op(
        UnaryOpCode.LOG,
        a,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=_output_float_dtype(a),
    )


@add_boilerplate("a")
def log10(a, out=None, where=True, dtype=None, **kwargs):
    """
    Return the base 10 logarithm of the input array, element-wise.

    Parameters
    ----------
    x : array_like
        Input values.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : ndarray
        The logarithm to the base 10 of `x`, element-wise. NaNs are
        returned where x is negative.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    numpy.log10

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_unary_op(
        UnaryOpCode.LOG10,
        a,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=_output_float_dtype(a),
    )


# Arithmetic operations


@add_boilerplate("a", "b")
def add(a, b, out=None, where=True, dtype=None):
    """
    Add arguments element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        The arrays to be added. If ``x1.shape != x2.shape``, they must be
        broadcastable to a common shape (which becomes the shape of the
        output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    add : ndarray or scalar
        The sum of `x1` and `x2`, element-wise.
        This is a scalar if both `x1` and `x2` are scalars.

    Notes
    -----
    Equivalent to `x1` + `x2` in terms of array broadcasting.

    See Also
    --------
    numpy.add

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_binary_op(
        BinaryOpCode.ADD,
        a,
        b,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
        where=where,
    )


@add_boilerplate("a")
def negative(a, out=None, where=True, dtype=None, **kwargs):
    """
    Numerical negative, element-wise.

    Parameters
    ----------
    x : array_like or scalar
        Input array.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : ndarray or scalar
        Returned array or scalar: `y = -x`.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    numpy.negative

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_unary_op(
        UnaryOpCode.NEGATIVE, a, dtype=dtype, dst=out, where=where
    )


@add_boilerplate("a", "b")
def multiply(a, b, out=None, where=True, dtype=None):
    """
    Multiply arguments element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        Input arrays to be multiplied. If ``x1.shape != x2.shape``, they must
        be broadcastable to a common shape (which becomes the shape of the
        output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : ndarray
        The product of `x1` and `x2`, element-wise.
        This is a scalar if both `x1` and `x2` are scalars.

    Notes
    -----
    Equivalent to `x1` * `x2` in terms of array broadcasting.

    See Also
    --------
    numpy.multiply

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_binary_op(
        BinaryOpCode.MULTIPLY,
        a,
        b,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
        where=where,
    )


@add_boilerplate("x1", "x2")
def power(x1, x2, out=None, where=True, dtype=None, **kwargs):
    """
    First array elements raised to powers from second array, element-wise.

    Raise each base in `x1` to the positionally-corresponding power in
    `x2`.  `x1` and `x2` must be broadcastable to the same shape. Note that an
    integer type raised to a negative integer power will raise a ValueError.

    Parameters
    ----------
    x1 : array_like
        The bases.
    x2 : array_like
        The exponents. If ``x1.shape != x2.shape``, they must be broadcastable
        to a common shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : ndarray
        The bases in `x1` raised to the exponents in `x2`.
        This is a scalar if both `x1` and `x2` are scalars.

    See Also
    --------
    numpy.power

    Availability
    --------
    GPU, CPU
    """
    if out is None and dtype is None:
        if x1.dtype.kind == "f" or x2.dtype.kind == "f":
            array_types = list()
            scalar_types = list()
            if x1.ndim > 0:
                array_types.append(x1.dtype)
            else:
                scalar_types.append(x1.dtype)
            if x2.ndim > 0:
                array_types.append(x2.dtype)
            else:
                scalar_types.append(x2.dtype)
            dtype = np.find_common_type(array_types, scalar_types)
    return ndarray.perform_binary_op(
        BinaryOpCode.POWER,
        x1,
        x2,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
        where=where,
    )


@add_boilerplate("a", "b")
def subtract(a, b, out=None, where=True, dtype=None, **kwargs):
    """
    Subtract arguments, element-wise.

    Parameters
    ----------
    x1, x2 : array_like
        The arrays to be subtracted from each other. If ``x1.shape !=
        x2.shape``, they must be broadcastable to a common shape (which becomes
        the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : ndarray
        The difference of `x1` and `x2`, element-wise.
        This is a scalar if both `x1` and `x2` are scalars.

    Notes
    -----
    Equivalent to ``x1 - x2`` in terms of array broadcasting.

    See Also
    --------
    numpy.subtract

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_binary_op(
        BinaryOpCode.SUBTRACT,
        a,
        b,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
        where=where,
    )


@add_boilerplate("a", "b")
def true_divide(a, b, out=None, where=True, dtype=None, **kwargs):
    """
    Returns a true division of the inputs, element-wise.

    Instead of the Python traditional 'floor division', this returns a true
    division.  True division adjusts the output type to present the best
    answer, regardless of input types.

    Parameters
    ----------
    x1 : array_like
        Dividend array.
    x2 : array_like
        Divisor array. If ``x1.shape != x2.shape``, they must be broadcastable
        to a common shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : ndarray or scalar
        This is a scalar if both `x1` and `x2` are scalars.

    Notes
    -----
    The floor division operator ``//`` was added in Python 2.2 making
    ``//`` and ``/`` equivalent operators.  The default floor division
    operation of ``/`` can be replaced by true division with ``from
    __future__ import division``.

    In Python 3.0, ``//`` is the floor division operator and ``/`` the
    true division operator.  The ``true_divide(x1, x2)`` function is
    equivalent to true division in Python.

    See Also
    --------
    numpy.true_divide

    Availability
    --------
    GPU, CPU
    """
    # Convert any non-floats to floating point arrays
    if a.dtype.kind != "f":
        a_type = np.dtype(np.float64)
    else:
        a_type = a.dtype
    if b.dtype.kind != "f":
        b_type = np.dtype(np.float64)
    else:
        b_type = b.dtype
    # If the types don't match then align them
    if a_type != b_type:
        array_types = list()
        scalar_types = list()
        if a.ndim > 0:
            array_types.append(a_type)
        else:
            scalar_types.append(a_type)
        if b.ndim > 0:
            array_types.append(b_type)
        else:
            scalar_types.append(b_type)
        common_type = np.find_common_type(array_types, scalar_types)
    else:
        common_type = a_type
    if a.dtype != common_type:
        temp = ndarray(
            a.shape,
            dtype=common_type,
            inputs=(a, b),
        )
        temp._thunk.convert(a._thunk, warn=False)
        a = temp
    if b.dtype != common_type:
        temp = ndarray(
            b.shape,
            dtype=common_type,
            inputs=(a, b),
        )
        temp._thunk.convert(b._thunk, warn=False)
        b = temp
    return ndarray.perform_binary_op(
        BinaryOpCode.DIVIDE,
        a,
        b,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
        where=where,
    )


divide = true_divide  # Had to be reordered to make the reference valid


@add_boilerplate("a", "b")
def floor_divide(a, b, out=None, where=True, dtype=None, **kwargs):
    """
    Return the largest integer smaller or equal to the division of the inputs.
    It is equivalent to the Python ``//`` operator and pairs with the
    Python ``%`` (`remainder`), function so that ``a = a % b + b * (a // b)``
    up to roundoff.

    Parameters
    ----------
    x1 : array_like
        Numerator.
    x2 : array_like
        Denominator. If ``x1.shape != x2.shape``, they must be broadcastable to
        a common shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : ndarray
        y = floor(`x1`/`x2`)
        This is a scalar if both `x1` and `x2` are scalars.

    See Also
    --------
    numpy.floor_divide

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_binary_op(
        BinaryOpCode.FLOOR_DIVIDE,
        a,
        b,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
        where=where,
    )


@add_boilerplate("a", "b")
def remainder(a, b, out=None, where=True, dtype=None):
    """
    Return element-wise remainder of division.

    Computes the remainder complementary to the `floor_divide` function.  It is
    equivalent to the Python modulus operator``x1 % x2`` and has the same sign
    as the divisor `x2`.

    Parameters
    ----------
    x1 : array_like
        Dividend array.
    x2 : array_like
        Divisor array. If ``x1.shape != x2.shape``, they must be broadcastable
        to a common shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : ndarray
        The element-wise remainder of the quotient ``floor_divide(x1, x2)``.
        This is a scalar if both `x1` and `x2` are scalars.

    See Also
    --------
    numpy.remainder

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_binary_op(
        BinaryOpCode.MOD,
        a,
        b,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
        where=where,
    )


mod = remainder  # Had to be reordered to make the reference safe

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
    GPU, CPU
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
    GPU, CPU
    """
    return val.imag


@add_boilerplate("x")
def conjugate(x):
    """
    Return the complex conjugate, element-wise.

    The complex conjugate of a complex number is obtained by changing the
    sign of its imaginary part.

    Parameters
    ----------
    x : array_like
        Input value.

    Returns
    -------
    y : ndarray
        The complex conjugate of `x`, with same dtype as `y`.
        This is a scalar if `x` is a scalar.

    Notes
    -----
    `conj` is an alias for `conjugate`:

    See Also
    --------
    numpy.conjugate

    Availability
    --------
    GPU, CPU
    """
    return x.conj()


conj = conjugate

# Extrema Finding


@add_boilerplate("a", "b")
def maximum(a, b, out=None, where=True, dtype=None, **kwargs):
    """
    Element-wise maximum of array elements.

    Compare two arrays and returns a new array containing the element-wise
    maxima. If one of the elements being compared is a NaN, then that
    element is returned. If both elements are NaNs then the first is
    returned. The latter distinction is important for complex NaNs, which
    are defined as at least one of the real or imaginary parts being a NaN.
    The net effect is that NaNs are propagated.

    Parameters
    ----------
    x1, x2 : array_like
        The arrays holding the elements to be compared. If ``x1.shape !=
        x2.shape``, they must be broadcastable to a common shape (which becomes
        the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : ndarray or scalar
        The maximum of `x1` and `x2`, element-wise.
        This is a scalar if both `x1` and `x2` are scalars.

        See Also
        --------
        numpy.maximum

        Availability
        --------
        GPU, CPU
    """
    return ndarray.perform_binary_op(
        BinaryOpCode.MAXIMUM,
        a,
        b,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
        where=where,
    )


@add_boilerplate("a")
def amax(a, axis=None, out=None, keepdims=False):
    """

    Return the maximum of an array or maximum along an axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple of ints, optional
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

    where : array_like of bool, optional
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
    GPU, CPU
    """
    return a.max(axis=axis, out=out, keepdims=keepdims)


max = amax


@add_boilerplate("a", "b")
def minimum(a, b, out=None, where=True, dtype=None, **kwargs):
    """
    Element-wise minimum of array elements.

    Compare two arrays and returns a new array containing the element-wise
    minima. If one of the elements being compared is a NaN, then that
    element is returned. If both elements are NaNs then the first is
    returned. The latter distinction is important for complex NaNs, which
    are defined as at least one of the real or imaginary parts being a NaN.
    The net effect is that NaNs are propagated.

    Parameters
    ----------
    x1, x2 : array_like
        The arrays holding the elements to be compared. If ``x1.shape !=
        x2.shape``, they must be broadcastable to a common shape (which becomes
        the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : ndarray or scalar
        The minimum of `x1` and `x2`, element-wise.
        This is a scalar if both `x1` and `x2` are scalars.

        See Also
        --------
        numpy.minimum

        Availability
        --------
        GPU, CPU
    """
    return ndarray.perform_binary_op(
        BinaryOpCode.MINIMUM,
        a,
        b,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
        where=where,
    )


@add_boilerplate("a")
def amin(a, axis=None, out=None, keepdims=False):
    """

    Return the minimum of an array or minimum along an axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple of ints, optional
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

    where : array_like of bool, optional
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
    GPU, CPU
    """
    return a.min(axis=axis, out=out, keepdims=keepdims)


min = amin

# Miscellaneous


@add_boilerplate("a", "v")
def convolve(a, v, mode="full"):
    """

    Returns the discrete, linear convolution of two one-dimensional sequences.

    If `v` is longer than `a`, the arrays are swapped before computation.

    Parameters
    ----------
    a : (N,) array_like
        First one-dimensional input array.
    v : (M,) array_like
        Second one-dimensional input array.
    mode : {'full', 'valid', 'same'}, optional
        'full':
          By default, mode is 'full'.  This returns the convolution
          at each point of overlap, with an output shape of (N+M-1,). At
          the end-points of the convolution, the signals do not overlap
          completely, and boundary effects may be seen.

        'same':
          Mode 'same' returns output of length ``max(M, N)``.  Boundary
          effects are still visible.

        'valid':
          Mode 'valid' returns output of length
          ``max(M, N) - min(M, N) + 1``.  The convolution product is only given
          for points where the signals overlap completely.  Values outside
          the signal boundary have no effect.

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

    Availability
    --------
    GPU, CPU
    """
    if mode != "same":
        raise NotImplementedError("Need to implement other convolution modes")

    if a.size < v.size:
        v, a = a, v

    return a.convolve(v, mode)


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
    GPU, CPU
    """
    return a.clip(a_min, a_max, out=out)


@add_boilerplate("a")
def sqrt(a, out=None, where=True, dtype=None, **kwargs):
    """
    Return the non-negative square-root of an array, element-wise.

    Parameters
    ----------
    x : array_like
        The values whose square-roots are required.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : ndarray
        An array of the same shape as `x`, containing the positive
        square-root of each element in `x`.  If any element in `x` is
        complex, a complex array is returned (and the square-roots of
        negative reals are calculated).  If all of the elements in `x`
        are real, so is `y`, with negative elements returning ``nan``.
        If `out` was provided, `y` is a reference to it.
        This is a scalar if `x` is a scalar.

        See Also
        --------
        numpy.sqrt

        Availability
        --------
        GPU, CPU
    """
    return ndarray.perform_unary_op(
        UnaryOpCode.SQRT,
        a,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=_output_float_dtype(a),
    )


def square(a, out=None, where=True, dtype=None, **kwargs):
    """
    Return the element-wise square of the input.

    Parameters
    ----------
    x : array_like
        Input data.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    out : ndarray or scalar
        Element-wise `x*x`, of the same shape and dtype as `x`.
        This is a scalar if `x` is a scalar.

        See Also
        --------
        numpy.square

        Availability
        --------
        GPU, CPU
    """
    return multiply(a, a, out=out, where=where, dtype=dtype)


@add_boilerplate("a")
def absolute(a, out=None, where=True, **kwargs):
    """
    Calculate the absolute value element-wise.

    Parameters
    ----------
    x : array_like
        Input array.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    absolute : ndarray
        An ndarray containing the absolute value of each element in `x`.  For
        complex input, ``a + ib``, the absolute value is :math:`\\sqrt{ a^2 +
        b^2 }`.  This is a scalar if `x` is a scalar.

        See Also
        --------
        numpy.absolute

        Availability
        --------
        GPU, CPU
    """
    # Handle the nice case of it being unsigned
    if a.dtype.type in (np.uint16, np.uint32, np.uint64, np.bool_):
        return a
    return ndarray.perform_unary_op(
        UnaryOpCode.ABSOLUTE, a, dst=out, where=where
    )


abs = absolute  # alias


def fabs(a, out=None, where=True, **kwargs):
    """
    Compute the absolute values element-wise.

    This function returns the absolute values (positive magnitude) of the
    data in `x`. Complex values are not handled, use `absolute` to find the
    absolute values of complex data.

    Parameters
    ----------
    x : array_like
        The array of numbers for which the absolute values are required. If
        `x` is a scalar, the result `y` will also be a scalar.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : ndarray or scalar
        The absolute values of `x`, the returned values are always floats.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    numpy.fabs

    Availability
    --------
    GPU, CPU
    """
    return absolute(a, out=out, where=where, **kwargs)


@add_boilerplate("a")
def sign(a, out=None, where=True, dtype=None, **kwargs):
    """
    Returns an element-wise indication of the sign of a number.

    The `sign` function returns ``-1 if x < 0, 0 if x==0, 1 if x > 0``.  nan
    is returned for nan inputs.

    For complex inputs, the `sign` function returns
    ``sign(x.real) + 0j if x.real != 0 else sign(x.imag) + 0j``.

    complex(nan, 0) is returned for complex nan inputs.

    Parameters
    ----------
    x : array_like
        Input values.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    y : ndarray
        The sign of `x`.
        This is a scalar if `x` is a scalar.

    See Also
    --------
    numpy.sign

    Availability
    --------
    GPU, CPU
    """
    return ndarray.perform_unary_op(
        UnaryOpCode.SIGN,
        a,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=_output_float_dtype(a),
    )


##################################
# Sorting, searching, and counting
##################################

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
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.

    Returns
    -------
    index_array : ndarray of ints
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.

    See Also
    --------
    numpy.argmax

    Availability
    --------
    GPU, CPU
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
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.

    Returns
    -------
    index_array : ndarray of ints
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.

    See Also
    --------
    numpy.argmin

    Availability
    --------
    GPU, CPU
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
    count : int or array of int
        Number of non-zero values in the array along a given axis.
        Otherwise, the total number of non-zero values in the array
        is returned.

    See Also
    --------
    numpy.count_nonzero

    Availability
    --------
    GPU, CPU
    """
    if a.size == 0:
        return 0
    return ndarray.perform_unary_reduction(
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
    axis : None or int or tuple of ints, optional
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
    m : ndarray, see dtype parameter above
        If `out=None`, returns a new array containing the mean values,
        otherwise a reference to the output array is returned.

    See Also
    --------
    numpy.mean

    Availability
    --------
    GPU, CPU
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
    x : array_like, 1 dimension, nonnegative ints
        Input array.
    weights : array_like, optional
        Weights, array of the same shape as `x`.
    minlength : int, optional
        A minimum number of bins for the output array.

    Returns
    -------
    out : ndarray of ints
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
    GPU, CPU
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
