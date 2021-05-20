# Copyright 2021 NVIDIA Corporation
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
import sys

import numpy as np

from .array import ndarray
from .config import BinaryOpCode, NumPyOpCode, UnaryOpCode
from .doc_utils import copy_docstring
from .runtime import runtime

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

try:
    import __builtin__ as builtins  # Python 2
except ModuleNotFoundError:
    import builtins as builtins  # Python 3

# ### ARRAY CREATION ROUTINES


@copy_docstring(np.arange)
def arange(*args, dtype=None, stacklevel=1):
    if len(args) == 1:
        (stop,) = args
        start = 0
        step = 1
    elif len(args) == 2:
        (
            start,
            stop,
        ) = args
        step = 1
    elif len(args) == 3:
        (start, stop, step) = args

    if dtype is None:
        dtype = np.array([stop]).dtype
    else:
        dtype = np.dtype(dtype)

    N = math.ceil((stop - start) / step)
    result = ndarray((N,), dtype, stacklevel=(stacklevel + 1))
    result._thunk.arange(start, stop, step, stacklevel=(stacklevel + 1))
    return result


@copy_docstring(np.array)
def array(obj, dtype=None, copy=True, order="K", subok=False, ndmin=0):
    if not isinstance(obj, ndarray):
        thunk = runtime.get_numpy_thunk(
            obj, stacklevel=2, share=(not copy), dtype=dtype
        )
        array = ndarray(shape=None, stacklevel=2, thunk=thunk)
    else:
        array = array
    if dtype is not None and array.dtype != dtype:
        array = array.astype(dtype)
    elif copy and obj is array:
        array = array.copy()
    if array.ndim < ndmin:
        shape = (np.newaxis,) * (ndmin - array.ndim) + array.shape
        array = array.reshape(shape)
    return array


@copy_docstring(np.diag)
def diag(v, k=0):
    array = ndarray.convert_to_legate_ndarray(v)
    if array.size == 1:
        return v
    elif array.ndim == 1:
        # Make a diagonal matrix from the array
        N = array.shape[0] + builtins.abs(k)
        matrix = ndarray((N, N), dtype=array.dtype)
        matrix._thunk.diag(array._thunk, extract=False, k=k, stacklevel=2)
        return matrix
    elif array.ndim == 2:
        # Extract the diagonal from the matrix
        # Solve for the size of the diagonal
        if k > 0:
            if k >= array.shape[1]:
                raise ValueError("'k' for diag must be in range")
            start = (0, k)
        elif k < 0:
            if -k >= array.shape[0]:
                raise ValueError("'k' for diag must be in range")
            start = (-k, 0)
        else:
            start = (0, 0)
        stop1 = (array.shape[0] - 1, array.shape[0] - 1 + k)
        stop2 = (array.shape[1] - 1 - k, array.shape[1] - 1)
        if stop1[0] < array.shape[0] and stop1[1] < array.shape[1]:
            distance = (stop1[0] - start[0]) + 1
            assert distance == ((stop1[1] - start[1]) + 1)
        else:
            assert stop2[0] < array.shape[0] and stop2[1] < array.shape[1]
            distance = (stop2[0] - start[0]) + 1
            assert distance == ((stop2[1] - start[1]) + 1)
        vector = ndarray(distance, dtype=array.dtype)
        vector._thunk.diag(array._thunk, extract=True, k=k, stacklevel=2)
        return vector
    elif array.ndim > 2:
        raise ValueError("diag requires 1- or 2-D array")


@copy_docstring(np.empty)
def empty(shape, dtype=np.float64, stacklevel=1):
    return ndarray(shape=shape, dtype=dtype, stacklevel=(stacklevel + 1))


@copy_docstring(np.empty_like)
def empty_like(a, dtype=None, stacklevel=1):
    array = ndarray.convert_to_legate_ndarray(a, stacklevel=(stacklevel + 1))
    shape = array.shape
    if dtype is not None:
        dtype = np.dtype(dtype)
    else:
        dtype = a.dtype
    return ndarray(
        shape, dtype=dtype, stacklevel=(stacklevel + 1), inputs=(array,)
    )


@copy_docstring(np.eye)
def eye(N, M=None, k=0, dtype=np.float64, stacklevel=1):
    if dtype is not None:
        dtype = np.dtype(dtype)
    if M is None:
        M = N
    result = ndarray((N, M), dtype, stacklevel=(stacklevel + 1))
    result._thunk.eye(k, stacklevel=(stacklevel + 1))
    return result


@copy_docstring(np.full)
def full(shape, value, dtype=None, stacklevel=1):
    if dtype is None:
        val = np.array(value)
    else:
        dtype = np.dtype(dtype)
        val = np.array(value, dtype=dtype)
    result = empty(shape, dtype=val.dtype, stacklevel=(stacklevel + 1))
    result._thunk.fill(val, stacklevel=(stacklevel + 1))
    return result


@copy_docstring(np.full_like)
def full_like(a, value, dtype=None, stacklevel=1):
    if dtype is not None:
        dtype = np.dtype(dtype)
    else:
        dtype = a.dtype
    result = empty_like(a, dtype=dtype, stacklevel=(stacklevel + 1))
    val = np.array(value, dtype=result.dtype)
    result._thunk.fill(val, stacklevel=(stacklevel + 1))
    return result


@copy_docstring(np.identity)
def identity(n, dtype=float):
    return eye(N=n, M=n, dtype=dtype, stacklevel=2)


@copy_docstring(np.linspace)
def linspace(
    start,
    stop,
    num=50,
    endpoint=True,
    retstep=False,
    dtype=None,
    axis=0,
    stacklevel=1,
):
    if num < 0:
        raise ValueError("Number of samples, %s, must be non-negative." % num)
    div = (num - 1) if endpoint else num

    start = ndarray.convert_to_legate_ndarray(start)
    stop = ndarray.convert_to_legate_ndarray(stop)

    dt = np.result_type(start, stop, float(num))
    if dtype is None:
        dtype = dt

    delta = stop - start
    y = arange(0, num, dtype=dt, stacklevel=stacklevel + 1)

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


@copy_docstring(np.ones)
def ones(shape, dtype=np.float64, stacklevel=1):
    return full(shape, 1, dtype=dtype, stacklevel=(stacklevel + 1))


@copy_docstring(np.ones_like)
def ones_like(a, dtype=None, stacklevel=1):
    usedtype = a.dtype
    if dtype is not None:
        usedtype = np.dtype(dtype)
    return full_like(a, 1, dtype=usedtype, stacklevel=(stacklevel + 1))


@copy_docstring(np.zeros)
def zeros(shape, dtype=np.float64, stacklevel=1):
    if dtype is not None:
        dtype = np.dtype(dtype)
    return full(shape, 0, dtype=dtype, stacklevel=(stacklevel + 1))


@copy_docstring(np.zeros_like)
def zeros_like(a, dtype=None, stacklevel=1):
    usedtype = a.dtype
    if dtype is not None:
        usedtype = np.dtype(dtype)
    return full_like(a, 0, dtype=usedtype, stacklevel=(stacklevel + 1))


@copy_docstring(np.copy)
def copy(a):
    array = ndarray.convert_to_legate_ndarray(a)
    result = empty_like(array, dtype=array.dtype, stacklevel=2)
    result._thunk.copy(array._thunk, deep=True, stacklevel=2)
    return result


# ### ARRAY MANIPULATION ROUTINES

# Changing array shape


@copy_docstring(np.ravel)
def ravel(a, order="C"):
    array = ndarray.convert_to_legate_ndarray(a)
    return array.ravel(order=order, stacklevel=2)


@copy_docstring(np.reshape)
def reshape(a, newshape, order="C"):
    array = ndarray.convert_to_legate_ndarray(a)
    return array.reshape(newshape, order=order, stacklevel=2)


@copy_docstring(np.transpose)
def transpose(a, axes=None):
    return a.transpose(axes=axes)


# Changing kind of array


@copy_docstring(np.asarray)
def asarray(a, dtype=None, order=None):
    if not isinstance(a, ndarray):
        thunk = runtime.get_numpy_thunk(
            a, stacklevel=2, share=True, dtype=dtype
        )
        array = ndarray(shape=None, stacklevel=2, thunk=thunk)
    else:
        array = a
    if dtype is not None and array.dtype != dtype:
        array = array.astype(dtype)
    return array


# Tiling


@copy_docstring(np.tile)
def tile(a, reps):
    array = ndarray.convert_to_legate_ndarray(a)
    if not hasattr(reps, "__len__"):
        reps = (reps,)
    # Figure out the shape of the destination array
    out_dims = array.ndim if array.ndim > len(reps) else len(reps)
    # Prepend ones until the dimensions match
    while len(reps) < out_dims:
        reps = (1,) + reps
    out_shape = ()
    # Prepend dimensions if necessary
    for dim in xrange(out_dims - array.ndim):
        out_shape += (reps[dim],)
    offset = len(out_shape)
    for dim in xrange(array.ndim):
        out_shape += (array.shape[dim] * reps[offset + dim],)
    assert len(out_shape) == out_dims
    result = ndarray(out_shape, dtype=array.dtype, inputs=(array,))
    result._thunk.tile(array._thunk, reps, stacklevel=2)
    return result


# ### BINARY OPERATIONS

# Elementwise bit operations


@copy_docstring(np.invert)
def invert(a, out=None, where=True, dtype=None):
    array = ndarray.convert_to_legate_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(out, share=True)
    if array.dtype.type == np.bool_:
        # Boolean values are special, just do negatiion
        return ndarray.perform_unary_op(
            UnaryOpCode.LOGICAL_NOT,
            array,
            dst=out,
            dtype=dtype,
            out_dtype=np.dtype(np.bool_),
            where=where,
        )
    else:
        return ndarray.perform_unary_op(
            UnaryOpCode.INVERT, array, dst=out, dtype=dtype, where=where
        )


# ### LINEAR ALGEBRA

# Matrix and vector products
@copy_docstring(np.dot)
def dot(a, b, out=None):
    a_array = ndarray.convert_to_legate_ndarray(a)
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(out, share=True)
    return a_array.dot(b, out=out, stacklevel=2)


# ### LOGIC FUNCTIONS


@copy_docstring(np.logical_not)
def logical_not(a, out=None, where=True, dtype=None, **kwargs):
    a_array = ndarray.convert_to_legate_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.LOGICAL_NOT,
        a_array,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=np.dtype(np.bool_),
    )


# truth value testing

# def all(a, axis = None, out = None, keepdims = False):
#    raise NotImplementedError("all")

# def any(a, axis = None, out = None, keepdims = False):
#    raise NotImplementedError("any")

# #Comparison


@copy_docstring(np.allclose)
def allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
    a_array = ndarray.convert_to_legate_ndarray(a)
    b_array = ndarray.convert_to_legate_ndarray(b)
    if equal_nan:
        raise NotImplementedError(
            "Legate does not support equal NaN yet for allclose"
        )
    args = (np.array(rtol, dtype=np.float64), np.array(atol, dtype=np.float64))
    return ndarray.perform_binary_reduction(
        BinaryOpCode.ALLCLOSE,
        a_array,
        b_array,
        dtype=np.dtype(np.bool),
        args=args,
    )


@copy_docstring(np.array_equal)
def array_equal(a, b):
    a_array = ndarray.convert_to_legate_ndarray(a)
    b_array = ndarray.convert_to_legate_ndarray(b)
    if a_array.ndim != b_array.ndim:
        return False
    if a_array.shape != b_array.shape:
        return False
    return ndarray.perform_binary_reduction(
        BinaryOpCode.EQUAL, a_array, b_array, dtype=np.dtype(np.bool_)
    )


@copy_docstring(np.equal)
def equal(a, b, out=None, where=True, dtype=np.dtype(np.bool), stacklevel=1):
    a_array = ndarray.convert_to_legate_ndarray(a, stacklevel=(stacklevel + 1))
    b_array = ndarray.convert_to_legate_ndarray(b, stacklevel=(stacklevel + 1))
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(
            out, stacklevel=(stacklevel + 1), share=True
        )
    return ndarray.perform_binary_op(
        BinaryOpCode.EQUAL,
        a_array,
        b_array,
        out=out,
        out_dtype=dtype,
        where=where,
        stacklevel=(stacklevel + 1),
    )


@copy_docstring(np.greater)
def greater(a, b, out=None, where=True, dtype=np.dtype(np.bool), stacklevel=1):
    a_array = ndarray.convert_to_legate_ndarray(a, stacklevel=(stacklevel + 1))
    b_array = ndarray.convert_to_legate_ndarray(b, stacklevel=(stacklevel + 1))
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(
            out, stacklevel=(stacklevel + 1), share=True
        )
    return ndarray.perform_binary_op(
        BinaryOpCode.GREATER,
        a_array,
        b_array,
        out=out,
        out_dtype=dtype,
        where=where,
        stacklevel=(stacklevel + 1),
    )


@copy_docstring(np.greater_equal)
def greater_equal(
    a, b, out=None, where=True, dtype=np.dtype(np.bool), stacklevel=1
):
    a_array = ndarray.convert_to_legate_ndarray(a, stacklevel=(stacklevel + 1))
    b_array = ndarray.convert_to_legate_ndarray(b, stacklevel=(stacklevel + 1))
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(
            out, stacklevel=(stacklevel + 1), share=True
        )
    return ndarray.perform_binary_op(
        BinaryOpCode.GREATER_EQUAL,
        a_array,
        b_array,
        out=out,
        out_dtype=dtype,
        where=where,
        stacklevel=(stacklevel + 1),
    )


@copy_docstring(np.isinf)
def isinf(a, out=None, where=True, dtype=None, **kwargs):
    a_array = ndarray.convert_to_legate_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.ISINF,
        a_array,
        dst=out,
        dtype=dtype,
        where=where,
        out_dtype=np.dtype(np.bool_),
    )


@copy_docstring(np.isnan)
def isnan(a, out=None, where=True, dtype=None, **kwargs):
    a_array = ndarray.convert_to_legate_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.ISNAN,
        a_array,
        dst=out,
        dtype=dtype,
        where=where,
        out_dtype=np.dtype(np.bool_),
    )


@copy_docstring(np.less)
def less(a, b, out=None, where=True, dtype=np.dtype(np.bool), stacklevel=1):
    a_array = ndarray.convert_to_legate_ndarray(a, stacklevel=(stacklevel + 1))
    b_array = ndarray.convert_to_legate_ndarray(b, stacklevel=(stacklevel + 1))
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(
            out, stacklevel=(stacklevel + 1), share=True
        )
    return ndarray.perform_binary_op(
        BinaryOpCode.LESS,
        a_array,
        b_array,
        out=out,
        out_dtype=dtype,
        where=where,
        stacklevel=(stacklevel + 1),
    )


@copy_docstring(np.less_equal)
def less_equal(
    a, b, out=None, where=True, dtype=np.dtype(np.bool), stacklevel=1
):
    a_array = ndarray.convert_to_legate_ndarray(a, stacklevel=(stacklevel + 1))
    b_array = ndarray.convert_to_legate_ndarray(b, stacklevel=(stacklevel + 1))
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(
            out, stacklevel=(stacklevel + 1), share=True
        )
    return ndarray.perform_binary_op(
        BinaryOpCode.LESS_EQUAL,
        a_array,
        b_array,
        out=out,
        out_dtype=dtype,
        where=where,
        stacklevel=(stacklevel + 1),
    )


@copy_docstring(np.not_equal)
def not_equal(
    a, b, out=None, where=True, dtype=np.dtype(np.bool), stacklevel=1
):
    a_array = ndarray.convert_to_legate_ndarray(a, stacklevel=(stacklevel + 1))
    b_array = ndarray.convert_to_legate_ndarray(b, stacklevel=(stacklevel + 1))
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(
            out, stacklevel=(stacklevel + 1), share=True
        )
    return ndarray.perform_binary_op(
        BinaryOpCode.NOT_EQUAL,
        a_array,
        b_array,
        out=out,
        out_dtype=dtype,
        where=where,
        stacklevel=(stacklevel + 1),
    )


# ### MATHEMATICAL FUNCTIONS


@copy_docstring(np.negative)
def negative(a, out=None, where=True, dtype=None, **kwargs):
    a_array = ndarray.convert_to_legate_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    if (
        a_array.dtype.type == np.uint16
        or a_array.dtype.type == np.uint32
        or a_array.dtype.type == np.uint64
    ):
        raise TypeError("cannot negate unsigned type " + str(a_array.dtype))
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.NEGATIVE, a_array, dtype=dtype, dst=out, where=where
    )


# Trigonometric functions


@copy_docstring(np.arccos)
def arccos(a, out=None, where=True, dtype=None, **kwargs):
    lg_array = ndarray.convert_to_legate_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    # Floats keep their floating point kind, otherwise switch to float64
    if lg_array.dtype.kind == "f" or lg_array.dtype.kind == "c":
        out_dtype = lg_array.dtype
    else:
        out_dtype = np.dtype(np.float64)
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.ARCCOS,
        lg_array,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=out_dtype,
    )


@copy_docstring(np.arcsin)
def arcsin(a, out=None, where=True, dtype=None, **kwargs):
    lg_array = ndarray.convert_to_legate_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    # Floats keep their floating point kind, otherwise switch to float64
    if lg_array.dtype.kind == "f" or lg_array.dtype.kind == "c":
        out_dtype = lg_array.dtype
    else:
        out_dtype = np.dtype(np.float64)
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.ARCSIN,
        lg_array,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=out_dtype,
    )


@copy_docstring(np.arctan)
def arctan(a, out=None, where=True, dtype=None, **kwargs):
    lg_array = ndarray.convert_to_legate_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    # Floats keep their floating point kind, otherwise switch to float64
    if lg_array.dtype.kind == "f" or lg_array.dtype.kind == "c":
        out_dtype = lg_array.dtype
    else:
        out_dtype = np.dtype(np.float64)
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.ARCTAN,
        lg_array,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=out_dtype,
    )


@copy_docstring(np.cos)
def cos(a, out=None, where=True, dtype=None, **kwargs):
    lg_array = ndarray.convert_to_legate_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    # Floats keep their floating point kind, otherwise switch to float64
    if lg_array.dtype.kind == "f" or lg_array.dtype.kind == "c":
        out_dtype = lg_array.dtype
    else:
        out_dtype = np.dtype(np.float64)
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.COS,
        lg_array,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=out_dtype,
    )


@copy_docstring(np.sin)
def sin(a, out=None, where=True, dtype=None, **kwargs):
    lg_array = ndarray.convert_to_legate_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    # Floats keep their floating point kind, otherwise switch to float64
    if lg_array.dtype.kind == "f" or lg_array.dtype.kind == "c":
        out_dtype = lg_array.dtype
    else:
        out_dtype = np.dtype(np.float64)
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.SIN,
        lg_array,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=out_dtype,
    )


@copy_docstring(np.tan)
def tan(a, out=None, where=True, dtype=None, **kwargs):
    lg_array = ndarray.convert_to_legate_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    # Floats keep their floating point kind, otherwise switch to float64
    if lg_array.dtype.kind == "f" or lg_array.dtype.kind == "c":
        out_dtype = lg_array.dtype
    else:
        out_dtype = np.dtype(np.float64)
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.TAN,
        lg_array,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=out_dtype,
    )


# Hyperbolic functions


@copy_docstring(np.tanh)
def tanh(a, out=None, where=True, dtype=None, **kwargs):
    lg_array = ndarray.convert_to_legate_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    # Floats keep their floating point kind, otherwise switch to float64
    if lg_array.dtype.kind == "f" or lg_array.dtype.kind == "c":
        out_dtype = lg_array.dtype
    else:
        out_dtype = np.dtype(np.float64)
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.TANH,
        lg_array,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=out_dtype,
    )


# Sums, products, differences


@copy_docstring(np.add)
def add(a, b, out=None, where=True, dtype=None, stacklevel=1):
    a_array = ndarray.convert_to_legate_ndarray(a, stacklevel=(stacklevel + 1))
    b_array = ndarray.convert_to_legate_ndarray(b, stacklevel=(stacklevel + 1))
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(
            out, stacklevel=(stacklevel + 1), share=True
        )
    return ndarray.perform_binary_op(
        BinaryOpCode.ADD,
        a_array,
        b_array,
        out=out,
        out_dtype=dtype,
        where=where,
        stacklevel=(stacklevel + 1),
    )


@copy_docstring(np.divide)
def divide(a, b, out=None, where=True, dtype=None):
    # For python 3 switch this to truedivide
    if sys.version_info > (3,):
        return true_divide(
            a, b, out=out, where=where, dtype=dtype, stacklevel=2
        )
    a_array = ndarray.convert_to_legate_ndarray(a)
    b_array = ndarray.convert_to_legate_ndarray(b)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(out, share=True)
    return ndarray.perform_binary_op(
        BinaryOpCode.DIVIDE,
        a_array,
        b_array,
        out=out,
        out_dtype=dtype,
        where=where,
    )


@copy_docstring(np.floor_divide)
def floor_divide(a, b, out=None, where=True, dtype=None):
    a_array = ndarray.convert_to_legate_ndarray(a)
    b_array = ndarray.convert_to_legate_ndarray(b)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(out, share=True)
    return ndarray.perform_binary_op(
        BinaryOpCode.FLOOR_DIVIDE,
        a_array,
        b_array,
        out=out,
        out_dtype=dtype,
        where=where,
    )


@copy_docstring(np.multiply)
def multiply(a, b, out=None, where=True, dtype=None, stacklevel=1):
    a_array = ndarray.convert_to_legate_ndarray(a, stacklevel=(stacklevel + 1))
    b_array = ndarray.convert_to_legate_ndarray(b, stacklevel=(stacklevel + 1))
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(
            out, stacklevel=(stacklevel + 1), share=True
        )
    return ndarray.perform_binary_op(
        BinaryOpCode.MULTIPLY,
        a_array,
        b_array,
        out=out,
        out_dtype=dtype,
        where=where,
        stacklevel=(stacklevel + 1),
    )


@copy_docstring(np.prod)
def prod(
    a,
    axis=None,
    dtype=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
    stacklevel=1,
):
    lg_array = ndarray.convert_to_legate_ndarray(
        a, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(
            out, stacklevel=(stacklevel + 1), share=True
        )
    return lg_array.prod(
        axis=axis,
        dtype=dtype,
        out=out,
        keepdims=keepdims,
        initial=initial,
        where=where,
        stacklevel=(stacklevel + 1),
    )


@copy_docstring(np.subtract)
def subtract(a, b, out=None, where=True, dtype=None, stacklevel=1):
    a_array = ndarray.convert_to_legate_ndarray(a, stacklevel=(stacklevel + 1))
    b_array = ndarray.convert_to_legate_ndarray(b, stacklevel=(stacklevel + 1))
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(
            out, stacklevel=(stacklevel + 1), share=True
        )
    return ndarray.perform_binary_op(
        BinaryOpCode.SUBTRACT,
        a_array,
        b_array,
        out=out,
        out_dtype=dtype,
        where=where,
        stacklevel=(stacklevel + 1),
    )


@copy_docstring(np.sum)
def sum(
    a,
    axis=None,
    dtype=None,
    out=None,
    keepdims=False,
    initial=None,
    where=True,
    stacklevel=1,
):
    lg_array = ndarray.convert_to_legate_ndarray(
        a, stacklevel=(stacklevel + 1)
    )
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(
            out, stacklevel=(stacklevel + 1), share=True
        )
    return lg_array.sum(
        axis=axis,
        dtype=dtype,
        out=out,
        keepdims=keepdims,
        initial=initial,
        where=where,
        stacklevel=(stacklevel + 1),
    )


@copy_docstring(np.true_divide)
def true_divide(a, b, out=None, where=True, dtype=None, stacklevel=1):
    a_array = ndarray.convert_to_legate_ndarray(a, stacklevel=(stacklevel + 1))
    b_array = ndarray.convert_to_legate_ndarray(b, stacklevel=(stacklevel + 1))
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    # Convert any non-floats to floating point arrays
    if a_array.dtype.kind != "f":
        a_type = np.dtype(np.float64)
    else:
        a_type = a_array.dtype
    if b_array.dtype.kind != "f":
        b_type = np.dtype(np.float64)
    else:
        b_type = b_array.dtype
    # If the types don't match then align them
    if a_type != b_type:
        array_types = list()
        scalar_types = list()
        if a_array.ndim > 0:
            array_types.append(a_type)
        else:
            scalar_types.append(a_type)
        if b_array.ndim > 0:
            array_types.append(b_type)
        else:
            scalar_types.append(b_type)
        common_type = np.find_common_type(array_types, scalar_types)
    else:
        common_type = a_type
    if a_array.dtype != common_type:
        temp = ndarray(
            a_array.shape,
            dtype=common_type,
            stacklevel=(stacklevel + 1),
            inputs=(a_array, b_array),
        )
        temp._thunk.convert(
            a_array._thunk, warn=False, stacklevel=(stacklevel + 1)
        )
        a_array = temp
    if b_array.dtype != common_type:
        temp = ndarray(
            b_array.shape,
            dtype=common_type,
            stacklevel=(stacklevel + 1),
            inputs=(a_array, b_array),
        )
        temp._thunk.convert(
            b_array._thunk, warn=False, stacklevel=(stacklevel + 1)
        )
        b_array = temp
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(
            out, stacklevel=(stacklevel + 1), share=True
        )
    return ndarray.perform_binary_op(
        BinaryOpCode.DIVIDE,
        a_array,
        b_array,
        out=out,
        out_dtype=dtype,
        where=where,
        stacklevel=(stacklevel + 1),
    )


# Exponents and logarithms


@copy_docstring(np.exp)
def exp(a, out=None, where=True, dtype=None, stacklevel=1, **kwargs):
    lg_array = ndarray.convert_to_legate_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if lg_array.dtype.kind == "f" or lg_array.dtype.kind == "c":
        out_dtype = lg_array.dtype
    else:
        out_dtype = np.dtype(np.float64)
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.EXP,
        lg_array,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=out_dtype,
    )


@copy_docstring(np.log)
def log(a, out=None, where=True, dtype=None, stacklevel=1, **kwargs):
    lg_array = ndarray.convert_to_legate_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if lg_array.dtype.kind == "f" or lg_array.dtype.kind == "c":
        out_dtype = lg_array.dtype
    else:
        out_dtype = np.dtype(np.float64)
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.LOG,
        lg_array,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=out_dtype,
    )


@copy_docstring(np.power)
def power(x1, x2, out=None, where=True, dtype=None, stacklevel=1, **kwargs):
    x1_array = ndarray.convert_to_legate_ndarray(
        x1, stacklevel=(stacklevel + 1)
    )
    x2_array = ndarray.convert_to_legate_ndarray(
        x2, stacklevel=(stacklevel + 1)
    )
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is None and dtype is None:
        if x1_array.dtype.kind == "f" or x2_array.dtype.kind == "f":
            array_types = list()
            scalar_types = list()
            if x1_array.ndim > 0:
                array_types.append(x1_array.dtype)
            else:
                scalar_types.append(x1_array.dtype)
            if x2_array.ndim > 0:
                array_types.append(x2_array.dtype)
            else:
                scalar_types.append(x2_array.dtype)
            dtype = np.find_common_type(array_types, scalar_types)
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(
            out, stacklevel=(stacklevel + 1), share=True
        )
    return ndarray.perform_binary_op(
        BinaryOpCode.POWER,
        x1_array,
        x2_array,
        out=out,
        out_dtype=dtype,
        where=where,
        stacklevel=(stacklevel + 1),
    )


@copy_docstring(np.square)
def square(a, out=None, where=True, dtype=None, **kwargs):
    lg_array = ndarray.convert_to_legate_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(out, share=True)
    # We implement this with multiply for now, locality should
    # be good enough for avoiding too much overhead with extra reads
    return ndarray.perform_binary_op(
        BinaryOpCode.MULTIPLY,
        lg_array,
        lg_array,
        out=out,
        out_dtype=dtype,
        where=where,
    )


# Miscellaneous


@copy_docstring(np.absolute)
def absolute(a, out=None, where=True, **kwargs):
    lg_array = ndarray.convert_to_legate_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    # Handle the nice case of it being unsigned
    if (
        lg_array.dtype.type == np.uint16
        or lg_array.dtype.type == np.uint32
        or lg_array.dtype.type == np.uint64
        or lg_array.dtype.type == np.bool_
    ):
        return lg_array
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.ABSOLUTE, lg_array, dst=out, where=where
    )


abs = absolute  # alias


@copy_docstring(np.ceil)
def ceil(a, out=None, where=True, dtype=None, **kwargs):
    lg_array = ndarray.convert_to_legate_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    # If this is an integer array then there is nothing to do for ceil
    if (
        lg_array.dtype.kind == "i"
        or lg_array.dtype.kind == "u"
        or lg_array.dtype.kind == "b"
    ):
        return lg_array
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.CEIL, lg_array, dst=out, dtype=dtype, where=where
    )


@copy_docstring(np.clip)
def clip(a, a_min, a_max, out=None):
    lg_array = ndarray.convert_to_legate_ndarray(a)
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(out, share=True)
    return lg_array.clip(a_min, a_max, out=out)


@copy_docstring(np.fabs)
def fabs(a, out=None, where=True, **kwargs):
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(out, share=True)
    return absolute(a, out=out, where=where, **kwargs)


@copy_docstring(np.floor)
def floor(a, out=None, where=True, dtype=None, **kwargs):
    lg_array = ndarray.convert_to_legate_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    # If this is an integer array then there is nothing to do for floor
    if (
        lg_array.dtype.kind == "i"
        or lg_array.dtype.kind == "u"
        or lg_array.dtype.kind == "b"
    ):
        return lg_array
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.FLOOR, lg_array, dst=out, dtype=dtype, where=where
    )


@copy_docstring(np.sqrt)
def sqrt(a, out=None, where=True, dtype=None, stacklevel=1, **kwargs):
    lg_array = ndarray.convert_to_legate_ndarray(
        a, stacklevel=(stacklevel + 1)
    )
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    # Floats keep their floating point kind, otherwise switch to float64
    if lg_array.dtype.kind == "f" or lg_array.dtype.kind == "c":
        out_dtype = lg_array.dtype
    else:
        out_dtype = np.dtype(np.float64)
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(
            out, stacklevel=(stacklevel + 1), share=True
        )
    return ndarray.perform_unary_op(
        UnaryOpCode.SQRT,
        lg_array,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=out_dtype,
        stacklevel=(stacklevel + 1),
    )


# ### SORTING, SEARCHING and COUNTING

# Searching


@copy_docstring(np.argmax)
def argmax(a, axis=None, out=None):
    lg_array = ndarray.convert_to_legate_ndarray(a)
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(out, share=True)
    return lg_array.argmax(axis=axis, out=out, stacklevel=2)


@copy_docstring(np.argmin)
def argmin(a, axis=None, out=None):
    lg_array = ndarray.convert_to_legate_ndarray(a)
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(out, share=True)
    return lg_array.argmin(axis=axis, out=out, stacklevel=2)


@copy_docstring(np.bincount)
def bincount(a, weights=None, minlength=0):
    lg_array = ndarray.convert_to_legate_ndarray(a)
    if weights is not None:
        lg_weights = ndarray.convert_to_legate_ndarray(weights)
        if lg_weights.shape != lg_array.shape:
            raise ValueError("weights array must be same shape for bincount")
    if lg_array.dtype.kind != "i" and lg_array.dtype.kind != "u":
        raise TypeError("input array for bincount must be integer type")
    # If nobody told us the size then compute it
    if minlength <= 0:
        minlength = int(amax(lg_array)) + 1
    if lg_array.size == 1:
        # Handle the special case of 0-D array
        if weights is None:
            out = zeros((minlength,), dtype=np.dtype(np.uint64), stacklevel=2)
            out[lg_array[0]] = 1
        else:
            out = zeros((minlength,), dtype=lg_weights.dtype, stacklevel=2)
            index = lg_array[0]
            out[index] = weights[index]
    else:
        # Normal case of bincount
        if weights is None:
            out = ndarray(
                (minlength,),
                dtype=np.dtype(np.uint64),
                inputs=(lg_array, weights),
            )
            out._thunk.bincount(lg_array._thunk, stacklevel=2)
        else:
            out = ndarray(
                (minlength,),
                dtype=lg_weights.dtype,
                inputs=(lg_array, weights),
            )
            out._thunk.bincount(
                lg_array._thunk, stacklevel=2, weights=lg_weights._thunk
            )
    return out


@copy_docstring(np.nonzero)
def nonzero(a):
    lg_array = ndarray.convert_to_legate_ndarray(a)
    return lg_array.nonzero()


@copy_docstring(np.where)
def where(a, x=None, y=None):
    if x is None or y is None:
        if x is not None or y is not None:
            raise ValueError(
                "both 'x' and 'y' parameters must be specified together for"
                " 'where'"
            )
        return nonzero(a)
    lg_array = ndarray.convert_to_legate_ndarray(a)
    x_array = ndarray.convert_to_legate_ndarray(x)
    y_array = ndarray.convert_to_legate_ndarray(y)
    # Check all the array types here
    if lg_array.dtype.type != np.bool_:
        temp = ndarray(
            shape=lg_array.shape,
            dtype=np.dtype(np.bool_),
            inputs=(lg_array, x_array, y_array),
        )
        temp._thunk.convert(lg_array._thunk, stacklevel=2)
        lg_array = temp
    if x_array.dtype != y_array.dtype:
        array_types = list()
        scalar_types = list()
        if x_array.size == 1:
            scalar_types.append(x_array.dtype)
        else:
            array_types.append(x_array.dtype)
        if y_array.size == 1:
            scalar_types.append(y_array.dtype)
        else:
            array_types.append(y_array.dtype)
        common_type = np.find_common_type(array_types, scalar_types)
        if x_array.dtype != common_type:
            temp = ndarray(
                shape=x_array.shape,
                dtype=common_type,
                inputs=(lg_array, x_array, y_array),
            )
            temp._thunk.convert(x_array._thunk, stacklevel=2)
            x_array = temp
        if y_array.dtype != common_type:
            temp = ndarray(
                shape=y_array.shape,
                dtype=common_type,
                inputs=(lg_array, x_array, y_array),
            )
            temp._thunk.convert(y_array._thunk, stacklevel=2)
            y_array = temp
    else:
        common_type = x_array.dtype
    return ndarray.perform_ternary_op(
        NumPyOpCode.WHERE,
        lg_array,
        x_array,
        y_array,
        out_dtype=common_type,
        check_types=False,
    )


# def extract(a, x):
#    raise NotImplementedError("extract")

# def sort(a, axis=-1, kind='quicksort', order=None):
#    # No need to sort anything with just one element
#    if a.size == 1:
#        raise NotImplementedError("Need to drop dimensions here")
#        return a
#    lg_array = ndarray.convert_to_legate_ndarray(a)
#    if order is not None:
#        raise NotImplementedError("No support for non-None 'order' for sort")
#    if kind != 'quicksort':
#        warnings.warn("Legate uses a different algorithm than "+str(kind)+
#                      " for sorting",
#                      category=RuntimeWarning, stacklevel=2)
#    # Make the out array
#    if lg_array.ndim > 1:
#        # We only support sorting the full array right now
#        if axis is not None:
#            raise NotImplementedError("Legate only supports sorting full "
#                                      "arrays at the moment")
#        # Flatten the output array
#        out_size = lg_array.shape[0]
#        for d in xrange(1,lg_array.ndim):
#            out_size *= lg_array.shape[d]
#        out = ndarray((out_size,), dtype=lg_array.dtype)
#    else:
#        out = ndarray(lg_array.shape, dtype=lg_array.dtype)
#    out._thunk.sort(lg_array._thunk, stacklevel=2)
#    return out

# Counting


@copy_docstring(np.count_nonzero)
def count_nonzero(a, axis=None):
    lg_array = ndarray.convert_to_legate_ndarray(a, stacklevel=2)
    return lg_array.count_nonzero(axis=axis, stacklevel=2)


# ### STATISTICS

# Order statistics


@copy_docstring(np.amax)
def amax(a, axis=None, out=None, keepdims=False, stacklevel=1):
    lg_array = ndarray.convert_to_legate_ndarray(
        a, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(
            out, stacklevel=(stacklevel + 1), share=True
        )
    return lg_array.max(
        axis=axis, out=out, keepdims=keepdims, stacklevel=(stacklevel + 1)
    )


@copy_docstring(np.amin)
def amin(a, axis=None, out=None, keepdims=False, stacklevel=1):
    lg_array = ndarray.convert_to_legate_ndarray(
        a, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(
            out, stacklevel=(stacklevel + 1), share=True
        )
    return lg_array.min(
        axis=axis, out=out, keepdims=keepdims, stacklevel=(stacklevel + 1)
    )


@copy_docstring(np.max)
def max(a, axis=None, out=None, keepdims=False):
    return amax(a, axis=axis, out=out, keepdims=keepdims, stacklevel=2)


@copy_docstring(np.maximum)
def maximum(a, b, out=None, where=True, dtype=None, stacklevel=1, **kwargs):
    a_array = ndarray.convert_to_legate_ndarray(a, stacklevel=(stacklevel + 1))
    b_array = ndarray.convert_to_legate_ndarray(b, stacklevel=(stacklevel + 1))
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(
            out, stacklevel=(stacklevel + 1), share=True
        )
    return ndarray.perform_binary_op(
        BinaryOpCode.MAXIMUM,
        a_array,
        b_array,
        out=out,
        out_dtype=dtype,
        where=where,
        stacklevel=(stacklevel + 1),
    )


@copy_docstring(np.min)
def min(a, axis=None, out=None, keepdims=False):
    return amin(a, axis=axis, out=out, keepdims=keepdims, stacklevel=2)


@copy_docstring(np.minimum)
def minimum(a, b, out=None, where=True, dtype=None, stacklevel=1, **kwargs):
    a_array = ndarray.convert_to_legate_ndarray(a, stacklevel=(stacklevel + 1))
    b_array = ndarray.convert_to_legate_ndarray(b, stacklevel=(stacklevel + 1))
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(
            out, stacklevel=(stacklevel + 1), share=True
        )
    return ndarray.perform_binary_op(
        BinaryOpCode.MINIMUM,
        a_array,
        b_array,
        out=out,
        out_dtype=dtype,
        where=where,
        stacklevel=(stacklevel + 1),
    )


@copy_docstring(np.add)
def mod(a, b, out=None, where=True, dtype=None, stacklevel=1):
    a_array = ndarray.convert_to_legate_ndarray(a, stacklevel=(stacklevel + 1))
    b_array = ndarray.convert_to_legate_ndarray(b, stacklevel=(stacklevel + 1))
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(
            out, stacklevel=(stacklevel + 1), share=True
        )
    return ndarray.perform_binary_op(
        BinaryOpCode.MOD,
        a_array,
        b_array,
        out=out,
        out_dtype=dtype,
        where=where,
        stacklevel=(stacklevel + 1),
    )


# Averages and variances


@copy_docstring(np.mean)
def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    lg_array = ndarray.convert_to_legate_ndarray(a)
    if out is not None:
        out = ndarray.convert_to_legate_ndarray(out, share=True)
    return lg_array.mean(
        axis=axis, dtype=dtype, out=out, keepdims=keepdims, stacklevel=2
    )


# ### STACKING and CONCATENATION ###


@copy_docstring(np.row_stack)
def row_stack(inputs):
    return vstack(inputs)


@copy_docstring(np.vstack)
def vstack(inputs):
    # Check to see if we can build a new tuple of legate arrays
    dtype = None
    legate_inputs = list()
    for inp in inputs:
        lg_array = ndarray.convert_to_legate_ndarray(inp)
        if dtype is None:
            ndim = lg_array.ndim
            shape = lg_array.shape
            dtype = lg_array.dtype
            if lg_array.ndim == 1:
                leading_dim = 1
            else:
                leading_dim = lg_array.shape[0]
        else:
            # Check that the types and shapes match
            if lg_array.ndim != ndim:
                raise TypeError(
                    "All arguments to vstack must have the same number of"
                    " dimensions"
                )
            if ndim > 1:
                for dim in xrange(1, ndim):
                    if shape[dim] != lg_array.shape[dim]:
                        raise TypeError(
                            "All arguments to vstack must have the same "
                            "dimension size in all dimensions except the first"
                        )
            if lg_array.dtype != dtype:
                raise TypeError(
                    "All arguments to vstack must have the same type"
                )
            if lg_array.ndim == 1:
                leading_dim += 1
            else:
                leading_dim += lg_array.shape[0]
        # Save the input array for later
        legate_inputs.append(lg_array)
    # Once we are here we have all our inputs arrays, so make the output
    if len(shape) == 1:
        out_shape = (leading_dim,) + shape
        out_array = ndarray(shape=out_shape, dtype=dtype, inputs=legate_inputs)
        # Copy the values over from the inputs
        for idx, inp in enumerate(legate_inputs):
            out_array[idx, :] = inp
    else:
        out_shape = (leading_dim,)
        for dim in xrange(1, ndim):
            out_shape += (shape[dim],)
        out_array = ndarray(shape=out_shape, dtype=dtype)
        # Copy the values over from the inputs
        offset = 0
        for inp in legate_inputs:
            out_array[offset : offset + inp.shape[0], ...] = inp
            offset += inp.shape[0]
    return out_array


# ### FILE I/O ###
# For now we just need to do the loading methods to get things converted into
# Legate arrays. The storing methods can go through the normal NumPy python


@copy_docstring(np.genfromtxt)
def genfromtxt(
    fname,
    dtype=float,
    comments="#",
    delimiter=None,
    skip_header=0,
    skip_footer=0,
    converters=None,
    missing_values=None,
    filling_values=None,
    usecols=None,
    names=None,
    excludelist=None,
    deletechars=" !#$%&'()*+, -./:;<=>?@[\\]^{|}~",
    replace_space="_",
    autostrip=False,
    case_sensitive=True,
    defaultfmt="f%i",
    unpack=None,
    usemask=False,
    loose=True,
    invalid_raise=True,
    max_rows=None,
    encoding="bytes",
):
    numpy_array = np.getfromtxt(
        fname,
        dtype=dtype,
        comments=comments,
        delimiter=delimiter,
        skip_header=skip_header,
        skip_footer=skip_footer,
        converters=converters,
        missing_values=missing_values,
        filling_values=filling_values,
        usecols=usecols,
        names=names,
        excludelist=excludelist,
        deletechars=deletechars,
        replace_space=replace_space,
        autostrip=autostrip,
        case_sensitive=case_sensitive,
        defaultfmt=defaultfmt,
        unpack=unpack,
        usemask=usemask,
        loose=loose,
        invalid_raise=invalid_raise,
        max_rows=max_rows,
        encoding=encoding,
    )
    return ndarray.convert_to_legate_ndarray(numpy_array)


@copy_docstring(np.frombuffer)
def frombuffer(buffer, dtype=float, count=-1, offset=0):
    numpy_array = np.frombuffer(
        buffer=buffer, dtype=dtype, count=count, offset=offset
    )
    return ndarray.convert_to_legate_ndarray(numpy_array)


@copy_docstring(np.fromfile)
def fromfile(file, dtype=float, count=-1, sep="", offset=0):
    numpy_array = np.fromfile(
        file=file, dtype=dtype, count=count, sep=sep, offset=offset
    )
    return ndarray.convert_to_legate_ndarray(numpy_array)


@copy_docstring(np.fromfunction)
def fromfunction(function, shape, **kwargs):
    numpy_array = np.fromfunction(function=function, shape=shape, **kwargs)
    return ndarray.convert_to_legate_ndarray(numpy_array)


@copy_docstring(np.fromiter)
def fromiter(iterable, dtype, count=-1):
    numpy_array = np.fromiter(iterable=iterable, dtype=dtype, count=count)
    return ndarray.convert_to_legate_ndarray(numpy_array)


@copy_docstring(np.fromregex)
def fromregex(file, regexp, dtype, encoding=None):
    numpy_array = np.fromregex(
        file=file, regexp=regexp, dtype=dtype, encoding=encoding
    )
    return ndarray.convert_to_legate_ndarray(numpy_array)


@copy_docstring(np.fromstring)
def fromstring(string, dtype=float, count=-1, sep=""):
    numpy_array = np.fromstring(
        string=string, dtype=dtype, count=count, sep=sep
    )
    return ndarray.convert_to_legate_ndarray(numpy_array)


@copy_docstring(np.load)
def load(
    file,
    mmap_mode=None,
    allow_pickle=False,
    fix_imports=True,
    encoding="ASCII",
):
    numpy_array = np.load(
        file,
        mmap_mode=mmap_mode,
        allow_pickle=allow_pickle,
        fix_imports=fix_imports,
        encoding=encoding,
    )
    return ndarray.convert_to_legate_ndarray(numpy_array)


@copy_docstring(np.loadtxt)
def loadtxt(
    fname,
    dtype=float,
    comments="#",
    delimiter=None,
    converters=None,
    skiprows=0,
    usecols=None,
    unpack=False,
    ndmin=0,
    encoding="bytes",
    max_rows=None,
):
    numpy_array = np.loadtxt(
        fname,
        dtype=dtype,
        comments=comments,
        delimiter=delimiter,
        converters=converters,
        skiprows=skiprows,
        usecols=usecols,
        unpack=unpack,
        ndmin=ndmin,
        encoding=encoding,
        max_rows=max_rows,
    )
    return ndarray.convert_to_legate_ndarray(numpy_array)


@copy_docstring(np.memmap)
def memmap(
    filename, dtype=np.uint8, mode="r+", offset=0, shape=None, order="C"
):
    numpy_array = np.memmap(
        filename,
        dtype=dtype,
        mode=mode,
        offset=offset,
        shape=shape,
        order=order,
    )
    return ndarray.convert_to_legate_ndarray(numpy_array, share=True)
