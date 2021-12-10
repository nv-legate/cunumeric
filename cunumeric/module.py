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
import re
import sys
from inspect import signature
from itertools import chain
from typing import Optional, Set

import numpy as np
import opt_einsum as oe

from .array import ndarray
from .config import BinaryOpCode, UnaryOpCode, UnaryRedCode
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

        def wrapper(*args, **kwargs):
            assert (where_idx is None or len(args) <= where_idx) and (
                out_idx is None or len(args) <= out_idx
            ), "'where' and 'out' should be passed as keyword arguments"
            stacklevel = kwargs.get("stacklevel", 0) + 1
            kwargs["stacklevel"] = stacklevel

            # Convert relevant arguments to cuNumeric ndarrays
            args = tuple(
                ndarray.convert_to_cunumeric_ndarray(
                    arg, stacklevel=stacklevel
                )
                if idx in indices and arg is not None
                else arg
                for (idx, arg) in enumerate(args)
            )
            for (k, v) in kwargs.items():
                if v is None:
                    continue
                elif k == "where":
                    kwargs[k] = ndarray.convert_to_predicate_ndarray(
                        v, stacklevel=stacklevel
                    )
                elif k == "out":
                    kwargs[k] = ndarray.convert_to_cunumeric_ndarray(
                        v, stacklevel=stacklevel, share=True
                    )
                elif k in keys:
                    kwargs[k] = ndarray.convert_to_cunumeric_ndarray(
                        v, stacklevel=stacklevel
                    )

            # Handle the case where all array-like parameters are scalar, by
            # performing the operation on the equivalent scalar numpy arrays.
            # NOTE: This implicitly blocks on the contents of these arrays.
            if (
                hasattr(np, func.__name__)
                and all(
                    arg._thunk.scalar
                    for (idx, arg) in enumerate(args)
                    if (idx in indices) and isinstance(arg, ndarray)
                )
                and all(
                    v._thunk.scalar
                    for (k, v) in kwargs.items()
                    if (k in keys or k == "where") and isinstance(v, ndarray)
                )
            ):
                out = None
                if "out" in kwargs:
                    out = kwargs["out"]
                    del kwargs["out"]
                del kwargs["stacklevel"]
                args = tuple(
                    arg._thunk.__numpy_array__(stacklevel=stacklevel)
                    if (idx in indices) and isinstance(arg, ndarray)
                    else arg
                    for (idx, arg) in enumerate(args)
                )
                for (k, v) in kwargs.items():
                    if (k in keys or k == "where") and isinstance(v, ndarray):
                        kwargs[k] = v._thunk.__numpy_array__(
                            stacklevel=stacklevel
                        )
                result = ndarray.convert_to_cunumeric_ndarray(
                    getattr(np, func.__name__)(*args, **kwargs)
                )
                if out is not None:
                    out._thunk.copy(result._thunk, stacklevel=stacklevel)
                    result = out
                return result

            return func(*args, **kwargs)

        return wrapper

    return decorator


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
        array = obj
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
    array = ndarray.convert_to_cunumeric_ndarray(v)
    if array._thunk.scalar:
        return array.copy()
    elif array.ndim == 1:
        # Make a diagonal matrix from the array
        N = array.shape[0] + builtins.abs(k)
        matrix = ndarray((N, N), dtype=array.dtype, inputs=(array,))
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
        vector = ndarray(distance, dtype=array.dtype, inputs=(array,))
        vector._thunk.diag(array._thunk, extract=True, k=k, stacklevel=2)
        return vector
    elif array.ndim > 2:
        raise ValueError("diag requires 1- or 2-D array")


@copy_docstring(np.empty)
def empty(shape, dtype=np.float64, stacklevel=1):
    return ndarray(shape=shape, dtype=dtype, stacklevel=(stacklevel + 1))


@copy_docstring(np.empty_like)
def empty_like(a, dtype=None, stacklevel=1):
    array = ndarray.convert_to_cunumeric_ndarray(
        a, stacklevel=(stacklevel + 1)
    )
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

    start = ndarray.convert_to_cunumeric_ndarray(start)
    stop = ndarray.convert_to_cunumeric_ndarray(stop)

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
    array = ndarray.convert_to_cunumeric_ndarray(a)
    result = empty_like(array, dtype=array.dtype, stacklevel=2)
    result._thunk.copy(array._thunk, deep=True, stacklevel=2)
    return result


@copy_docstring(np.tril)
def tril(m, k=0):
    return trilu(m, k, True)


@copy_docstring(np.triu)
def triu(m, k=0):
    return trilu(m, k, False)


def trilu(m, k, lower, stacklevel=2):
    array = ndarray.convert_to_cunumeric_ndarray(m)
    if array.ndim < 1:
        raise TypeError("Array must be at least 1-D")
    shape = m.shape if m.ndim >= 2 else m.shape * 2
    result = ndarray(
        shape, dtype=array.dtype, stacklevel=stacklevel + 1, inputs=(array,)
    )
    result._thunk.trilu(array._thunk, k, lower, stacklevel=2)
    return result


# ### ARRAY MANIPULATION ROUTINES

# Changing array shape


@copy_docstring(np.ravel)
def ravel(a, order="C"):
    array = ndarray.convert_to_cunumeric_ndarray(a)
    return array.ravel(order=order, stacklevel=2)


@copy_docstring(np.reshape)
def reshape(a, newshape, order="C"):
    array = ndarray.convert_to_cunumeric_ndarray(a)
    return array.reshape(newshape, order=order, stacklevel=2)


@copy_docstring(np.transpose)
def transpose(a, axes=None):
    return a.transpose(axes=axes)


@copy_docstring(np.flip)
def flip(m, axis=None):
    array = ndarray.convert_to_cunumeric_ndarray(m)
    return array.flip(axis=axis)


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
    array = ndarray.convert_to_cunumeric_ndarray(a)
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


# Spliting arrays
@copy_docstring(np.vsplit)
def vsplit(a, indices):
    return split(a, indices, axis=0)


@copy_docstring(np.hsplit)
def hsplit(a, indices):
    return split(a, indices, axis=1)


@copy_docstring(np.dsplit)
def dsplit(a, indices):
    return split(a, indices, axis=2)


@copy_docstring(np.split)
def split(a, indices, axis=0):
    return array_split(a, indices, axis, equal=True)


@copy_docstring(np.array_split)
def array_split(a, indices, axis=0, equal=False):
    array = ndarray.convert_to_cunumeric_ndarray(a)
    dtype = type(indices)
    split_pts = []

    if dtype == int:
        res = array.shape[axis] % indices
        len_subarr = array.shape[axis] // indices
        first_idx = len_subarr
        end_idx = array.shape[axis]

        if array.ndim >= axis:
            if equal and array.shape[axis] % indices == 0:
                raise ValueError(
                    "array split does not result in an equal divison"
                )
        else:
            raise ValueError(
                "array({}) has less dimensions than axis({})".format(
                    array.shape, axis
                )
            )

        # the requested # of subarray is larger than the size of array
        # -> size of 1 subarrays + empty subarrays
        if len_subarr == 0:
            len_subarr = 1
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
    elif dtype == np.array or dtype == list:
        split_pts = indices
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
            new_subarray = array[(tuple)(in_shape)].copy()
        else:
            out_shape[axis] = 0
            new_subarray = ndarray(tuple(out_shape), dtype=array.dtype)
        result.append(new_subarray)
        start_idx = pts

    # If the last element in `indices` is larger than array.shape[axis],
    # an empty dummy array should be created
    if start_idx > array.shape[axis]:
        out_shape[axis] = 0
        result.append(ndarray(tuple(out_shape), dtype=array.dtype))

    return result


# ### BINARY OPERATIONS

# Elementwise bit operations


@copy_docstring(np.invert)
def invert(a, out=None, where=True, dtype=None):
    array = ndarray.convert_to_cunumeric_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
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
    a_array = ndarray.convert_to_cunumeric_ndarray(a)
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
    return a_array.dot(b, out=out, stacklevel=2)


# Trivial multi-tensor contraction strategy: contract in input order
class NullOptimizer(oe.paths.PathOptimizer):
    def __call__(self, inputs, output, size_dict, memory_limit=None):
        return [(0, 1)] + [(0, -1)] * (len(inputs) - 2)


# Generalized tensor contraction
@add_boilerplate("a", "b")
def _contract(expr, a, b=None, out=None, stacklevel=1):
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

    # TODO: Handle duplicate modes on inputs
    if len(set(a_modes)) != len(a_modes) or len(set(b_modes)) != len(b_modes):
        raise NotImplementedError("Duplicate mode labels on input tensor")

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
            assert all(ax >= 0 for ax in axes)
            c = a.transpose(axes, stacklevel=(stacklevel + 1))

    else:
        # Binary contraction case
        # Create result array, if output array can't be directly targeted
        if out is not None and out_dtype == c_dtype and out_shape == c_shape:
            c = out
        else:
            c = ndarray(
                shape=c_shape,
                dtype=c_dtype,
                stacklevel=(stacklevel + 1),
                inputs=(a, b),
            )
        # Check for type conversion on the way in
        if a.dtype != c.dtype:
            temp = ndarray(
                shape=a.shape,
                dtype=c.dtype,
                stacklevel=(stacklevel + 1),
                inputs=(a,),
            )
            temp._thunk.convert(a._thunk, stacklevel=(stacklevel + 1))
            a = temp
        if b.dtype != c.dtype:
            temp = ndarray(
                shape=b.shape,
                dtype=c.dtype,
                stacklevel=(stacklevel + 1),
                inputs=(b,),
            )
            temp._thunk.convert(b._thunk, stacklevel=(stacklevel + 1))
            b = temp
        # Perform operation
        c._thunk.contract(
            c_modes,
            a._thunk,
            a_modes,
            b._thunk,
            b_modes,
            mode2extent,
            stacklevel=(stacklevel + 1),
        )

    # Postprocess result before returning
    if out is c:
        # We already decided above to use the output array directly
        return out
    if out_dtype != c_dtype or out_shape != c_bloated_shape:
        # We need to broadcast the result of the contraction or switch types
        # before returning
        if out is None:
            out = zeros(out_shape, out_dtype, stacklevel=(stacklevel + 1))
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


@copy_docstring(np.einsum)
def einsum(expr, *operands, out=None, optimize=False):
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


@copy_docstring(np.tensordot)
def tensordot(a, b, axes=2):
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


# ### LOGIC FUNCTIONS


@copy_docstring(np.logical_not)
def logical_not(a, out=None, where=True, dtype=None, **kwargs):
    a_array = ndarray.convert_to_cunumeric_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.LOGICAL_NOT,
        a_array,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=np.dtype(np.bool_),
    )


# truth value testing


@copy_docstring(np.all)
def all(a, axis=None, out=None, keepdims=False, where=True):
    lg_array = ndarray.convert_to_cunumeric_ndarray(a, stacklevel=2)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
    return lg_array.all(axis=axis, out=out, keepdims=keepdims, where=where)


@copy_docstring(np.any)
def any(a, axis=None, out=None, keepdims=False, where=True):
    lg_array = ndarray.convert_to_cunumeric_ndarray(a, stacklevel=2)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
    return lg_array.any(axis=axis, out=out, keepdims=keepdims, where=where)


# #Comparison


@copy_docstring(np.allclose)
def allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
    a_array = ndarray.convert_to_cunumeric_ndarray(a)
    b_array = ndarray.convert_to_cunumeric_ndarray(b)
    if equal_nan:
        raise NotImplementedError(
            "cuNumeric does not support equal NaN yet for allclose"
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
    a_array = ndarray.convert_to_cunumeric_ndarray(a)
    b_array = ndarray.convert_to_cunumeric_ndarray(b)
    if a_array.ndim != b_array.ndim:
        return False
    if a_array.shape != b_array.shape:
        return False
    return ndarray.perform_binary_reduction(
        BinaryOpCode.EQUAL, a_array, b_array, dtype=np.dtype(np.bool_)
    )


@copy_docstring(np.equal)
def equal(a, b, out=None, where=True, dtype=np.dtype(np.bool), stacklevel=1):
    a_array = ndarray.convert_to_cunumeric_ndarray(
        a, stacklevel=(stacklevel + 1)
    )
    b_array = ndarray.convert_to_cunumeric_ndarray(
        b, stacklevel=(stacklevel + 1)
    )
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(
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
    a_array = ndarray.convert_to_cunumeric_ndarray(
        a, stacklevel=(stacklevel + 1)
    )
    b_array = ndarray.convert_to_cunumeric_ndarray(
        b, stacklevel=(stacklevel + 1)
    )
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(
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
    a_array = ndarray.convert_to_cunumeric_ndarray(
        a, stacklevel=(stacklevel + 1)
    )
    b_array = ndarray.convert_to_cunumeric_ndarray(
        b, stacklevel=(stacklevel + 1)
    )
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(
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
    a_array = ndarray.convert_to_cunumeric_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
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
    a_array = ndarray.convert_to_cunumeric_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
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
    a_array = ndarray.convert_to_cunumeric_ndarray(
        a, stacklevel=(stacklevel + 1)
    )
    b_array = ndarray.convert_to_cunumeric_ndarray(
        b, stacklevel=(stacklevel + 1)
    )
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(
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
    a_array = ndarray.convert_to_cunumeric_ndarray(
        a, stacklevel=(stacklevel + 1)
    )
    b_array = ndarray.convert_to_cunumeric_ndarray(
        b, stacklevel=(stacklevel + 1)
    )
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(
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


@copy_docstring(np.logical_and)
def logical_and(
    a, b, out=None, where=True, dtype=np.dtype(np.bool), stacklevel=1
):
    a_array = ndarray.convert_to_cunumeric_ndarray(
        a, stacklevel=(stacklevel + 1)
    )
    b_array = ndarray.convert_to_cunumeric_ndarray(
        b, stacklevel=(stacklevel + 1)
    )
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(
            out, stacklevel=(stacklevel + 1), share=True
        )
    return ndarray.perform_binary_op(
        BinaryOpCode.LOGICAL_AND,
        a_array,
        b_array,
        out=out,
        out_dtype=dtype,
        where=where,
        stacklevel=(stacklevel + 1),
    )


@copy_docstring(np.logical_or)
def logical_or(
    a, b, out=None, where=True, dtype=np.dtype(np.bool), stacklevel=1
):
    a_array = ndarray.convert_to_cunumeric_ndarray(
        a, stacklevel=(stacklevel + 1)
    )
    b_array = ndarray.convert_to_cunumeric_ndarray(
        b, stacklevel=(stacklevel + 1)
    )
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(
            out, stacklevel=(stacklevel + 1), share=True
        )
    return ndarray.perform_binary_op(
        BinaryOpCode.LOGICAL_OR,
        a_array,
        b_array,
        out=out,
        out_dtype=dtype,
        where=where,
        stacklevel=(stacklevel + 1),
    )


@copy_docstring(np.logical_xor)
def logical_xor(
    a, b, out=None, where=True, dtype=np.dtype(np.bool), stacklevel=1
):
    a_array = ndarray.convert_to_cunumeric_ndarray(
        a, stacklevel=(stacklevel + 1)
    )
    b_array = ndarray.convert_to_cunumeric_ndarray(
        b, stacklevel=(stacklevel + 1)
    )
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(
            out, stacklevel=(stacklevel + 1), share=True
        )
    return ndarray.perform_binary_op(
        BinaryOpCode.LOGICAL_XOR,
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
    a_array = ndarray.convert_to_cunumeric_ndarray(
        a, stacklevel=(stacklevel + 1)
    )
    b_array = ndarray.convert_to_cunumeric_ndarray(
        b, stacklevel=(stacklevel + 1)
    )
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(
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
    a_array = ndarray.convert_to_cunumeric_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    if (
        a_array.dtype.type == np.uint16
        or a_array.dtype.type == np.uint32
        or a_array.dtype.type == np.uint64
    ):
        raise TypeError("cannot negate unsigned type " + str(a_array.dtype))
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.NEGATIVE, a_array, dtype=dtype, dst=out, where=where
    )


@copy_docstring(np.rint)
def rint(a, out=None, where=True, dtype=None, stacklevel=1, **kwargs):
    lg_array = ndarray.convert_to_cunumeric_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if lg_array.dtype.kind == "f" or lg_array.dtype.kind == "c":
        out_dtype = lg_array.dtype
    else:
        out_dtype = np.dtype(np.float64)
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.RINT,
        lg_array,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=out_dtype,
    )


# Trigonometric functions


@copy_docstring(np.arccos)
def arccos(a, out=None, where=True, dtype=None, **kwargs):
    lg_array = ndarray.convert_to_cunumeric_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    # Floats keep their floating point kind, otherwise switch to float64
    if lg_array.dtype.kind == "f" or lg_array.dtype.kind == "c":
        out_dtype = lg_array.dtype
    else:
        out_dtype = np.dtype(np.float64)
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
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
    lg_array = ndarray.convert_to_cunumeric_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    # Floats keep their floating point kind, otherwise switch to float64
    if lg_array.dtype.kind == "f" or lg_array.dtype.kind == "c":
        out_dtype = lg_array.dtype
    else:
        out_dtype = np.dtype(np.float64)
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
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
    lg_array = ndarray.convert_to_cunumeric_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    # Floats keep their floating point kind, otherwise switch to float64
    if lg_array.dtype.kind == "f" or lg_array.dtype.kind == "c":
        out_dtype = lg_array.dtype
    else:
        out_dtype = np.dtype(np.float64)
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
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
    lg_array = ndarray.convert_to_cunumeric_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    # Floats keep their floating point kind, otherwise switch to float64
    if lg_array.dtype.kind == "f" or lg_array.dtype.kind == "c":
        out_dtype = lg_array.dtype
    else:
        out_dtype = np.dtype(np.float64)
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.COS,
        lg_array,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=out_dtype,
    )


@copy_docstring(np.sign)
def sign(a, out=None, where=True, dtype=None, **kwargs):
    lg_array = ndarray.convert_to_cunumeric_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    # Floats keep their floating point kind, otherwise switch to float64
    if lg_array.dtype.kind == "f" or lg_array.dtype.kind == "c":
        out_dtype = lg_array.dtype
    else:
        out_dtype = np.dtype(np.float64)
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.SIGN,
        lg_array,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=out_dtype,
    )


@copy_docstring(np.sin)
def sin(a, out=None, where=True, dtype=None, **kwargs):
    lg_array = ndarray.convert_to_cunumeric_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    # Floats keep their floating point kind, otherwise switch to float64
    if lg_array.dtype.kind == "f" or lg_array.dtype.kind == "c":
        out_dtype = lg_array.dtype
    else:
        out_dtype = np.dtype(np.float64)
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
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
    lg_array = ndarray.convert_to_cunumeric_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    # Floats keep their floating point kind, otherwise switch to float64
    if lg_array.dtype.kind == "f" or lg_array.dtype.kind == "c":
        out_dtype = lg_array.dtype
    else:
        out_dtype = np.dtype(np.float64)
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
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
    lg_array = ndarray.convert_to_cunumeric_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    # Floats keep their floating point kind, otherwise switch to float64
    if lg_array.dtype.kind == "f" or lg_array.dtype.kind == "c":
        out_dtype = lg_array.dtype
    else:
        out_dtype = np.dtype(np.float64)
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
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
    a_array = ndarray.convert_to_cunumeric_ndarray(
        a, stacklevel=(stacklevel + 1)
    )
    b_array = ndarray.convert_to_cunumeric_ndarray(
        b, stacklevel=(stacklevel + 1)
    )
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(
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
    a_array = ndarray.convert_to_cunumeric_ndarray(a)
    b_array = ndarray.convert_to_cunumeric_ndarray(b)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
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
    a_array = ndarray.convert_to_cunumeric_ndarray(a)
    b_array = ndarray.convert_to_cunumeric_ndarray(b)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
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
    a_array = ndarray.convert_to_cunumeric_ndarray(
        a, stacklevel=(stacklevel + 1)
    )
    b_array = ndarray.convert_to_cunumeric_ndarray(
        b, stacklevel=(stacklevel + 1)
    )
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(
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
    lg_array = ndarray.convert_to_cunumeric_ndarray(
        a, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(
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
    a_array = ndarray.convert_to_cunumeric_ndarray(
        a, stacklevel=(stacklevel + 1)
    )
    b_array = ndarray.convert_to_cunumeric_ndarray(
        b, stacklevel=(stacklevel + 1)
    )
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(
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
    lg_array = ndarray.convert_to_cunumeric_ndarray(
        a, stacklevel=(stacklevel + 1)
    )
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(
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
    a_array = ndarray.convert_to_cunumeric_ndarray(
        a, stacklevel=(stacklevel + 1)
    )
    b_array = ndarray.convert_to_cunumeric_ndarray(
        b, stacklevel=(stacklevel + 1)
    )
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
        out = ndarray.convert_to_cunumeric_ndarray(
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
    lg_array = ndarray.convert_to_cunumeric_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if lg_array.dtype.kind == "f" or lg_array.dtype.kind == "c":
        out_dtype = lg_array.dtype
    else:
        out_dtype = np.dtype(np.float64)
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.EXP,
        lg_array,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=out_dtype,
    )


@copy_docstring(np.exp2)
def exp2(a, out=None, where=True, dtype=None, stacklevel=1, **kwargs):
    lg_array = ndarray.convert_to_cunumeric_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if lg_array.dtype.kind == "f" or lg_array.dtype.kind == "c":
        out_dtype = lg_array.dtype
    else:
        out_dtype = np.dtype(np.float64)
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.EXP2,
        lg_array,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=out_dtype,
    )


@copy_docstring(np.log)
def log(a, out=None, where=True, dtype=None, stacklevel=1, **kwargs):
    lg_array = ndarray.convert_to_cunumeric_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if lg_array.dtype.kind == "f" or lg_array.dtype.kind == "c":
        out_dtype = lg_array.dtype
    else:
        out_dtype = np.dtype(np.float64)
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.LOG,
        lg_array,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=out_dtype,
    )


@copy_docstring(np.log10)
def log10(a, out=None, where=True, dtype=None, stacklevel=1, **kwargs):
    lg_array = ndarray.convert_to_cunumeric_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if lg_array.dtype.kind == "f" or lg_array.dtype.kind == "c":
        out_dtype = lg_array.dtype
    else:
        out_dtype = np.dtype(np.float64)
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.LOG10,
        lg_array,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=out_dtype,
    )


@copy_docstring(np.power)
def power(x1, x2, out=None, where=True, dtype=None, stacklevel=1, **kwargs):
    x1_array = ndarray.convert_to_cunumeric_ndarray(
        x1, stacklevel=(stacklevel + 1)
    )
    x2_array = ndarray.convert_to_cunumeric_ndarray(
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
        out = ndarray.convert_to_cunumeric_ndarray(
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
    lg_array = ndarray.convert_to_cunumeric_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
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
    lg_array = ndarray.convert_to_cunumeric_ndarray(a)
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
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.ABSOLUTE, lg_array, dst=out, where=where
    )


abs = absolute  # alias


@copy_docstring(np.ceil)
def ceil(a, out=None, where=True, dtype=None, **kwargs):
    lg_array = ndarray.convert_to_cunumeric_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    # If this is an integer array then there is nothing to do for ceil
    if (
        lg_array.dtype.kind == "i"
        or lg_array.dtype.kind == "u"
        or lg_array.dtype.kind == "b"
    ):
        return lg_array
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.CEIL, lg_array, dst=out, dtype=dtype, where=where
    )


@copy_docstring(np.clip)
def clip(a, a_min, a_max, out=None):
    lg_array = ndarray.convert_to_cunumeric_ndarray(a)
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
    return lg_array.clip(a_min, a_max, out=out)


@copy_docstring(np.fabs)
def fabs(a, out=None, where=True, **kwargs):
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
    return absolute(a, out=out, where=where, **kwargs)


@copy_docstring(np.floor)
def floor(a, out=None, where=True, dtype=None, **kwargs):
    lg_array = ndarray.convert_to_cunumeric_ndarray(a)
    where = ndarray.convert_to_predicate_ndarray(where, stacklevel=2)
    # If this is an integer array then there is nothing to do for floor
    if (
        lg_array.dtype.kind == "i"
        or lg_array.dtype.kind == "u"
        or lg_array.dtype.kind == "b"
    ):
        return lg_array
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
    return ndarray.perform_unary_op(
        UnaryOpCode.FLOOR, lg_array, dst=out, dtype=dtype, where=where
    )


@copy_docstring(np.sqrt)
def sqrt(a, out=None, where=True, dtype=None, stacklevel=1, **kwargs):
    lg_array = ndarray.convert_to_cunumeric_ndarray(
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
        out = ndarray.convert_to_cunumeric_ndarray(
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


@copy_docstring(np.convolve)
def convolve(a, v, mode="full"):
    a_lg = ndarray.convert_to_cunumeric_ndarray(a, stacklevel=2)
    v_lg = ndarray.convert_to_cunumeric_ndarray(v, stacklevel=2)

    if mode != "same":
        raise NotImplementedError("Need to implement other convolution modes")

    if a_lg.size < v_lg.size:
        v_lg, a_lg = a_lg, v_lg

    return a_lg.convolve(v_lg, mode, stacklevel=2)


# ### SORTING, SEARCHING and COUNTING

# Searching


@copy_docstring(np.argmax)
def argmax(a, axis=None, out=None):
    lg_array = ndarray.convert_to_cunumeric_ndarray(a)
    if out is not None:
        if out.dtype != np.int64:
            raise ValueError("output array must have int64 dtype")
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
    return lg_array.argmax(axis=axis, out=out, stacklevel=2)


@copy_docstring(np.argmin)
def argmin(a, axis=None, out=None):
    lg_array = ndarray.convert_to_cunumeric_ndarray(a)
    if out is not None:
        if out is not None and out.dtype != np.int64:
            raise ValueError("output array must have int64 dtype")
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
    return lg_array.argmin(axis=axis, out=out, stacklevel=2)


@copy_docstring(np.bincount)
def bincount(a, weights=None, minlength=0):
    lg_array = ndarray.convert_to_cunumeric_ndarray(a)
    if weights is not None:
        lg_weights = ndarray.convert_to_cunumeric_ndarray(weights)
        if lg_weights.shape != lg_array.shape:
            raise ValueError("weights array must be same shape for bincount")
        if lg_weights.dtype.kind == "c":
            raise ValueError("weights must be convertible to float64")
        # Make sure the weights are float64
        lg_weights = lg_weights.astype(np.float64)
    if lg_array.dtype.kind != "i" and lg_array.dtype.kind != "u":
        raise TypeError("input array for bincount must be integer type")
    # If nobody told us the size then compute it
    if minlength <= 0:
        minlength = int(amax(lg_array)) + 1
    if lg_array.size == 1:
        # Handle the special case of 0-D array
        if weights is None:
            out = zeros((minlength,), dtype=np.dtype(np.int64), stacklevel=2)
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
                dtype=np.dtype(np.int64),
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
    lg_array = ndarray.convert_to_cunumeric_ndarray(a)
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
    lg_array = ndarray.convert_to_cunumeric_ndarray(a)
    x_array = ndarray.convert_to_cunumeric_ndarray(x)
    y_array = ndarray.convert_to_cunumeric_ndarray(y)
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
    return ndarray.perform_where(
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
#    lg_array = ndarray.convert_to_cunumeric_ndarray(a)
#    if order is not None:
#        raise NotImplementedError("No support for non-None 'order' for sort")
#    if kind != 'quicksort':
#        warnings.warn("cuNumeric uses a different algorithm than "+str(kind)+
#                      " for sorting",
#                      category=RuntimeWarning, stacklevel=2)
#    # Make the out array
#    if lg_array.ndim > 1:
#        # We only support sorting the full array right now
#        if axis is not None:
#            raise NotImplementedError("cuNumeric only supports sorting full "
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
@add_boilerplate("a")
def count_nonzero(a, axis=None, stacklevel=1):
    if a.size == 0:
        return 0
    return ndarray.perform_unary_reduction(
        UnaryRedCode.COUNT_NONZERO,
        a,
        axis=axis,
        dtype=np.dtype(np.uint64),
        stacklevel=(stacklevel + 1),
        check_types=False,
    )


# ### STATISTICS

# Order statistics


@copy_docstring(np.amax)
def amax(a, axis=None, out=None, keepdims=False, stacklevel=1):
    lg_array = ndarray.convert_to_cunumeric_ndarray(
        a, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(
            out, stacklevel=(stacklevel + 1), share=True
        )
    return lg_array.max(
        axis=axis, out=out, keepdims=keepdims, stacklevel=(stacklevel + 1)
    )


@copy_docstring(np.amin)
def amin(a, axis=None, out=None, keepdims=False, stacklevel=1):
    lg_array = ndarray.convert_to_cunumeric_ndarray(
        a, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(
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
    a_array = ndarray.convert_to_cunumeric_ndarray(
        a, stacklevel=(stacklevel + 1)
    )
    b_array = ndarray.convert_to_cunumeric_ndarray(
        b, stacklevel=(stacklevel + 1)
    )
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(
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
    a_array = ndarray.convert_to_cunumeric_ndarray(
        a, stacklevel=(stacklevel + 1)
    )
    b_array = ndarray.convert_to_cunumeric_ndarray(
        b, stacklevel=(stacklevel + 1)
    )
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(
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
    a_array = ndarray.convert_to_cunumeric_ndarray(
        a, stacklevel=(stacklevel + 1)
    )
    b_array = ndarray.convert_to_cunumeric_ndarray(
        b, stacklevel=(stacklevel + 1)
    )
    where = ndarray.convert_to_predicate_ndarray(
        where, stacklevel=(stacklevel + 1)
    )
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(
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
    lg_array = ndarray.convert_to_cunumeric_ndarray(a)
    if out is not None:
        out = ndarray.convert_to_cunumeric_ndarray(out, share=True)
    return lg_array.mean(
        axis=axis, dtype=dtype, out=out, keepdims=keepdims, stacklevel=2
    )


# ### STACKING and CONCATENATION ###


@copy_docstring(np.row_stack)
def row_stack(inputs):
    return vstack(inputs)


@copy_docstring(np.vstack)
def vstack(inputs):
    # Check to see if we can build a new tuple of cuNumeric arrays
    dtype = None
    cunumeric_inputs = list()
    for inp in inputs:
        lg_array = ndarray.convert_to_cunumeric_ndarray(inp)
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
        cunumeric_inputs.append(lg_array)
    # Once we are here we have all our inputs arrays, so make the output
    if len(shape) == 1:
        out_shape = (leading_dim,) + shape
        out_array = ndarray(
            shape=out_shape, dtype=dtype, inputs=cunumeric_inputs
        )
        # Copy the values over from the inputs
        for idx, inp in enumerate(cunumeric_inputs):
            out_array[idx, :] = inp
    else:
        out_shape = (leading_dim,)
        for dim in xrange(1, ndim):
            out_shape += (shape[dim],)
        out_array = ndarray(
            shape=out_shape, dtype=dtype, inputs=cunumeric_inputs
        )
        # Copy the values over from the inputs
        offset = 0
        for inp in cunumeric_inputs:
            out_array[offset : offset + inp.shape[0], ...] = inp
            offset += inp.shape[0]
    return out_array


# ### FILE I/O ###
# For now we just need to do the loading methods to get things converted into
# cuNumeric arrays. The storing methods can go through the normal NumPy python


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
    return ndarray.convert_to_cunumeric_ndarray(numpy_array)


@copy_docstring(np.frombuffer)
def frombuffer(buffer, dtype=float, count=-1, offset=0):
    numpy_array = np.frombuffer(
        buffer=buffer, dtype=dtype, count=count, offset=offset
    )
    return ndarray.convert_to_cunumeric_ndarray(numpy_array)


@copy_docstring(np.fromfile)
def fromfile(file, dtype=float, count=-1, sep="", offset=0):
    numpy_array = np.fromfile(
        file=file, dtype=dtype, count=count, sep=sep, offset=offset
    )
    return ndarray.convert_to_cunumeric_ndarray(numpy_array)


@copy_docstring(np.fromfunction)
def fromfunction(function, shape, **kwargs):
    numpy_array = np.fromfunction(function=function, shape=shape, **kwargs)
    return ndarray.convert_to_cunumeric_ndarray(numpy_array)


@copy_docstring(np.fromiter)
def fromiter(iterable, dtype, count=-1):
    numpy_array = np.fromiter(iterable=iterable, dtype=dtype, count=count)
    return ndarray.convert_to_cunumeric_ndarray(numpy_array)


@copy_docstring(np.fromregex)
def fromregex(file, regexp, dtype, encoding=None):
    numpy_array = np.fromregex(
        file=file, regexp=regexp, dtype=dtype, encoding=encoding
    )
    return ndarray.convert_to_cunumeric_ndarray(numpy_array)


@copy_docstring(np.fromstring)
def fromstring(string, dtype=float, count=-1, sep=""):
    numpy_array = np.fromstring(
        string=string, dtype=dtype, count=count, sep=sep
    )
    return ndarray.convert_to_cunumeric_ndarray(numpy_array)


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
    return ndarray.convert_to_cunumeric_ndarray(numpy_array)


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
    return ndarray.convert_to_cunumeric_ndarray(numpy_array)


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
    return ndarray.convert_to_cunumeric_ndarray(numpy_array, share=True)
