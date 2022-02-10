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
from collections import Counter
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


# ### ARRAY CREATION ROUTINES


def arange(*args, dtype=None):
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
    result = ndarray((N,), dtype)
    result._thunk.arange(start, stop, step)
    return result


def array(obj, dtype=None, copy=True, order="K", subok=False, ndmin=0):
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


@add_boilerplate("a")
def choose(a, choices, out=None, mode="raise"):
    return a.choose(choices=choices, out=out, mode=mode)


@add_boilerplate("v")
def diag(v, k=0):
    if v.ndim == 0:
        raise ValueError("Input must be 1- or 2-d")
    elif v.ndim == 1:
        return v.diagonal(offset=k, axis1=0, axis2=1, extract=False)
    elif v.ndim == 2:
        return v.diagonal(offset=k, axis1=0, axis2=1, extract=True)
    elif v.ndim > 2:
        raise ValueError("diag requires 1- or 2-D array, use diagonal instead")


@add_boilerplate("a")
def diagonal(a, offset=0, axis1=None, axis2=None, extract=True, axes=None):
    """
    Return specified diagonals.

    See description in numpy
    ------------------------
    https://numpy.org/doc/stable/reference/generated/numpy.diag.html
    https://numpy.org/doc/stable/reference/generated/numpy.diagonal.html#numpy.diagonal
    https://numpy.org/doc/stable/reference/generated/numpy.ndarray.diagonal.html


    cuNumeric implementation differences:
    ------------------------------------
    - It never returns a view
    - support 1D arrays (similar to `diag`)
    - support extra arguments:
         -- extract: used to create diagonal from 1D array vs extracting
           the diagonal from"
         -- axes: list of axes for diagonal ( size of the list should be in
           between 2 and size of the array)

    """
    return a.diagonal(
        offset=offset, axis1=axis1, axis2=axis2, extract=extract, axes=axes
    )


def empty(shape, dtype=np.float64):
    return ndarray(shape=shape, dtype=dtype)


@add_boilerplate("a")
def empty_like(a, dtype=None):
    shape = a.shape
    if dtype is not None:
        dtype = np.dtype(dtype)
    else:
        dtype = a.dtype
    return ndarray(shape, dtype=dtype, inputs=(a,))


def eye(N, M=None, k=0, dtype=np.float64):
    if dtype is not None:
        dtype = np.dtype(dtype)
    if M is None:
        M = N
    result = ndarray((N, M), dtype)
    result._thunk.eye(k)
    return result


def full(shape, value, dtype=None):
    if dtype is None:
        val = np.array(value)
    else:
        dtype = np.dtype(dtype)
        val = np.array(value, dtype=dtype)
    result = empty(shape, dtype=val.dtype)
    result._thunk.fill(val)
    return result


def full_like(a, value, dtype=None):
    if dtype is not None:
        dtype = np.dtype(dtype)
    else:
        dtype = a.dtype
    result = empty_like(a, dtype=dtype)
    val = np.array(value, dtype=result.dtype)
    result._thunk.fill(val)
    return result


def identity(n, dtype=float):
    return eye(N=n, M=n, dtype=dtype)


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


def ones(shape, dtype=np.float64):
    return full(shape, 1, dtype=dtype)


def ones_like(a, dtype=None):
    usedtype = a.dtype
    if dtype is not None:
        usedtype = np.dtype(dtype)
    return full_like(a, 1, dtype=usedtype)


def zeros(shape, dtype=np.float64):
    if dtype is not None:
        dtype = np.dtype(dtype)
    return full(shape, 0, dtype=dtype)


def zeros_like(a, dtype=None):
    usedtype = a.dtype
    if dtype is not None:
        usedtype = np.dtype(dtype)
    return full_like(a, 0, dtype=usedtype)


@add_boilerplate("a")
def copy(a):
    result = empty_like(a, dtype=a.dtype)
    result._thunk.copy(a._thunk, deep=True)
    return result


def tril(m, k=0):
    return trilu(m, k, True)


def triu(m, k=0):
    return trilu(m, k, False)


@add_boilerplate("m")
def trilu(m, k, lower):
    if m.ndim < 1:
        raise TypeError("Array must be at least 1-D")
    shape = m.shape if m.ndim >= 2 else m.shape * 2
    result = ndarray(shape, dtype=m.dtype, inputs=(m,))
    result._thunk.trilu(m._thunk, k, lower)
    return result


# ### ARRAY MANIPULATION ROUTINES

# Changing array shape


@add_boilerplate("a")
def ravel(a, order="C"):
    return a.ravel(order=order)


@add_boilerplate("a")
def reshape(a, newshape, order="C"):
    return a.reshape(newshape, order=order)


@add_boilerplate("a")
def transpose(a, axes=None):
    return a.transpose(axes=axes)


@add_boilerplate("m")
def flip(m, axis=None):
    return m.flip(axis=axis)


# Changing kind of array


def asarray(a, dtype=None, order=None):
    if not isinstance(a, ndarray):
        thunk = runtime.get_numpy_thunk(a, share=True, dtype=dtype)
        array = ndarray(shape=None, thunk=thunk)
    else:
        array = a
    if dtype is not None and array.dtype != dtype:
        array = array.astype(dtype)
    return array


# Tiling


@add_boilerplate("a")
def tile(a, reps):
    if not hasattr(reps, "__len__"):
        reps = (reps,)
    # Figure out the shape of the destination array
    out_dims = a.ndim if a.ndim > len(reps) else len(reps)
    # Prepend ones until the dimensions match
    while len(reps) < out_dims:
        reps = (1,) + reps
    out_shape = ()
    # Prepend dimensions if necessary
    for dim in range(out_dims - a.ndim):
        out_shape += (reps[dim],)
    offset = len(out_shape)
    for dim in range(a.ndim):
        out_shape += (a.shape[dim] * reps[offset + dim],)
    assert len(out_shape) == out_dims
    result = ndarray(out_shape, dtype=a.dtype, inputs=(a,))
    result._thunk.tile(a._thunk, reps)
    return result


# Spliting arrays
def vsplit(a, indices):
    return split(a, indices, axis=0)


def hsplit(a, indices):
    return split(a, indices, axis=1)


def dsplit(a, indices):
    return split(a, indices, axis=2)


def split(a, indices, axis=0):
    return array_split(a, indices, axis, equal=True)


def array_split(a, indices, axis=0, equal=False):
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


# stack / concat operations ###
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


def row_stack(inputs):
    return vstack(inputs)


def column_stack(array_list):
    return hstack(array_list)


def vstack(array_list):
    fname = vstack.__name__
    # Reshape arrays in the `array_list` if needed before concatenation
    array_list, common_info = check_shape_dtype(array_list, fname)
    if common_info.ndim == 1:
        for i, arr in enumerate(array_list):
            array_list[i] = arr.reshape([1, arr.shape[0]])
        common_info.shape = array_list[0].shape
    return _concatenate(
        array_list,
        axis=0,
        dtype=common_info.dtype,
        common_info=common_info,
    )


def hstack(array_list):
    fname = hstack.__name__
    array_list, common_info = check_shape_dtype(array_list, fname)
    if (
        common_info.ndim == 1
    ):  # When ndim == 1, hstack concatenates arrays along the first axis
        return _concatenate(
            array_list,
            axis=0,
            dtype=common_info.dtype,
            common_info=common_info,
        )
    else:
        return _concatenate(
            array_list,
            axis=1,
            dtype=common_info.dtype,
            common_info=common_info,
        )


def dstack(array_list):
    fname = dstack.__name__
    array_list, common_info = check_shape_dtype(array_list, fname)
    # Reshape arrays to (1,N,1) for ndim ==1 or (M,N,1) for ndim == 2:
    if common_info.ndim <= 2:
        shape = list(array_list[0].shape)
        if common_info.ndim == 1:
            shape.insert(0, 1)
        shape.append(1)
        common_info.shape = shape
        for i, arr in enumerate(array_list):
            array_list[i] = arr.reshape(shape)
    return _concatenate(
        array_list,
        axis=2,
        dtype=common_info.dtype,
        common_info=common_info,
    )


def stack(array_list, axis=0, out=None):
    fname = stack.__name__
    array_list, common_info = check_shape_dtype(array_list, fname)
    if axis > common_info.ndim:
        raise ValueError(
            "The target axis should be smaller or"
            " equal to the number of dimensions"
            " of input arrays"
        )
    else:
        shape = list(common_info.shape)
        shape.insert(axis, 1)
        for i, arr in enumerate(array_list):
            array_list[i] = arr.reshape(shape)
        common_info.shape = shape
    return _concatenate(array_list, axis, out=out, common_info=common_info)


def concatenate(inputs, axis=0, out=None, dtype=None, casting="same_kind"):
    # Check to see if we can build a new tuple of cuNumeric arrays
    cunumeric_inputs, common_info = check_shape_dtype(
        inputs, concatenate.__name__, dtype, casting
    )
    return _concatenate(
        cunumeric_inputs, axis, out, dtype, casting, common_info
    )


# ### BINARY OPERATIONS

# Elementwise bit operations


@add_boilerplate("a")
def invert(a, out=None, where=True, dtype=None):
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


# ### LINEAR ALGEBRA

# Matrix and vector products
@add_boilerplate("a", "b")
def dot(a, b, out=None):
    return a.dot(b, out=out)


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


@add_boilerplate("a")
def logical_not(a, out=None, where=True, dtype=None, **kwargs):
    return ndarray.perform_unary_op(
        UnaryOpCode.LOGICAL_NOT,
        a,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=np.dtype(np.bool_),
    )


# truth value testing


@add_boilerplate("a")
def all(a, axis=None, out=None, keepdims=False, where=True):
    return a.all(axis=axis, out=out, keepdims=keepdims, where=where)


@add_boilerplate("a")
def any(a, axis=None, out=None, keepdims=False, where=True):
    return a.any(axis=axis, out=out, keepdims=keepdims, where=where)


# #Comparison


@add_boilerplate("a", "b")
def allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False):
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
    if a.shape != b.shape:
        return False
    return ndarray.perform_binary_reduction(
        BinaryOpCode.EQUAL, a, b, dtype=np.dtype(np.bool_)
    )


@add_boilerplate("a", "b")
def equal(a, b, out=None, where=True, dtype=np.dtype(np.bool)):
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
def greater(a, b, out=None, where=True, dtype=np.dtype(np.bool)):
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
def greater_equal(a, b, out=None, where=True, dtype=np.dtype(np.bool)):
    return ndarray.perform_binary_op(
        BinaryOpCode.GREATER_EQUAL,
        a,
        b,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
        where=where,
    )


@add_boilerplate("a")
def isinf(a, out=None, where=True, dtype=None, **kwargs):
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
    return ndarray.perform_unary_op(
        UnaryOpCode.ISNAN,
        a,
        dst=out,
        dtype=dtype,
        where=where,
        out_dtype=np.dtype(np.bool_),
    )


@add_boilerplate("a", "b")
def less(a, b, out=None, where=True, dtype=np.dtype(np.bool)):
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
def less_equal(a, b, out=None, where=True, dtype=np.dtype(np.bool)):
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
def logical_and(a, b, out=None, where=True, dtype=np.dtype(np.bool)):
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
def logical_or(a, b, out=None, where=True, dtype=np.dtype(np.bool)):
    return ndarray.perform_binary_op(
        BinaryOpCode.LOGICAL_OR,
        a,
        b,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
        where=where,
    )


@add_boilerplate("a", "b")
def logical_xor(a, b, out=None, where=True, dtype=np.dtype(np.bool)):
    return ndarray.perform_binary_op(
        BinaryOpCode.LOGICAL_XOR,
        a,
        b,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
        where=where,
    )


@add_boilerplate("a", "b")
def not_equal(a, b, out=None, where=True, dtype=np.dtype(np.bool)):
    return ndarray.perform_binary_op(
        BinaryOpCode.NOT_EQUAL,
        a,
        b,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
        where=where,
    )


# ### MATHEMATICAL FUNCTIONS


@add_boilerplate("a")
def negative(a, out=None, where=True, dtype=None, **kwargs):
    return ndarray.perform_unary_op(
        UnaryOpCode.NEGATIVE, a, dtype=dtype, dst=out, where=where
    )


def _output_float_dtype(input):
    # Floats keep their floating point kind, otherwise switch to float64
    if input.dtype.kind in ("f", "c"):
        return input.dtype
    else:
        return np.dtype(np.float64)


@add_boilerplate("a")
def rint(a, out=None, where=True, dtype=None, **kwargs):
    return ndarray.perform_unary_op(
        UnaryOpCode.RINT,
        a,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=_output_float_dtype(a),
    )


# Trigonometric functions


@add_boilerplate("a")
def arccos(a, out=None, where=True, dtype=None, **kwargs):
    return ndarray.perform_unary_op(
        UnaryOpCode.ARCCOS,
        a,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=_output_float_dtype(a),
    )


@add_boilerplate("a")
def arcsin(a, out=None, where=True, dtype=None, **kwargs):
    return ndarray.perform_unary_op(
        UnaryOpCode.ARCSIN,
        a,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=_output_float_dtype(a),
    )


@add_boilerplate("a")
def arctan(a, out=None, where=True, dtype=None, **kwargs):
    return ndarray.perform_unary_op(
        UnaryOpCode.ARCTAN,
        a,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=_output_float_dtype(a),
    )


@add_boilerplate("a")
def cos(a, out=None, where=True, dtype=None, **kwargs):
    return ndarray.perform_unary_op(
        UnaryOpCode.COS,
        a,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=_output_float_dtype(a),
    )


@add_boilerplate("a")
def sign(a, out=None, where=True, dtype=None, **kwargs):
    return ndarray.perform_unary_op(
        UnaryOpCode.SIGN,
        a,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=_output_float_dtype(a),
    )


@add_boilerplate("a")
def sin(a, out=None, where=True, dtype=None, **kwargs):
    return ndarray.perform_unary_op(
        UnaryOpCode.SIN,
        a,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=_output_float_dtype(a),
    )


@add_boilerplate("a")
def tan(a, out=None, where=True, dtype=None, **kwargs):
    return ndarray.perform_unary_op(
        UnaryOpCode.TAN,
        a,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=_output_float_dtype(a),
    )


# Hyperbolic functions


@add_boilerplate("a")
def tanh(a, out=None, where=True, dtype=None, **kwargs):
    return ndarray.perform_unary_op(
        UnaryOpCode.TANH,
        a,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=_output_float_dtype(a),
    )


# Sums, products, differences


@add_boilerplate("a", "b")
def add(a, b, out=None, where=True, dtype=None):
    return ndarray.perform_binary_op(
        BinaryOpCode.ADD,
        a,
        b,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
        where=where,
    )


def divide(a, b, out=None, where=True, dtype=None):
    return true_divide(a, b, out=out, where=where, dtype=dtype)


@add_boilerplate("a", "b")
def floor_divide(a, b, out=None, where=True, dtype=None):
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
def multiply(a, b, out=None, where=True, dtype=None):
    return ndarray.perform_binary_op(
        BinaryOpCode.MULTIPLY,
        a,
        b,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
        where=where,
    )


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
    return a.prod(
        axis=axis,
        dtype=dtype,
        out=out,
        keepdims=keepdims,
        initial=initial,
        where=where,
    )


@add_boilerplate("a", "b")
def subtract(a, b, out=None, where=True, dtype=None):
    return ndarray.perform_binary_op(
        BinaryOpCode.SUBTRACT,
        a,
        b,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
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
    return a.sum(
        axis=axis,
        dtype=dtype,
        out=out,
        keepdims=keepdims,
        initial=initial,
        where=where,
    )


@add_boilerplate("a", "b")
def true_divide(a, b, out=None, where=True, dtype=None):
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


# Exponents and logarithms


@add_boilerplate("a")
def exp(a, out=None, where=True, dtype=None, **kwargs):
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
    return ndarray.perform_unary_op(
        UnaryOpCode.LOG10,
        a,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=_output_float_dtype(a),
    )


@add_boilerplate("x1", "x2")
def power(x1, x2, out=None, where=True, dtype=None, **kwargs):
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


def square(a, out=None, where=True, dtype=None, **kwargs):
    return multiply(a, a, out=out, where=where, dtype=dtype)


# Miscellaneous


@add_boilerplate("a")
def absolute(a, out=None, where=True, **kwargs):
    # Handle the nice case of it being unsigned
    if a.dtype.type in (np.uint16, np.uint32, np.uint64, np.bool_):
        return a
    return ndarray.perform_unary_op(
        UnaryOpCode.ABSOLUTE, a, dst=out, where=where
    )


abs = absolute  # alias


@add_boilerplate("a")
def ceil(a, out=None, where=True, dtype=None, **kwargs):
    # If this is an integer array then there is nothing to do for ceil
    if a.dtype.kind in ("i", "u", "b"):
        return a
    return ndarray.perform_unary_op(
        UnaryOpCode.CEIL, a, dst=out, dtype=dtype, where=where
    )


@add_boilerplate("a")
def clip(a, a_min, a_max, out=None):
    return a.clip(a_min, a_max, out=out)


def fabs(a, out=None, where=True, **kwargs):
    return absolute(a, out=out, where=where, **kwargs)


@add_boilerplate("a")
def floor(a, out=None, where=True, dtype=None, **kwargs):
    # If this is an integer array then there is nothing to do for floor
    if a.dtype.kind in ("i", "u", "b"):
        return a
    return ndarray.perform_unary_op(
        UnaryOpCode.FLOOR, a, dst=out, dtype=dtype, where=where
    )


@add_boilerplate("a")
def sqrt(a, out=None, where=True, dtype=None, **kwargs):
    return ndarray.perform_unary_op(
        UnaryOpCode.SQRT,
        a,
        dtype=dtype,
        dst=out,
        where=where,
        out_dtype=_output_float_dtype(a),
    )


@add_boilerplate("a", "v")
def convolve(a, v, mode="full"):
    if mode != "same":
        raise NotImplementedError("Need to implement other convolution modes")

    if a.size < v.size:
        v, a = a, v

    return a.convolve(v, mode)


# ### SORTING, SEARCHING and COUNTING

# Searching


@add_boilerplate("a")
def argmax(a, axis=None, out=None):
    if out is not None:
        if out.dtype != np.int64:
            raise ValueError("output array must have int64 dtype")
    return a.argmax(axis=axis, out=out)


@add_boilerplate("a")
def argmin(a, axis=None, out=None):
    if out is not None:
        if out is not None and out.dtype != np.int64:
            raise ValueError("output array must have int64 dtype")
    return a.argmin(axis=axis, out=out)


@add_boilerplate("a", "weights")
def bincount(a, weights=None, minlength=0):
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


@add_boilerplate("a")
def nonzero(a):
    return a.nonzero()


@add_boilerplate("a", "x", "y")
def where(a, x=None, y=None):
    if x is None or y is None:
        if x is not None or y is not None:
            raise ValueError(
                "both 'x' and 'y' parameters must be specified together for"
                " 'where'"
            )
        return nonzero(a)
    return ndarray.perform_where(a, x, y)


# Sorting


@add_boilerplate("a")
def argsort(a, axis=-1, kind="stable", order=None):
    return a.argsort(axis=axis, kind=kind, order=order)


def lexsort(a, axis=-1):
    raise NotImplementedError("Not yet implemented")


def msort(a):
    return sort(a)


@add_boilerplate("a")
def sort(a, axis=-1, kind="stable", order=None):
    out = a.copy()
    out_array = ndarray.convert_to_cunumeric_ndarray(out)
    out_array._thunk.sort(axis=axis, kind=kind, order=order)
    return out_array


def sort_complex(a):
    return sort(a)


# Counting


@add_boilerplate("a")
def count_nonzero(a, axis=None):
    if a.size == 0:
        return 0
    return ndarray.perform_unary_reduction(
        UnaryRedCode.COUNT_NONZERO,
        a,
        axis=axis,
        dtype=np.dtype(np.uint64),
        check_types=False,
    )


# ### STATISTICS

# Order statistics


@add_boilerplate("a")
def amax(a, axis=None, out=None, keepdims=False):
    return a.max(axis=axis, out=out, keepdims=keepdims)


@add_boilerplate("a")
def amin(a, axis=None, out=None, keepdims=False):
    return a.min(axis=axis, out=out, keepdims=keepdims)


def max(a, axis=None, out=None, keepdims=False):
    return amax(a, axis=axis, out=out, keepdims=keepdims)


@add_boilerplate("a", "b")
def maximum(a, b, out=None, where=True, dtype=None, **kwargs):
    return ndarray.perform_binary_op(
        BinaryOpCode.MAXIMUM,
        a,
        b,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
        where=where,
    )


def min(a, axis=None, out=None, keepdims=False):
    return amin(a, axis=axis, out=out, keepdims=keepdims)


@add_boilerplate("a", "b")
def minimum(a, b, out=None, where=True, dtype=None, **kwargs):
    return ndarray.perform_binary_op(
        BinaryOpCode.MINIMUM,
        a,
        b,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
        where=where,
    )


@add_boilerplate("a", "b")
def mod(a, b, out=None, where=True, dtype=None):
    return ndarray.perform_binary_op(
        BinaryOpCode.MOD,
        a,
        b,
        out=out,
        dtype=dtype,
        out_dtype=dtype,
        where=where,
    )


# Averages and variances


@add_boilerplate("a")
def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    return a.mean(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
