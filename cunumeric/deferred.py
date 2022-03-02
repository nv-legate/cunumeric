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

import weakref
from collections.abc import Iterable
from functools import reduce
from itertools import product

import numpy as np

import legate.core.types as ty
from legate.core import Future, ReductionOp, Store

from .config import (
    BinaryOpCode,
    CuNumericOpCode,
    CuNumericRedopCode,
    RandGenCode,
    UnaryOpCode,
    UnaryRedCode,
)
from .linalg.cholesky import cholesky
from .thunk import NumPyThunk
from .utils import get_arg_value_dtype


def _complex_field_dtype(dtype):
    if dtype == np.complex64:
        return np.dtype(np.float32)
    elif dtype == np.complex128:
        return np.dtype(np.float64)
    elif dtype == np.complex256:
        return np.dtype(np.float128)
    else:
        assert False


def _prod(tpl):
    return reduce(lambda a, b: a * b, tpl, 1)


def auto_convert(indices, keys=[]):
    indices = set(indices)

    def decorator(func):
        def wrapper(*args, **kwargs):
            self = args[0]

            args = tuple(
                self.runtime.to_deferred_array(arg) if idx in indices else arg
                for (idx, arg) in enumerate(args)
            )
            for key in keys:
                v = kwargs.get(key, None)
                if v is None:
                    continue
                v = self.runtime.to_deferred_array(v)
                kwargs[key] = v

            return func(*args, **kwargs)

        return wrapper

    return decorator


# This is a dummy object that is only used as an initializer for the
# RegionField object above. It is thrown away as soon as the
# RegionField is constructed.
class _CuNumericNDarray(object):
    __slots__ = ["__array_interface__"]

    def __init__(self, shape, field_type, base_ptr, strides, read_only):
        # See: https://docs.scipy.org/doc/numpy/reference/arrays.interface.html
        self.__array_interface__ = {
            "version": 3,
            "shape": shape,
            "typestr": field_type.str,
            "data": (base_ptr, read_only),
            "strides": strides,
        }


_UNARY_RED_TO_REDUCTION_OPS = {
    UnaryRedCode.SUM: ReductionOp.ADD,
    UnaryRedCode.PROD: ReductionOp.MUL,
    UnaryRedCode.MAX: ReductionOp.MAX,
    UnaryRedCode.MIN: ReductionOp.MIN,
    UnaryRedCode.ARGMAX: CuNumericRedopCode.ARGMAX,
    UnaryRedCode.ARGMIN: CuNumericRedopCode.ARGMIN,
    UnaryRedCode.CONTAINS: ReductionOp.ADD,
    UnaryRedCode.COUNT_NONZERO: ReductionOp.ADD,
    UnaryRedCode.ALL: ReductionOp.MUL,
    UnaryRedCode.ANY: ReductionOp.ADD,
}


def max_identity(ty):
    if ty.kind == "i" or ty.kind == "u":
        return np.iinfo(ty).min
    elif ty.kind == "f":
        return np.finfo(ty).min
    elif ty.kind == "c":
        return max_identity(np.float64) + max_identity(np.float64) * 1j
    elif ty.kind == "b":
        return False
    else:
        raise ValueError(f"Unsupported dtype: {ty}")


def min_identity(ty):
    if ty.kind == "i" or ty.kind == "u":
        return np.iinfo(ty).max
    elif ty.kind == "f":
        return np.finfo(ty).max
    elif ty.kind == "c":
        return min_identity(np.float64) + min_identity(np.float64) * 1j
    elif ty.kind == "b":
        return True
    else:
        raise ValueError(f"Unsupported dtype: {ty}")


_UNARY_RED_IDENTITIES = {
    UnaryRedCode.SUM: lambda _: 0,
    UnaryRedCode.PROD: lambda _: 1,
    UnaryRedCode.MIN: min_identity,
    UnaryRedCode.MAX: max_identity,
    UnaryRedCode.ARGMAX: lambda ty: (np.iinfo(np.int64).min, max_identity(ty)),
    UnaryRedCode.ARGMIN: lambda ty: (np.iinfo(np.int64).min, min_identity(ty)),
    UnaryRedCode.CONTAINS: lambda _: False,
    UnaryRedCode.COUNT_NONZERO: lambda _: 0,
    UnaryRedCode.ALL: lambda _: True,
    UnaryRedCode.ANY: lambda _: False,
}


class DeferredArray(NumPyThunk):
    """This is a deferred thunk for describing NumPy computations.
    It is backed by either a Legion logical region or a Legion future
    for describing the result of a compuation.

    :meta private:
    """

    def __init__(self, runtime, base, dtype, numpy_array=None):
        NumPyThunk.__init__(self, runtime, dtype)
        assert base is not None
        assert isinstance(base, Store)
        self.base = base  # a Legate Store
        self.numpy_array = (
            None if numpy_array is None else weakref.ref(numpy_array)
        )

    def __str__(self):
        return f"DeferredArray(base: {self.base})"

    @property
    def storage(self):
        storage = self.base.storage
        if self.base.kind == Future:
            return storage
        else:
            return (storage.region, storage.field.field_id)

    @property
    def shape(self):
        return tuple(self.base.shape)

    @property
    def ndim(self):
        return len(self.shape)

    def _copy_if_overlapping(self, other):
        if not self.base.overlaps(other.base):
            return self
        copy = self.runtime.create_empty_thunk(
            self.shape, self.dtype, inputs=[self]
        )
        copy.copy(self, deep=True)
        return copy

    def __numpy_array__(self):
        if self.numpy_array is not None:
            result = self.numpy_array()
            if result is not None:
                return result
        elif self.size == 0:
            # Return an empty array with the right number of dimensions
            # and type
            return np.empty(shape=self.shape, dtype=self.dtype)

        if self.scalar:
            result = np.full(
                self.shape,
                self.get_scalar_array(),
                dtype=self.dtype,
            )
        else:
            alloc = self.base.get_inline_allocation(self.context)

            def construct_ndarray(shape, address, strides):
                initializer = _CuNumericNDarray(
                    shape, self.dtype, address, strides, False
                )
                return np.asarray(initializer)

            result = alloc.consume(construct_ndarray)

        self.numpy_array = weakref.ref(result)
        return result

    # TODO: We should return a view of the field instead of a copy
    def imag(self):
        result = self.runtime.create_empty_thunk(
            self.shape,
            dtype=_complex_field_dtype(self.dtype),
            inputs=[self],
        )

        result.unary_op(
            UnaryOpCode.IMAG,
            result.dtype,
            self,
            True,
            [],
        )

        return result

    # TODO: We should return a view of the field instead of a copy
    def real(self):
        result = self.runtime.create_empty_thunk(
            self.shape,
            dtype=_complex_field_dtype(self.dtype),
            inputs=[self],
        )

        result.unary_op(
            UnaryOpCode.REAL,
            result.dtype,
            self,
            True,
            [],
        )

        return result

    def conj(self):
        result = self.runtime.create_empty_thunk(
            self.shape,
            dtype=self.dtype,
            inputs=[self],
        )

        result.unary_op(
            UnaryOpCode.CONJ,
            result.dtype,
            self,
            True,
            [],
        )

        return result

    # Copy source array to the destination array
    @auto_convert([1])
    def copy(self, rhs, deep=False):
        if self.scalar and rhs.scalar:
            self.base.set_storage(rhs.base.storage)
            return
        self.unary_op(
            UnaryOpCode.COPY,
            rhs.dtype,
            rhs,
            True,
            [],
        )

    @property
    def scalar(self):
        return self.base.scalar

    def get_scalar_array(self):
        assert self.scalar
        buf = self.base.storage.get_buffer(self.dtype.itemsize)
        result = np.frombuffer(buf, dtype=self.dtype, count=1)
        return result.reshape(())

    def _create_indexing_array(self, key):
        # Convert everything into deferred arrays of int64
        if isinstance(key, tuple):
            tuple_of_arrays = ()
            for k in key:
                if not isinstance(k, NumPyThunk):
                    raise NotImplementedError(
                        "need support for mixed advanced indexing"
                    )
                tuple_of_arrays += (k,)
        else:
            assert isinstance(key, NumPyThunk)
            # Handle the boolean array case
            if key.dtype == np.bool:
                if key.ndim != self.ndim:
                    raise TypeError(
                        "Boolean advanced indexing dimension mismatch"
                    )
                # For boolean arrays do the non-zero operation to make
                # them into a normal indexing array
                tuple_of_arrays = key.nonzero()
            else:
                tuple_of_arrays = (key,)
        if len(tuple_of_arrays) != self.ndim:
            raise TypeError("Advanced indexing dimension mismatch")
        if self.ndim > 1:
            # Check that all the arrays can be broadcast together
            # Concatenate all the arrays into a single array
            raise NotImplementedError("need support for concatenating arrays")
        else:
            return self.runtime.to_deferred_array(tuple_of_arrays[0])

    @staticmethod
    def _unpack_ellipsis(key, ndim):
        num_ellipsis = sum(k is Ellipsis for k in key)
        num_newaxes = sum(k is np.newaxis for k in key)

        if num_ellipsis == 0:
            return key
        elif num_ellipsis > 1:
            raise ValueError("Only a single ellipsis must be present")

        free_dims = ndim - (len(key) - num_newaxes - num_ellipsis)
        to_replace = (slice(None),) * free_dims
        unpacked = ()
        for k in key:
            if k is Ellipsis:
                unpacked += to_replace
            else:
                unpacked += (k,)
        return unpacked

    def _get_view(self, key):
        key = self._unpack_ellipsis(key, self.ndim)
        store = self.base
        shift = 0
        for dim, k in enumerate(key):
            if k is np.newaxis:
                store = store.promote(dim + shift, 1)
            elif isinstance(k, slice):
                store = store.slice(dim + shift, k)
            elif np.isscalar(k):
                if k < 0:
                    k += store.shape[dim + shift]
                store = store.project(dim + shift, k)
                shift -= 1
            else:
                assert False

        return DeferredArray(
            self.runtime,
            base=store,
            dtype=self.dtype,
        )

    def _broadcast(self, shape):
        result = self.base
        diff = len(shape) - result.ndim
        for dim in range(diff):
            result = result.promote(dim, shape[dim])

        for dim in range(len(shape)):
            if result.shape[dim] != shape[dim]:
                assert result.shape[dim] == 1
                result = result.project(dim, 0).promote(dim, shape[dim])

        return result

    def get_item(self, key):
        # Check to see if this is advanced indexing or not
        if self._is_advanced_indexing(key):
            # Create the indexing array
            index_array = self._create_indexing_array(key)
            # Create a new array to be the result
            result = self.runtime.create_empty_thunk(
                index_array.base.shape,
                self.dtype,
                inputs=[self],
            )

            if self.ndim != index_array.ndim:
                raise NotImplementedError(
                    "need support for indirect partitioning"
                )

            copy = self.context.create_copy()

            copy.add_input(self.base)
            copy.add_source_indirect(index_array.base)
            copy.add_output(result.base)

            copy.add_alignment(index_array.base, result.base)

            copy.execute()
        else:
            result = self._get_view(key)

            if result.shape == ():
                input = result
                result = self.runtime.create_empty_thunk(
                    (), self.dtype, inputs=[self]
                )

                task = self.context.create_task(CuNumericOpCode.READ)
                task.add_input(input.base)
                task.add_output(result.base)

                task.execute()

        return result

    @auto_convert([2])
    def set_item(self, key, rhs):
        assert self.dtype == rhs.dtype
        # Check to see if this is advanced indexing or not
        if self._is_advanced_indexing(key):
            # Create the indexing array
            index_array = self._create_indexing_array(key)
            if index_array.shape != rhs.shape:
                raise ValueError(
                    "Advanced indexing array does not match source shape"
                )
            if self.ndim != index_array.ndim:
                raise NotImplementedError(
                    "need support for indirect partitioning"
                )

            copy = self.context.create_copy()

            copy.add_input(rhs.base)
            copy.add_target_indirect(index_array.base)
            copy.add_output(self.base)

            copy.add_alignment(index_array.base, rhs.base)

            copy.execute()

        else:
            view = self._get_view(key)

            if view.shape == ():
                # We're just writing a single value
                assert rhs.size == 1

                task = self.context.create_task(CuNumericOpCode.WRITE)
                # Since we pass the view with write discard privilege,
                # we should make sure that the mapper either creates a fresh
                # instance just for this one-element view or picks one of the
                # existing valid instances for the parent.
                task.add_output(view.base)
                task.add_input(rhs.base)
                task.execute()
            else:
                # In Python, any inplace update of form arr[key] op= value
                # goes through three steps: 1) __getitem__ fetching the object
                # for the key, 2) __iop__ for the update, and 3) __setitem__
                # to set the result back. In cuNumeric, the object we
                # return in step (1) is actually a subview to the array arr
                # through which we make updates in place, so after step (2) is
                # done, # the effect of inplace update is already reflected
                # to the arr. Therefore, we skip the copy to avoid redundant
                # copies if we know that we hit such a scenario.
                # TODO: We should make this work for the advanced indexing case
                if view.base == rhs.base:
                    return

                if view.base.overlaps(rhs.base):
                    rhs_copy = self.runtime.create_empty_thunk(
                        rhs.shape,
                        rhs.dtype,
                        inputs=[rhs],
                    )
                    rhs_copy.copy(rhs, deep=False)
                    rhs = rhs_copy

                view.copy(rhs, deep=False)

    def reshape(self, newshape, order):
        assert isinstance(newshape, Iterable)
        if order == "A":
            order = "C"

        if order != "C":
            # If we don't have a transform then we need to make a copy
            self.runtime.warn(
                "cuNumeric has not implemented reshape using Fortran-like "
                "index order and is falling back to canonical numpy. You may "
                "notice significantly decreased performance for this "
                "function call.",
                category=RuntimeWarning,
            )
            numpy_array = self.__numpy_array__()
            # Force a copy here because we know we can't build a view
            result_array = numpy_array.reshape(newshape, order=order).copy()
            result = self.runtime.get_numpy_thunk(result_array)

            return result

        if self.shape == newshape:
            return self

        # Find a combination of domain transformations to convert this store
        # to the new shape. First we identify a pair of subsets of the source
        # and target extents whose products are the same, and infer necessary
        # domain transformations to align the two. In case where the target
        # isn't a transformed view of the source, the data is copied. This
        # table summarizes five possible cases:
        #
        # +-------+---------+------+-----------------------------------+
        # |Source | Target  | Copy | Plan                              |
        # +-------+---------+------+-----------------------------------+
        # |(a,b,c)| (abc,)  | Yes  | Delinearize(tgt, (a,b,c)) <- src  |
        # +-------+---------+------+-----------------------------------+
        # |(abc,) | (a,b,c,)| No   | tgt = Delinearize(src, (a,b,c))   |
        # +-------+---------+------+-----------------------------------+
        # |(a,b)  | (c,d)   | Yes  | tmp = new store((ab,))            |
        # |       |         |      | Delinearize(tmp, (a,b)) <- src    |
        # |       |         |      | tgt = Delinearize(tmp, (c,d))     |
        # +-------+---------+------+-----------------------------------+
        # |(a,1)  | (a,)    | No   | tgt = Project(src, 0, 0)          |
        # +-------+---------+------+-----------------------------------+
        # |(a,)   | (a,1)   | No   | tgt = Promote(src, 0, 1)          |
        # +-------+---------+------+-----------------------------------+
        #
        # Update 9/22/2021: the non-affineness with delinearization leads
        # to non-contiguous subregions in several places, and thus we
        # decided to avoid using it and make copies instead. This means
        # the third case in the table above now leads to two copies, one from
        # the source to a 1-D temporary array and one from that temporary
        # to the target array. We expect that such reshaping requests are
        # infrequent enough that the extra copies are causing any noticeable
        # performance issues, but we will revisit this decision later once
        # we have enough evidence that that's not the case.

        in_dim = 0
        out_dim = 0

        in_shape = self.shape
        out_shape = newshape

        in_ndim = len(in_shape)
        out_ndim = len(out_shape)

        groups = []

        while in_dim < in_ndim and out_dim < out_ndim:
            prev_in_dim = in_dim
            prev_out_dim = out_dim

            in_prod = 1
            out_prod = 1

            while True:
                if in_prod < out_prod:
                    in_prod *= in_shape[in_dim]
                    in_dim += 1
                else:
                    out_prod *= out_shape[out_dim]
                    out_dim += 1
                if in_prod == out_prod:
                    if in_dim < in_ndim and in_shape[in_dim] == 1:
                        in_dim += 1
                    break

            in_group = in_shape[prev_in_dim:in_dim]
            out_group = out_shape[prev_out_dim:out_dim]
            groups.append((in_group, out_group))

        while in_dim < in_ndim:
            assert in_shape[in_dim] == 1
            groups.append(((1,), ()))
            in_dim += 1

        while out_dim < out_ndim:
            assert out_shape[out_dim] == 1
            groups.append(((), (1,)))
            out_dim += 1

        needs_linearization = any(len(src_g) > 1 for src_g, _ in groups)
        needs_delinearization = any(len(tgt_g) > 1 for _, tgt_g in groups)
        needs_copy = needs_linearization or needs_delinearization

        if needs_copy:
            tmp_shape = ()
            for src_g, tgt_g in groups:
                if len(src_g) > 1 and len(tgt_g) > 1:
                    tmp_shape += (_prod(tgt_g),)
                else:
                    tmp_shape += tgt_g

            result = self.runtime.create_empty_thunk(
                tmp_shape, dtype=self.dtype, inputs=[self]
            )

            src = self.base
            tgt = result.base

            src_dim = 0
            tgt_dim = 0
            for src_g, tgt_g in groups:
                diff = 1
                if src_g == tgt_g:
                    assert len(src_g) == 1
                elif len(src_g) == 0:
                    assert tgt_g == (1,)
                    src = src.promote(src_dim, 1)
                elif len(tgt_g) == 0:
                    assert src_g == (1,)
                    tgt = tgt.promote(tgt_dim, 1)
                elif len(src_g) == 1:
                    src = src.delinearize(src_dim, tgt_g)
                    diff = len(tgt_g)
                else:
                    tgt = tgt.delinearize(tgt_dim, src_g)
                    diff = len(src_g)

                src_dim += diff
                tgt_dim += diff

            assert src.shape == tgt.shape

            src_array = DeferredArray(self.runtime, src, self.dtype)
            tgt_array = DeferredArray(self.runtime, tgt, self.dtype)
            tgt_array.copy(src_array, deep=True)

            if needs_delinearization and needs_linearization:
                src = result.base
                src_dim = 0
                for src_g, tgt_g in groups:
                    if len(src_g) > 1 and len(tgt_g) > 1:
                        src = src.delinearize(src_dim, tgt_g)
                        src_dim += len(tgt_g)

                assert src.shape == newshape
                src_array = DeferredArray(self.runtime, src, self.dtype)
                result = self.runtime.create_empty_thunk(
                    newshape, dtype=self.dtype, inputs=[self]
                )
                result.copy(src_array, deep=True)

        else:
            src = self.base
            src_dim = 0
            for src_g, tgt_g in groups:
                diff = 1
                if src_g == tgt_g:
                    assert len(src_g) == 1
                elif len(src_g) == 0:
                    assert tgt_g == (1,)
                    src = src.promote(src_dim, 1)
                elif len(tgt_g) == 0:
                    assert src_g == (1,)
                    src = src.project(src_dim, 1)
                    diff = 0
                else:
                    # unreachable
                    assert False

                src_dim += diff

            result = DeferredArray(self.runtime, src, self.dtype)

        return result

    def squeeze(self, axis):
        result = self.base
        if axis is None:
            shift = 0
            for dim in range(self.ndim):
                if result.shape[dim + shift] == 1:
                    result = result.project(dim + shift, 0)
                    shift -= 1
        elif isinstance(axis, int):
            result = result.project(axis, 0)
        elif isinstance(axis, tuple):
            shift = 0
            for dim in axis:
                result = result.project(dim + shift, 0)
                shift -= 1
        else:
            raise TypeError(
                '"axis" argument for squeeze must be int-like or tuple-like'
            )
        return DeferredArray(self.runtime, result, self.dtype)

    def swapaxes(self, axis1, axis2):
        if self.size == 1 or axis1 == axis2:
            return self
        # Make a new deferred array object and swap the results
        assert axis1 < self.ndim and axis2 < self.ndim

        dims = list(range(self.ndim))
        dims[axis1], dims[axis2] = dims[axis2], dims[axis1]

        result = self.base.transpose(dims)
        result = DeferredArray(self.runtime, result, self.dtype)

        return result

    # Convert the source array to the destination array
    @auto_convert([1])
    def convert(self, rhs, warn=True):
        lhs_array = self
        rhs_array = rhs
        assert lhs_array.dtype != rhs_array.dtype

        if warn:
            self.runtime.warn(
                "cuNumeric performing implicit type conversion from "
                + str(rhs_array.dtype)
                + " to "
                + str(lhs_array.dtype),
                category=UserWarning,
            )

        lhs = lhs_array.base
        rhs = rhs_array.base

        task = self.context.create_task(CuNumericOpCode.CONVERT)
        task.add_output(lhs)
        task.add_input(rhs)
        task.add_dtype_arg(lhs_array.dtype)

        task.add_alignment(lhs, rhs)

        task.execute()

    @auto_convert([1, 2])
    def convolve(self, v, lhs, mode):
        input = self.base
        filter = v.base
        out = lhs.base

        task = self.context.create_task(CuNumericOpCode.CONVOLVE)

        offsets = (filter.shape + 1) // 2
        stencils = []
        for offset in offsets:
            stencils.append((-offset, 0, offset))
        stencils = list(product(*stencils))
        stencils.remove((0,) * self.ndim)

        p_out = task.declare_partition(out)
        p_input = task.declare_partition(input)
        p_stencils = []
        for _ in stencils:
            p_stencils.append(task.declare_partition(input, complete=False))

        task.add_output(out, partition=p_out)
        task.add_input(filter)
        task.add_input(input, partition=p_input)
        for p_stencil in p_stencils:
            task.add_input(input, partition=p_stencil)
        task.add_scalar_arg(self.shape, (ty.int64,))

        task.add_constraint(p_out == p_input)
        for stencil, p_stencil in zip(stencils, p_stencils):
            task.add_constraint(p_input + stencil <= p_stencil)
        task.add_broadcast(filter)

        task.execute()

    # Fill the cuNumeric array with the value in the numpy array
    def _fill(self, value):
        assert value.scalar

        if self.scalar:
            # Handle the 0D case special
            self.base.set_storage(value.storage)
        else:
            assert self.base is not None
            dtype = self.dtype
            argval = False
            # If this is a fill for an arg value, make sure to pass
            # the value dtype so that we get it packed correctly
            if dtype.kind == "V":
                dtype = get_arg_value_dtype(dtype)
                argval = True
            else:
                dtype = None

            task = self.context.create_task(CuNumericOpCode.FILL)
            task.add_output(self.base)
            task.add_input(value)
            task.add_scalar_arg(argval, bool)

            task.execute()

    def fill(self, numpy_array):
        assert isinstance(numpy_array, np.ndarray)
        assert numpy_array.size == 1
        assert self.dtype == numpy_array.dtype
        # Have to copy the numpy array because this launch is asynchronous
        # and we need to make sure the application doesn't mutate the value
        # so make a future result, this is immediate so no dependence
        value = self.runtime.create_scalar(numpy_array.data, self.dtype)
        store = self.context.create_store(
            self.dtype, shape=(1,), storage=value, optimize_scalar=True
        )
        self._fill(store)

    @auto_convert([1, 2])
    def dot(self, src1, src2):
        rhs1_array = src1
        rhs2_array = src2
        lhs_array = self

        if rhs1_array.ndim == 1 and rhs2_array.ndim == 1:
            # Vector dot product case
            assert lhs_array.size == 1
            assert rhs1_array.shape == rhs2_array.shape or (
                rhs1_array.size == 1 and rhs2_array.size == 1
            )

            if rhs1_array.dtype == np.float16:
                lhs_array = self.runtime.create_empty_thunk(
                    self.shape, np.dtype(np.float32), inputs=[self]
                )

            lhs_array.fill(np.array(0, dtype=lhs_array.dtype))

            task = self.context.create_task(CuNumericOpCode.DOT)
            task.add_reduction(lhs_array.base, ty.ReductionOp.ADD)
            task.add_input(rhs1_array.base)
            task.add_input(rhs2_array.base)

            task.add_alignment(rhs1_array.base, rhs2_array.base)

            task.execute()

            if rhs1_array.dtype == np.float16:
                self.convert(lhs_array, warn=False)

        elif (
            rhs1_array.ndim == 1
            and rhs2_array.ndim == 2
            or rhs1_array.ndim == 2
            and rhs2_array.ndim == 1
        ):
            # Matrix-vector or vector-matrix multiply
            assert lhs_array.ndim == 1

            left_matrix = rhs1_array.ndim == 2

            if left_matrix and rhs1_array.shape[0] == 1:
                rhs1_array = rhs1_array.get_item((0, slice(None)))
                lhs_array.dot(rhs1_array, rhs2_array)
                return
            elif not left_matrix and rhs2_array.shape[1] == 1:
                rhs2_array = rhs2_array.get_item((slice(None), 0))
                lhs_array.dot(rhs1_array, rhs2_array)
                return

            # If the inputs are 16-bit floats, we should use 32-bit float
            # for accumulation
            if rhs1_array.dtype == np.float16:
                lhs_array = self.runtime.create_empty_thunk(
                    self.shape, np.dtype(np.float32), inputs=[self]
                )

            # TODO: We should be able to do this in the core
            lhs_array.fill(np.array(0, dtype=lhs_array.dtype))

            if left_matrix:
                rhs1 = rhs1_array.base
                (m, n) = rhs1.shape
                rhs2_array = rhs2_array._copy_if_overlapping(
                    lhs_array,
                )
                rhs2 = rhs2_array.base.promote(0, m)
                lhs = lhs_array.base.promote(1, n)
            else:
                rhs2 = rhs2_array.base
                (m, n) = rhs2.shape
                rhs1_array = rhs1_array._copy_if_overlapping(
                    lhs_array,
                )
                rhs1 = rhs1_array.base.promote(1, n)
                lhs = lhs_array.base.promote(0, m)

            task = self.context.create_task(CuNumericOpCode.MATVECMUL)
            task.add_reduction(lhs, ReductionOp.ADD)
            task.add_input(rhs1)
            task.add_input(rhs2)
            task.add_scalar_arg(left_matrix, bool)

            task.add_alignment(lhs, rhs1)
            task.add_alignment(lhs, rhs2)

            task.execute()

            # If we used an accumulation buffer, we should copy the results
            # back to the lhs
            if rhs1_array.dtype == np.float16:
                self.convert(lhs_array, warn=False)

        elif rhs1_array.ndim == 2 and rhs2_array.ndim == 2:
            # Matrix-matrix multiply
            M = lhs_array.shape[0]
            N = lhs_array.shape[1]
            K = rhs1_array.shape[1]
            assert M == rhs1_array.shape[0]  # Check M
            assert N == rhs2_array.shape[1]  # Check N
            assert K == rhs2_array.shape[0]  # Check K

            if M == 1 and N == 1:
                rhs1_array = rhs1_array.get_item((0, slice(None)))
                rhs2_array = rhs2_array.get_item((slice(None), 0))
                lhs_array.dot(rhs1_array, rhs2_array)
                return

            if rhs1_array.dtype == np.float16:
                lhs_array = self.runtime.create_empty_thunk(
                    self.shape, np.dtype(np.float32), inputs=[self]
                )

            rhs1_array = rhs1_array._copy_if_overlapping(lhs_array)
            rhs2_array = rhs2_array._copy_if_overlapping(lhs_array)

            # TODO: We should be able to do this in the core
            lhs_array.fill(np.array(0, dtype=lhs_array.dtype))

            lhs = lhs_array.base.promote(1, K)
            rhs1 = rhs1_array.base.promote(2, N)
            rhs2 = rhs2_array.base.promote(0, M)

            task = self.context.create_task(CuNumericOpCode.MATMUL)
            task.add_reduction(lhs, ReductionOp.ADD)
            task.add_input(rhs1)
            task.add_input(rhs2)

            task.add_alignment(lhs, rhs1)
            task.add_alignment(lhs, rhs2)

            task.execute()

            # If we used an accumulation buffer, we should copy the results
            # back to the lhs
            if rhs1_array.dtype == np.float16:
                self.convert(lhs_array, warn=False)
        else:
            raise NotImplementedError(
                f"dot between {rhs1_array.ndim}d and {rhs2_array.ndim}d arrays"
            )

    @auto_convert([2, 4])
    def contract(
        self,
        lhs_modes,
        rhs1_thunk,
        rhs1_modes,
        rhs2_thunk,
        rhs2_modes,
        mode2extent,
    ):
        lhs_thunk = self

        # TODO: More sanity checks (no duplicate modes, no singleton modes, no
        # broadcasting, ...)

        # Casting should have been handled by the frontend
        assert lhs_thunk.dtype is rhs1_thunk.dtype
        assert lhs_thunk.dtype is rhs2_thunk.dtype

        # Handle store overlap
        rhs1_thunk = rhs1_thunk._copy_if_overlapping(lhs_thunk)
        rhs2_thunk = rhs2_thunk._copy_if_overlapping(lhs_thunk)

        # Clear output array
        # TODO: We should be able to do this in the core
        lhs_thunk.fill(np.array(0, dtype=lhs_thunk.dtype))

        lhs = lhs_thunk.base
        rhs1 = rhs1_thunk.base
        rhs2 = rhs2_thunk.base

        # The underlying libraries are not guaranteed to work with stride
        # values of 0. The frontend should therefore handle broadcasting
        # directly, instead of promoting stores.
        assert not lhs.has_fake_dims()
        assert not rhs1.has_fake_dims()
        assert not rhs2.has_fake_dims()

        # Transpose arrays according to alphabetical order of mode labels
        def alphabetical_transpose(store, modes):
            perm = [dim for (_, dim) in sorted(zip(modes, range(len(modes))))]
            return store.transpose(perm)

        lhs = alphabetical_transpose(lhs, lhs_modes)
        rhs1 = alphabetical_transpose(rhs1, rhs1_modes)
        rhs2 = alphabetical_transpose(rhs2, rhs2_modes)

        # Promote dimensions as required to align the stores
        lhs_dim_mask = []
        rhs1_dim_mask = []
        rhs2_dim_mask = []
        for (dim, mode) in enumerate(sorted(mode2extent.keys())):
            extent = mode2extent[mode]

            def add_mode(store, modes, dim_mask):
                if mode not in modes:
                    dim_mask.append(False)
                    return store.promote(dim, extent)
                else:
                    dim_mask.append(True)
                    # Broadcasting should have been handled already
                    assert store.shape[dim] == extent
                    return store

            lhs = add_mode(lhs, lhs_modes, lhs_dim_mask)
            rhs1 = add_mode(rhs1, rhs1_modes, rhs1_dim_mask)
            rhs2 = add_mode(rhs2, rhs2_modes, rhs2_dim_mask)
        assert lhs.shape == rhs1.shape
        assert lhs.shape == rhs2.shape

        # Prepare the launch
        task = self.context.create_task(CuNumericOpCode.CONTRACT)
        task.add_reduction(lhs, ReductionOp.ADD)
        task.add_input(rhs1)
        task.add_input(rhs2)
        task.add_scalar_arg(tuple(lhs_dim_mask), (bool,))
        task.add_scalar_arg(tuple(rhs1_dim_mask), (bool,))
        task.add_scalar_arg(tuple(rhs2_dim_mask), (bool,))
        task.add_alignment(lhs, rhs1)
        task.add_alignment(lhs, rhs2)
        task.execute()

    # Create array from input array and indices
    def choose(self, *args, rhs):
        # convert all arrays to deferred
        index_arr = self.runtime.to_deferred_array(rhs)
        ch_def = tuple(self.runtime.to_deferred_array(c) for c in args)

        out_arr = self.base
        # broadcast input array and all choices arrays to the same shape
        index_arr = index_arr._broadcast(out_arr.shape)
        ch_tuple = tuple(c._broadcast(out_arr.shape) for c in ch_def)

        task = self.context.create_task(CuNumericOpCode.CHOOSE)
        task.add_output(out_arr)
        task.add_input(index_arr)
        for c in ch_tuple:
            task.add_input(c)

        task.add_alignment(index_arr, out_arr)
        for c in ch_tuple:
            task.add_alignment(index_arr, c)
        task.execute()

    # Create or extract a diagonal from a matrix
    @auto_convert([1])
    def _diag_helper(
        self,
        rhs,
        offset,
        naxes,
        extract,
    ):
        # fill output array with 0
        self.fill(np.array(0, dtype=self.dtype))
        if extract:
            diag = self.base
            matrix = rhs.base
            ndim = rhs.ndim
            start = matrix.ndim - naxes
            n = ndim - 1
            if naxes == 2:
                # get slice of the original array by the offset
                if offset > 0:
                    matrix = matrix.slice(start + 1, slice(offset, None))
                if matrix.shape[n - 1] < matrix.shape[n]:
                    diag = diag.promote(start + 1, matrix.shape[ndim - 1])
                else:
                    diag = diag.promote(start, matrix.shape[ndim - 2])
            else:
                # promote output to the shape of the input  array
                for i in range(1, naxes):
                    diag = diag.promote(start, matrix.shape[-i - 1])
        else:
            matrix = self.base
            diag = rhs.base
            ndim = self.ndim
            # get slice of the original array by the offset
            if offset > 0:
                matrix = matrix.slice(1, slice(offset, None))
            elif offset < 0:
                matrix = matrix.slice(0, slice(-offset, None))

            if matrix.shape[0] < matrix.shape[1]:
                diag = diag.promote(1, matrix.shape[1])
            else:
                diag = diag.promote(0, matrix.shape[0])

        task = self.context.create_task(CuNumericOpCode.DIAG)

        if extract:
            task.add_reduction(diag, ReductionOp.ADD)
            task.add_input(matrix)
            task.add_alignment(matrix, diag)
        else:
            task.add_output(matrix)
            task.add_input(diag)
            task.add_input(matrix)
            task.add_alignment(diag, matrix)

        task.add_scalar_arg(naxes, ty.int32)
        task.add_scalar_arg(extract, bool)

        task.execute()

    # Create an identity array with the ones offset from the diagonal by k
    def eye(self, k):
        assert self.ndim == 2  # Only 2-D arrays should be here
        # First issue a fill to zero everything out
        self.fill(np.array(0, dtype=self.dtype))

        task = self.context.create_task(CuNumericOpCode.EYE)
        task.add_output(self.base)
        task.add_scalar_arg(k, ty.int32)

        task.execute()

    def arange(self, start, stop, step):
        assert self.ndim == 1  # Only 1-D arrays should be here
        if self.scalar:
            # Handle the special case of a single value here
            assert self.shape[0] == 1
            array = np.array(start, dtype=self.dtype)
            future = self.runtime.create_scalar(array.data, array.dtype)
            self.base.set_storage(future)
            return

        def create_scalar(value, dtype):
            array = np.array(value, dtype)
            return self.runtime.create_scalar(
                array.data,
                array.dtype,
                shape=(1,),
                wrap=True,
            ).base

        task = self.context.create_task(CuNumericOpCode.ARANGE)
        task.add_output(self.base)
        task.add_input(create_scalar(start, self.dtype))
        task.add_input(create_scalar(stop, self.dtype))
        task.add_input(create_scalar(step, self.dtype))

        task.execute()

    # Tile the src array onto the destination array
    @auto_convert([1])
    def tile(self, rhs, reps):
        src_array = rhs
        dst_array = self
        assert src_array.ndim <= dst_array.ndim
        assert src_array.dtype == dst_array.dtype
        if src_array.scalar:
            self._fill(src_array.base)
            return

        task = self.context.create_task(CuNumericOpCode.TILE)

        task.add_output(self.base)
        task.add_input(rhs.base)

        task.add_broadcast(rhs.base)

        task.execute()

    # Transpose the matrix dimensions
    @auto_convert([1])
    def transpose(self, rhs, axes):
        rhs_array = rhs
        lhs_array = self
        assert lhs_array.dtype == rhs_array.dtype
        assert lhs_array.ndim == rhs_array.ndim
        assert lhs_array.ndim == len(axes)
        lhs_array.base = rhs_array.base.transpose(axes)

    @auto_convert([1])
    def trilu(self, rhs, k, lower):
        lhs = self.base
        rhs = rhs._broadcast(lhs.shape)

        task = self.context.create_task(CuNumericOpCode.TRILU)

        task.add_output(lhs)
        task.add_input(rhs)
        task.add_scalar_arg(lower, bool)
        task.add_scalar_arg(k, ty.int32)

        task.add_alignment(lhs, rhs)

        task.execute()

    @auto_convert([1])
    def flip(self, rhs, axes):
        input = rhs.base
        output = self.base

        if axes is None:
            axes = list(range(self.ndim))
        elif not isinstance(axes, tuple):
            axes = (axes,)

        task = self.context.create_task(CuNumericOpCode.FLIP)
        task.add_output(output)
        task.add_input(input)
        task.add_scalar_arg(axes, (ty.int32,))

        task.add_broadcast(input)
        task.add_alignment(input, output)

        task.execute()

    # Perform a bin count operation on the array
    @auto_convert([1], ["weights"])
    def bincount(self, rhs, weights=None):
        weight_array = weights
        src_array = rhs
        dst_array = self
        assert src_array.size > 1
        assert dst_array.ndim == 1
        if weight_array is not None:
            assert src_array.shape == weight_array.shape or (
                src_array.size == 1 and weight_array.size == 1
            )
        else:
            weight_array = self.runtime.create_scalar(
                np.array(1, dtype=np.int64),
                np.dtype(np.int64),
                shape=(),
                wrap=True,
            )

        dst_array.fill(np.array(0, dst_array.dtype))

        task = self.context.create_task(CuNumericOpCode.BINCOUNT)
        task.add_reduction(dst_array.base, ReductionOp.ADD)
        task.add_input(src_array.base)
        task.add_input(weight_array.base)

        task.add_broadcast(dst_array.base)
        if not weight_array.scalar:
            task.add_alignment(src_array.base, weight_array.base)

        task.execute()

    def nonzero(self):
        results = tuple(
            self.runtime.create_unbound_thunk(np.dtype(np.int64))
            for _ in range(self.ndim)
        )

        task = self.context.create_task(CuNumericOpCode.NONZERO)

        task.add_input(self.base)
        for result in results:
            task.add_output(result.base)

        task.execute()
        return results

    def random(self, gen_code, args=[]):
        task = self.context.create_task(CuNumericOpCode.RAND)

        task.add_output(self.base)
        task.add_scalar_arg(gen_code.value, ty.int32)
        epoch = self.runtime.get_next_random_epoch()
        task.add_scalar_arg(epoch, ty.uint32)
        task.add_scalar_arg(self.compute_strides(self.shape), (ty.int64,))
        self.add_arguments(task, args)

        task.execute()

    def random_uniform(self):
        assert self.dtype == np.float64
        self.random(RandGenCode.UNIFORM)

    def random_normal(self):
        assert self.dtype == np.float64
        self.random(RandGenCode.NORMAL)

    def random_integer(self, low, high):
        assert self.dtype.kind == "i"
        low = np.array(low, self.dtype)
        high = np.array(high, self.dtype)
        self.random(RandGenCode.INTEGER, [low, high])

    # Perform the unary operation and put the result in the array
    @auto_convert([3])
    def unary_op(self, op, op_dtype, src, where, args):
        lhs = self.base
        rhs = src._broadcast(lhs.shape)

        task = self.context.create_task(CuNumericOpCode.UNARY_OP)
        task.add_output(lhs)
        task.add_input(rhs)
        task.add_scalar_arg(op.value, ty.int32)
        self.add_arguments(task, args)

        task.add_alignment(lhs, rhs)

        task.execute()

    # Perform a unary reduction operation from one set of dimensions down to
    # fewer
    @auto_convert([2])
    def unary_reduction(
        self,
        op,
        src,
        where,
        axes,
        keepdims,
        args,
        initial,
    ):
        lhs_array = self
        rhs_array = src
        assert lhs_array.ndim <= rhs_array.ndim

        # See if we are doing reduction to a point or another region
        if lhs_array.size == 1:
            assert axes is None or len(axes) == (
                rhs_array.ndim - lhs_array.ndim
            )

            task = self.context.create_task(CuNumericOpCode.SCALAR_UNARY_RED)

            fill_value = _UNARY_RED_IDENTITIES[op](rhs_array.dtype)

            lhs_array.fill(np.array(fill_value, dtype=lhs_array.dtype))

            task.add_reduction(lhs_array.base, _UNARY_RED_TO_REDUCTION_OPS[op])
            task.add_input(rhs_array.base)
            task.add_scalar_arg(op, ty.int32)

            self.add_arguments(task, args)

            task.execute()

        else:
            argred = op in (UnaryRedCode.ARGMAX, UnaryRedCode.ARGMIN)

            if argred:
                argred_dtype = self.runtime.get_arg_dtype(rhs_array.dtype)
                lhs_array = self.runtime.create_empty_thunk(
                    lhs_array.shape,
                    dtype=argred_dtype,
                    inputs=[self],
                )

            # Before we perform region reduction, make sure to have the lhs
            # initialized. If an initial value is given, we use it, otherwise
            # we use the identity of the reduction operator
            if initial is not None:
                assert not argred
                fill_value = initial
            else:
                fill_value = _UNARY_RED_IDENTITIES[op](rhs_array.dtype)
            lhs_array.fill(np.array(fill_value, lhs_array.dtype))

            # If output dims is not 0, then we must have axes
            assert axes is not None
            # Reduction to a smaller array
            result = lhs_array.base
            if keepdims:
                for axis in axes:
                    result = result.project(axis, 0)
            for axis in axes:
                result = result.promote(axis, rhs_array.shape[axis])
            # Iterate over all the dimension(s) being collapsed and build a
            # temporary field that we will reduce down to the final value
            if len(axes) > 1:
                raise NotImplementedError(
                    "Need support for reducing multiple dimensions"
                )

            task = self.context.create_task(CuNumericOpCode.UNARY_RED)

            task.add_input(rhs_array.base)
            task.add_reduction(result, _UNARY_RED_TO_REDUCTION_OPS[op])
            task.add_scalar_arg(axis, ty.int32)
            task.add_scalar_arg(op, ty.int32)

            self.add_arguments(task, args)

            task.add_alignment(result, rhs_array.base)

            task.execute()

            if argred:
                self.unary_op(
                    UnaryOpCode.GETARG,
                    self.dtype,
                    lhs_array,
                    True,
                    [],
                )

    # Perform the binary operation and put the result in the lhs array
    @auto_convert([2, 3])
    def binary_op(self, op_code, src1, src2, where, args):
        lhs = self.base
        rhs1 = src1._broadcast(lhs.shape)
        rhs2 = src2._broadcast(lhs.shape)

        # Populate the Legate launcher
        task = self.context.create_task(CuNumericOpCode.BINARY_OP)
        task.add_output(lhs)
        task.add_input(rhs1)
        task.add_input(rhs2)
        task.add_scalar_arg(op_code.value, ty.int32)
        self.add_arguments(task, args)

        task.add_alignment(lhs, rhs1)
        task.add_alignment(lhs, rhs2)

        task.execute()

    @auto_convert([2, 3])
    def binary_reduction(self, op, src1, src2, broadcast, args):
        lhs = self.base
        rhs1 = src1.base
        rhs2 = src2.base
        assert lhs.scalar

        if broadcast is not None:
            rhs1 = rhs1._broadcast(broadcast)
            rhs2 = rhs2._broadcast(broadcast)

        # Populate the Legate launcher
        if op == BinaryOpCode.NOT_EQUAL:
            redop = ReductionOp.ADD
            self.fill(np.array(False))
        else:
            redop = ReductionOp.MUL
            self.fill(np.array(True))
        task = self.context.create_task(CuNumericOpCode.BINARY_RED)
        task.add_reduction(lhs, redop)
        task.add_input(rhs1)
        task.add_input(rhs2)
        task.add_scalar_arg(op.value, ty.int32)
        self.add_arguments(task, args)

        task.add_alignment(rhs1, rhs2)

        task.execute()

    @auto_convert([1, 2, 3])
    def where(self, src1, src2, src3):
        lhs = self.base
        rhs1 = src1._broadcast(lhs.shape)
        rhs2 = src2._broadcast(lhs.shape)
        rhs3 = src3._broadcast(lhs.shape)

        # Populate the Legate launcher
        task = self.context.create_task(CuNumericOpCode.WHERE)
        task.add_output(lhs)
        task.add_input(rhs1)
        task.add_input(rhs2)
        task.add_input(rhs3)

        task.add_alignment(lhs, rhs1)
        task.add_alignment(lhs, rhs2)
        task.add_alignment(lhs, rhs3)

        task.execute()

    # A helper method for attaching arguments
    def add_arguments(self, task, args):
        if args is None:
            return
        for numpy_array in args:
            assert numpy_array.size == 1
            scalar = self.runtime.create_scalar(
                numpy_array.data,
                numpy_array.dtype,
                shape=(1,),
                wrap=True,
            )
            task.add_input(scalar.base)

    @staticmethod
    def compute_strides(shape):
        stride = 1
        result = ()
        for dim in reversed(shape):
            result = (stride,) + result
            stride *= dim
        return result

    @auto_convert([1])
    def cholesky(self, src, no_tril=False):
        cholesky(self, src)
        if not no_tril:
            self.trilu(self, 0, True)
