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
from collections import Counter
from collections.abc import Iterable
from enum import IntEnum, unique
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
from .sort import sort
from .thunk import NumPyThunk
from .utils import get_arg_value_dtype, is_advanced_indexing


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
        return np.finfo(np.float64).min + np.finfo(np.float64).min * 1j
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
        return np.finfo(np.float64).max + np.finfo(np.float64).max * 1j
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


@unique
class BlasOperation(IntEnum):
    VV = 1
    MV = 2
    MM = 3


class DeferredArray(NumPyThunk):
    """This is a deferred thunk for describing NumPy computations.
    It is backed by either a Legion logical region or a Legion future
    for describing the result of a computation.

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

    def _zip_indices(self, start_index, arrays):
        if not isinstance(arrays, tuple):
            raise TypeError("zip_indices expects tuple of arrays")
        # start_index is the index from witch indices arrays are passed
        # for example of arr[:, indx, :], start_index =1
        if start_index == -1:
            start_index = 0

        new_arrays = tuple()
        # check array's type and convert them to deferred arrays
        for a in arrays:
            a = self.runtime.to_deferred_array(a)
            data_type = a.dtype
            if data_type != np.int64:
                raise TypeError("index arrays should be int64 type")
            new_arrays += (a,)
        arrays = new_arrays

        # find a broadcasted shape for all arrays passed as indices
        shapes = tuple(a.shape for a in arrays)
        if len(arrays) > 1:
            # TODO: replace with cunumeric.broadcast_shapes, when available
            b_shape = np.broadcast_shapes(*shapes)
        else:
            b_shape = arrays[0].shape

        # key dim - dimension of indices arrays
        key_dim = len(b_shape)
        out_shape = b_shape

        # broadcast shapes
        new_arrays = tuple()
        for a in arrays:
            if a.shape != b_shape:
                new_arrays += (a._broadcast(b_shape),)
            else:
                new_arrays += (a.base,)
        arrays = new_arrays

        if len(arrays) < self.ndim:
            # the case when # of arrays passed is smaller than dimension of
            # the input array
            N = len(arrays)
            # output shape
            out_shape = (
                tuple(self.shape[i] for i in range(0, start_index))
                + b_shape
                + tuple(
                    self.shape[i] for i in range(start_index + N, self.ndim)
                )
            )
            new_arrays = tuple()
            # promote all index arrays to have the same shape as output
            for a in arrays:
                for i in range(0, start_index):
                    a = a.promote(i, self.shape[i])
                for i in range(start_index + N, self.ndim):
                    a = a.promote(key_dim + i - N, self.shape[i])
                new_arrays += (a,)
            arrays = new_arrays
        elif len(arrays) > self.ndim:
            raise ValueError("wrong number of index arrays passed")

        # create output array which will store Point<N> field where
        # N is number of index arrays
        # shape of the output array should be the same as the shape of each
        # index array
        # NOTE: We need to instantiate a RegionField of non-primitive
        # dtype, to store N-dimensional index points, to be used as the
        # indirection field in a copy.
        # Such dtypes are technically not supported,
        # but it should be safe to directly create a DeferredArray
        # of that dtype, so long as we don't try to convert it to a
        # NumPy array.
        N = self.ndim
        pointN_dtype = self.runtime.get_point_type(N)
        store = self.context.create_store(
            pointN_dtype, shape=out_shape, optimize_scalar=True
        )
        output_arr = DeferredArray(
            self.runtime, base=store, dtype=pointN_dtype
        )

        # call ZIP function to combine index arrays into a singe array
        task = self.context.create_task(CuNumericOpCode.ZIP)
        task.add_output(output_arr.base)
        task.add_scalar_arg(self.ndim, ty.int64)  # N of points in Point<N>
        task.add_scalar_arg(key_dim, ty.int64)  # key_dim
        task.add_scalar_arg(start_index, ty.int64)  # start_index
        task.add_scalar_arg(self.shape, (ty.int64,))
        for a in arrays:
            task.add_input(a)
            task.add_alignment(output_arr.base, a)
        task.execute()

        return output_arr

    def _copy_store(self, store):
        store_to_copy = DeferredArray(
            self.runtime,
            base=store,
            dtype=self.dtype,
        )
        store_copy = self.runtime.create_empty_thunk(
            store_to_copy.shape,
            self.dtype,
            inputs=[store_to_copy],
        )
        store_copy.copy(store_to_copy, deep=True)
        return store_copy, store_copy.base

    def _create_indexing_array(self, key, is_set=False):
        store = self.base
        rhs = self
        # the index where the first index_array is passed to the [] operator
        start_index = -1
        if (
            isinstance(key, NumPyThunk)
            and key.dtype == bool
            and key.shape == rhs.shape
        ):
            if not isinstance(key, DeferredArray):
                key = self.runtime.to_deferred_array(key)

            out_dtype = rhs.dtype
            # in cease this operation is called for the set_item, we
            # return Point<N> type field that is later used for
            # indirect copy operation
            if is_set:
                N = rhs.ndim
                out_dtype = rhs.runtime.get_point_type(N)

            out = rhs.runtime.create_unbound_thunk(out_dtype)
            task = rhs.context.create_task(CuNumericOpCode.ADVANCED_INDEXING)
            task.add_output(out.base)
            task.add_input(rhs.base)
            task.add_input(key.base)
            task.add_scalar_arg(is_set, bool)
            task.add_alignment(rhs.base, key.base)
            task.add_broadcast(
                key.base, axes=tuple(range(1, len(key.base.shape)))
            )
            task.add_broadcast(
                rhs.base, axes=tuple(range(1, len(rhs.base.shape)))
            )
            task.execute()
            return False, rhs, out, self

        if isinstance(key, NumPyThunk):
            key = (key,)

        assert isinstance(key, tuple)
        key = self._unpack_ellipsis(key, self.ndim)
        shift = 0
        last_index = self.ndim
        # in case when index arrays are passed in the scaterred way,
        # we need to transpose original array so all index arrays
        # are close to each other
        transpose_needed = False
        transpose_indices = tuple()
        key_transpose_indices = tuple()
        tuple_of_arrays = ()

        # First, we need to check if transpose is needed
        for dim, k in enumerate(key):
            if np.isscalar(k) or isinstance(k, NumPyThunk):
                if start_index == -1:
                    start_index = dim
                key_transpose_indices += (dim,)
                transpose_needed = transpose_needed or ((dim - last_index) > 1)
                if (
                    isinstance(k, NumPyThunk)
                    and k.dtype == bool
                    and k.ndim >= 2
                ):
                    for i in range(dim, dim + k.ndim):
                        transpose_indices += (shift + i,)
                    shift += k.ndim - 1
                else:
                    transpose_indices += ((dim + shift),)
                last_index = dim

        if transpose_needed:
            start_index = 0
            post_indices = tuple(
                i for i in range(store.ndim) if i not in transpose_indices
            )
            transpose_indices += post_indices
            post_indices = tuple(
                i for i in range(len(key)) if i not in key_transpose_indices
            )
            key_transpose_indices += post_indices
            store = store.transpose(transpose_indices)
            key = tuple(key[i] for i in key_transpose_indices)

        shift = 0
        for dim, k in enumerate(key):
            if np.isscalar(k):
                if k < 0:
                    k += store.shape[dim + shift]
                store = store.project(dim + shift, k)
                shift -= 1
            elif k is np.newaxis:
                store = store.promote(dim + shift, 1)
            elif isinstance(k, slice):
                store = store.slice(dim + shift, k)
            elif isinstance(k, NumPyThunk):
                if not isinstance(key, DeferredArray):
                    k = self.runtime.to_deferred_array(k)
                if k.dtype == bool:
                    for i in range(k.ndim):
                        if k.shape[i] != store.shape[dim + i + shift]:
                            raise ValueError(
                                "shape of boolean index did not match "
                                "indexed array "
                            )
                    # in case of the mixed indises we all nonzero
                    # for the bool array
                    k = k.nonzero()
                    shift += len(k) - 1
                    tuple_of_arrays += k
                else:
                    tuple_of_arrays += (k,)
            else:
                raise TypeError(
                    "Unsupported entry type passed to advanced ",
                    "indexing operation",
                )
        if store.transformed:
            # in the case this operation is called for the set_item, we need
            # to apply all the transformations done to `store` to `self`
            # as well before creating a copy
            if is_set:
                self = DeferredArray(self.runtime, store, self.dtype)
            # after store is transformed we need to to return a copy of
            # the store since Copy operation can't be done on
            # the store with transformation
            rhs, store = self._copy_store(store)

        if len(tuple_of_arrays) <= rhs.ndim:
            output_arr = rhs._zip_indices(start_index, tuple_of_arrays)
            return True, rhs, output_arr, self
        else:
            raise ValueError("Advanced indexing dimension mismatch")

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
        if is_advanced_indexing(key):
            # Create the indexing array
            (
                copy_needed,
                rhs,
                index_array,
                self,
            ) = self._create_indexing_array(key)
            store = rhs.base
            if copy_needed:
                # Create a new array to be the result
                result = self.runtime.create_empty_thunk(
                    index_array.base.shape,
                    self.dtype,
                    inputs=[self],
                )
                copy = self.context.create_copy()

                copy.add_input(store)
                copy.add_source_indirect(index_array.base)
                copy.add_output(result.base)

                copy.execute()
            else:
                return index_array

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
        if is_advanced_indexing(key):
            # Create the indexing array
            (
                copy_needed,
                lhs,
                index_array,
                self,
            ) = self._create_indexing_array(key, True)

            if rhs.shape != index_array.shape:
                rhs_tmp = rhs._broadcast(index_array.base.shape)
                rhs_tmp, rhs = rhs._copy_store(rhs_tmp)
            else:
                if rhs.base.transformed:
                    rhs, rhs_base = rhs._copy_store(rhs.base)
                rhs = rhs.base

            copy = self.context.create_copy()
            copy.add_input(rhs)
            copy.add_target_indirect(index_array.base)
            copy.add_output(lhs.base)
            copy.execute()

            # TODO this copy will be removed when affine copies are
            # supported in Legion/Realm
            if lhs is not self:
                self.copy(lhs, deep=True)

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
                    src = src.project(src_dim, 0)
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

    @auto_convert([1])
    def fft(self, lhs, axes, kind, direction):
        # For now, deferred only supported with GPU, use eager / numpy for CPU
        if self.runtime.num_gpus == 0:
            lhs_eager = lhs.runtime.to_eager_array(lhs)
            self.runtime.to_eager_array(self).fft(
                lhs_eager, axes, kind, direction
            )
            lhs.base = lhs.runtime.to_deferred_array(lhs_eager).base
        else:
            input = self.base
            output = lhs.base

            task = self.context.create_task(CuNumericOpCode.FFT)
            p_output = task.declare_partition(output)
            p_input = task.declare_partition(input)

            task.add_output(output, partition=p_output)
            task.add_input(input, partition=p_input)
            task.add_scalar_arg(kind.type_id, ty.int32)
            task.add_scalar_arg(direction.value, ty.int32)
            task.add_scalar_arg(
                len(set(axes)) != len(axes)
                or len(axes) != input.ndim
                or tuple(axes) != tuple(sorted(axes)),
                ty.int8,
            )
            for ax in axes:
                task.add_scalar_arg(ax, ty.int64)

            task.add_broadcast(input)
            task.add_constraint(p_output == p_input)

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
        supported_dtypes = [
            np.float16,
            np.float32,
            np.float64,
            np.complex64,
            np.complex128,
        ]
        lhs_thunk = self

        # Sanity checks
        # no duplicate modes within an array
        assert len(lhs_modes) == len(set(lhs_modes))
        assert len(rhs1_modes) == len(set(rhs1_modes))
        assert len(rhs2_modes) == len(set(rhs2_modes))
        # no singleton modes
        mode_counts = Counter()
        mode_counts.update(lhs_modes)
        mode_counts.update(rhs1_modes)
        mode_counts.update(rhs2_modes)
        for count in mode_counts.values():
            assert count == 2 or count == 3
        # arrays and mode lists agree on dimensionality
        assert lhs_thunk.ndim == len(lhs_modes)
        assert rhs1_thunk.ndim == len(rhs1_modes)
        assert rhs2_thunk.ndim == len(rhs2_modes)
        # array shapes agree with mode extents (broadcasting should have been
        # handled by the frontend)
        assert all(
            mode2extent[mode] == dim_sz
            for (mode, dim_sz) in zip(
                lhs_modes + rhs1_modes + rhs2_modes,
                lhs_thunk.shape + rhs1_thunk.shape + rhs2_thunk.shape,
            )
        )
        # casting has been handled by the frontend
        assert lhs_thunk.dtype == rhs1_thunk.dtype
        assert lhs_thunk.dtype == rhs2_thunk.dtype

        # Handle store overlap
        rhs1_thunk = rhs1_thunk._copy_if_overlapping(lhs_thunk)
        rhs2_thunk = rhs2_thunk._copy_if_overlapping(lhs_thunk)

        # Test for special cases where we can use BLAS
        blas_op = None
        if any(c != 2 for c in mode_counts.values()):
            pass
        elif (
            len(lhs_modes) == 0
            and len(rhs1_modes) == 1
            and len(rhs2_modes) == 1
        ):
            # this case works for any arithmetic type, not just floats
            blas_op = BlasOperation.VV
        elif (
            lhs_thunk.dtype in supported_dtypes
            and len(lhs_modes) == 1
            and (
                len(rhs1_modes) == 2
                and len(rhs2_modes) == 1
                or len(rhs1_modes) == 1
                and len(rhs2_modes) == 2
            )
        ):
            blas_op = BlasOperation.MV
        elif (
            lhs_thunk.dtype in supported_dtypes
            and len(lhs_modes) == 2
            and len(rhs1_modes) == 2
            and len(rhs2_modes) == 2
        ):
            blas_op = BlasOperation.MM

        # Our half-precision BLAS tasks expect a single-precision accumulator.
        # This is done to avoid the precision loss that results from repeated
        # reductions into a half-precision accumulator, and to enable the use
        # of tensor cores. In the general-purpose tensor contraction case
        # below the tasks do this adjustment internally.
        if blas_op is not None and lhs_thunk.dtype == np.float16:
            lhs_thunk = self.runtime.create_empty_thunk(
                lhs_thunk.shape, np.dtype(np.float32), inputs=[lhs_thunk]
            )

        # Clear output array
        lhs_thunk.fill(np.array(0, dtype=lhs_thunk.dtype))

        # Pull out the stores
        lhs = lhs_thunk.base
        rhs1 = rhs1_thunk.base
        rhs2 = rhs2_thunk.base

        # The underlying libraries are not guaranteed to work with stride
        # values of 0. The frontend should therefore handle broadcasting
        # directly, instead of promoting stores.
        assert not lhs.has_fake_dims()
        assert not rhs1.has_fake_dims()
        assert not rhs2.has_fake_dims()

        # Special cases where we can use BLAS
        if blas_op is not None:

            if blas_op == BlasOperation.VV:
                # Vector dot product
                task = self.context.create_task(CuNumericOpCode.DOT)
                task.add_reduction(lhs, ReductionOp.ADD)
                task.add_input(rhs1)
                task.add_input(rhs2)
                task.add_alignment(rhs1, rhs2)
                task.execute()

            elif blas_op == BlasOperation.MV:
                # Matrix-vector or vector-matrix multiply

                # b,(ab/ba)->a --> (ab/ba),b->a
                if len(rhs1_modes) == 1:
                    rhs1, rhs2 = rhs2, rhs1
                    rhs1_modes, rhs2_modes = rhs2_modes, rhs1_modes
                # ba,b->a --> ab,b->a
                if rhs1_modes[0] == rhs2_modes[0]:
                    rhs1 = rhs1.transpose([1, 0])
                    rhs1_modes = [rhs1_modes[1], rhs1_modes[0]]

                (m, n) = rhs1.shape
                rhs2 = rhs2.promote(0, m)
                lhs = lhs.promote(1, n)

                task = self.context.create_task(CuNumericOpCode.MATVECMUL)
                task.add_reduction(lhs, ReductionOp.ADD)
                task.add_input(rhs1)
                task.add_input(rhs2)
                task.add_alignment(lhs, rhs1)
                task.add_alignment(lhs, rhs2)
                task.execute()

            elif blas_op == BlasOperation.MM:
                # Matrix-matrix multiply

                # (cb/bc),(ab/ba)->ac --> (ab/ba),(cb/bc)->ac
                if lhs_modes[0] not in rhs1_modes:
                    rhs1, rhs2 = rhs2, rhs1
                    rhs1_modes, rhs2_modes = rhs2_modes, rhs1_modes
                assert (
                    lhs_modes[0] in rhs1_modes and lhs_modes[1] in rhs2_modes
                )
                # ba,?->ac --> ab,?->ac
                if lhs_modes[0] != rhs1_modes[0]:
                    rhs1 = rhs1.transpose([1, 0])
                    rhs1_modes = [rhs1_modes[1], rhs1_modes[0]]
                # ?,cb->ac --> ?,bc->ac
                if lhs_modes[1] != rhs2_modes[1]:
                    rhs2 = rhs2.transpose([1, 0])
                    rhs2_modes = [rhs2_modes[1], rhs2_modes[0]]

                m = lhs.shape[0]
                n = lhs.shape[1]
                k = rhs1.shape[1]
                assert m == rhs1.shape[0]
                assert n == rhs2.shape[1]
                assert k == rhs2.shape[0]
                lhs = lhs.promote(1, k)
                rhs1 = rhs1.promote(2, n)
                rhs2 = rhs2.promote(0, m)

                task = self.context.create_task(CuNumericOpCode.MATMUL)
                task.add_reduction(lhs, ReductionOp.ADD)
                task.add_input(rhs1)
                task.add_input(rhs2)
                task.add_alignment(lhs, rhs1)
                task.add_alignment(lhs, rhs2)
                task.execute()

            else:
                assert False

            # If we used a single-precision intermediate accumulator, cast the
            # result back to half-precision.
            if rhs1_thunk.dtype == np.float16:
                self.convert(
                    lhs_thunk,
                    warn=False,
                )

            return

        # General-purpose contraction
        if lhs_thunk.dtype not in supported_dtypes:
            raise TypeError(f"Unsupported type: {lhs_thunk.dtype}")

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
        trace,
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
                if trace:
                    if matrix.ndim == 2:
                        diag = diag.promote(0, matrix.shape[0])
                        diag = diag.project(1, 0).promote(1, matrix.shape[1])
                    else:
                        for i in range(0, naxes):
                            diag = diag.promote(start, matrix.shape[-i - 1])
                else:
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

    # Repeat elements of an array.
    def repeat(self, repeats, axis, scalar_repeats):
        out = self.runtime.create_unbound_thunk(self.dtype, ndim=self.ndim)
        task = self.context.create_task(CuNumericOpCode.REPEAT)
        task.add_input(self.base)
        task.add_output(out.base)
        # We pass axis now but don't use for 1D case (will use for ND case
        task.add_scalar_arg(axis, ty.int32)
        task.add_scalar_arg(scalar_repeats, bool)
        if scalar_repeats:
            task.add_scalar_arg(repeats, ty.int64)
        else:
            shape = self.shape
            repeats = self.runtime.to_deferred_array(repeats).base
            for dim, extent in enumerate(shape):
                if dim == axis:
                    continue
                repeats = repeats.promote(dim, extent)
            task.add_input(repeats)
            task.add_alignment(self.base, repeats)
        task.execute()
        return out

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

        task.add_broadcast(self.base, axes=range(1, self.ndim))

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
    @auto_convert([2])
    def unary_op(self, op, src, where, args):
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

        argred = op in (UnaryRedCode.ARGMAX, UnaryRedCode.ARGMIN)

        # See if we are doing reduction to a point or another region
        if lhs_array.size == 1:
            assert axes is None or len(axes) == (
                rhs_array.ndim - lhs_array.ndim
            )

            task = self.context.create_task(CuNumericOpCode.SCALAR_UNARY_RED)

            if initial is not None:
                assert not argred
                fill_value = initial
            else:
                fill_value = _UNARY_RED_IDENTITIES[op](rhs_array.dtype)

            lhs_array.fill(np.array(fill_value, dtype=lhs_array.dtype))

            task.add_reduction(lhs_array.base, _UNARY_RED_TO_REDUCTION_OPS[op])
            task.add_input(rhs_array.base)
            task.add_scalar_arg(op, ty.int32)

            self.add_arguments(task, args)

            task.execute()

        else:
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
        cholesky(self, src, no_tril)

    def unique(self):
        result = self.runtime.create_unbound_thunk(self.dtype)

        task = self.context.create_task(CuNumericOpCode.UNIQUE)

        task.add_output(result.base)
        task.add_input(self.base)

        if self.runtime.num_gpus > 0:
            task.add_nccl_communicator()

        task.execute()

        if self.runtime.num_gpus == 0 and self.runtime.num_procs > 1:
            result.base = self.context.tree_reduce(
                CuNumericOpCode.UNIQUE_REDUCE, result.base
            )

        return result

    @auto_convert([1])
    def sort(self, rhs, argsort=False, axis=-1, kind="quicksort", order=None):

        if kind == "stable":
            stable = True
        else:
            stable = False

        if order is not None:
            raise NotImplementedError(
                "cuNumeric does not support sorting with 'order' as "
                "ndarray only supports numeric values"
            )
        if axis is not None and (axis >= rhs.ndim or axis < -rhs.ndim):
            raise ValueError("invalid axis")

        sort(self, rhs, argsort, axis, stable)

    @auto_convert([1])
    def partition(
        self,
        rhs,
        kth,
        argpartition=False,
        axis=-1,
        kind="introselect",
        order=None,
    ):

        if order is not None:
            raise NotImplementedError(
                "cuNumeric does not support partitioning with 'order' as "
                "ndarray only supports numeric values"
            )
        if axis is not None and (axis >= rhs.ndim or axis < -rhs.ndim):
            raise ValueError("invalid axis")

        # fallback to sort for now
        sort(self, rhs, argpartition, axis, False)

    def create_window(self, op_code, M, *args):
        task = self.context.create_task(CuNumericOpCode.WINDOW)
        task.add_output(self.base)
        task.add_scalar_arg(op_code, ty.int32)
        task.add_scalar_arg(M, ty.int64)
        for arg in args:
            task.add_scalar_arg(arg, ty.float64)
        task.execute()
