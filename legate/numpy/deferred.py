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

import warnings
import weakref
from collections.abc import Iterable
from functools import reduce

import numpy as np

import legate.core.types as ty
from legate.core import *  # noqa F403

from .config import *  # noqa F403
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
            stacklevel = kwargs.get("stacklevel") + 1

            args = tuple(
                self.runtime.to_deferred_array(arg, stacklevel=stacklevel)
                if idx in indices
                else arg
                for (idx, arg) in enumerate(args)
            )
            for key in keys:
                v = kwargs.get(key, None)
                if v is None:
                    continue
                v = self.runtime.to_deferred_array(v, stacklevel=stacklevel)
                kwargs[key] = v
            kwargs["stacklevel"] = stacklevel

            return func(*args, **kwargs)

        return wrapper

    return decorator


def profile(func):
    def wrapper(*args, **kwargs):
        self = args[0]
        stacklevel = kwargs.get("stacklevel") + 1
        callsite = kwargs.get("callsite")
        kwargs["stacklevel"] = stacklevel

        result = func(*args, **kwargs)

        self.runtime.profile_callsite(stacklevel, True, callsite)

        return result

    return wrapper


def shadow_debug(func_name, indices, keys=[]):
    indices = set(indices)

    def decorator(func):
        def wrapper(*args, **kwargs):
            self = args[0]
            if not self.runtime.shadow_debug:
                return func(*args, **kwargs)

            stacklevel = kwargs.get("stacklevel") + 1
            kwargs["stacklevel"] = stacklevel
            result = func(*args, **kwargs)

            shadow_args = tuple(
                arg.shadow if idx in indices else arg
                for (idx, arg) in enumerate(args)
            )
            shadow_args = shadow_args[1:]
            shadow_kwargs = kwargs.copy()
            if "callsite" in shadow_kwargs:
                del shadow_kwargs["callsite"]
            for key in keys:
                v = shadow_kwargs.get(key, None)
                if v is None:
                    continue
                shadow_kwargs[key] = v.shadow

            assert self.shadow.shadow
            getattr(self.shadow, func_name)(*shadow_args, **shadow_kwargs)
            self.runtime.check_shadow(self, func_name)

            return result

        return wrapper

    return decorator


# This is a dummy object that is only used as an initializer for the
# RegionField object above. It is thrown away as soon as the
# RegionField is constructed.
class _LegateNDarray(object):
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


class DeferredArray(NumPyThunk):
    """This is a deferred thunk for describing NumPy computations.
    It is backed by either a Legion logical region or a Legion future
    for describing the result of a compuation.

    :meta private:
    """

    def __init__(self, runtime, base, dtype, scalar, numpy_array=None):
        NumPyThunk.__init__(self, runtime, dtype)
        assert base is not None
        assert isinstance(base, Store)
        self.base = base  # a Legate Store
        self.scalar = scalar
        self.numpy_array = (
            None if numpy_array is None else weakref.ref(numpy_array)
        )

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

    def __numpy_array__(self, stacklevel=0):
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
                self.get_scalar_array(stacklevel=(stacklevel + 1)),
                dtype=self.dtype,
            )
        else:
            alloc = self.base.get_inline_allocation(self.context)

            def construct_ndarray(shape, address, strides):
                initializer = _LegateNDarray(
                    shape, self.dtype, address, strides, False
                )
                return np.asarray(initializer)

            result = alloc.consume(construct_ndarray)

        self.numpy_array = weakref.ref(result)
        return result

    # TODO: We should return a view of the field instead of a copy
    def imag(self, stacklevel=0, callsite=None):
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
            stacklevel=stacklevel + 1,
            callsite=callsite,
        )

        return result

    # TODO: We should return a view of the field instead of a copy
    def real(self, stacklevel=0, callsite=None):
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
            stacklevel=stacklevel + 1,
            callsite=callsite,
        )

        return result

    def conj(self, stacklevel=0, callsite=None):
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
            stacklevel=stacklevel + 1,
            callsite=callsite,
        )

        return result

    # Copy source array to the destination array
    def copy(self, rhs, deep=False, stacklevel=0, callsite=None):
        self.unary_op(
            UnaryOpCode.COPY,
            rhs.dtype,
            rhs,
            True,
            [],
            stacklevel=stacklevel + 1,
            callsite=callsite,
        )

    def get_scalar_array(self, stacklevel):
        assert self.size == 1
        assert self.base.scalar
        # Look at the type of the data and figure out how to read this data
        # First four bytes are for the type code, so we need to skip those
        buf = self.base.storage.get_buffer(self.dtype.itemsize + 8)[8:]
        result = np.frombuffer(buf, dtype=self.dtype, count=1)
        return result.reshape(())

    def _create_indexing_array(self, key, stacklevel):
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
                tuple_of_arrays = key.nonzero(stacklevel + 1)
            else:
                tuple_of_arrays = (key,)
        if len(tuple_of_arrays) != self.ndim:
            raise TypeError("Advanced indexing dimension mismatch")
        if self.ndim > 1:
            # Check that all the arrays can be broadcast together
            # Concatenate all the arrays into a single array
            raise NotImplementedError("need support for concatenating arrays")
        else:
            return self.runtime.to_deferred_array(
                tuple_of_arrays[0], stacklevel=(stacklevel + 1)
            )

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
            scalar=False,
        )

    def _broadcast(self, shape):
        result = self.base
        diff = len(shape) - result.ndim
        for dim in range(diff):
            result = result.promote(dim, shape[dim])

        for dim in range(shape.ndim):
            if result.shape[dim] != shape[dim]:
                assert result.shape[dim] == 1
                result = result.project(dim, 0).promote(dim, shape[dim])

        return result

    def get_item(self, key, stacklevel, view=None, dim_map=None):
        assert self.size > 1
        # Check to see if this is advanced indexing or not
        if self._is_advanced_indexing(key):
            # Create the indexing array
            index_array = self._create_indexing_array(
                key, stacklevel=(stacklevel + 1)
            )
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
                    1, self.dtype, inputs=[self]
                )

                task = self.context.create_task(NumPyOpCode.READ)
                task.add_input(input.base)
                task.add_output(result.base)

                task.execute()

        if self.runtime.shadow_debug:
            result.shadow = self.shadow.get_item(
                key, stacklevel=(stacklevel + 1), view=view, dim_map=dim_map
            )
        return result

    @auto_convert([2])
    def set_item(self, key, rhs, stacklevel=0):
        assert self.dtype == rhs.dtype
        # Check to see if this is advanced indexing or not
        if self._is_advanced_indexing(key):
            # Create the indexing array
            index_array = self._create_indexing_array(
                key, stacklevel=(stacklevel + 1)
            )
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

        elif self.size == 1:
            assert rhs.size == 1
            # Special case of writing a single value
            # We can just copy the future because they are immutable
            self.base = rhs.base
        else:
            view = self._get_view(key)

            if view.shape == ():
                # We're just writing a single value
                assert rhs.size == 1

                task = self.context.create_task(NumPyOpCode.WRITE)
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
                # to set the result back. In Legate Numpy, the object we
                # return in step (1) is actually a subview to the array arr
                # through which we make udpates in place, so after step (2) is
                # done, # the effect of inplace update is already reflected
                # to the arr. Therefore, we skip the copy to avoid redundant
                # copies if we know that we hit such a scenario.
                # TODO: We should make this work for the advanced indexing case
                if view.base.storage.same_handle(rhs.base.storage):
                    return

                if self.runtime.shadow_debug:
                    view.shadow = self.runtime.to_eager_array(
                        view,
                        stacklevel + 1,
                    )

                if view.base.overlaps(rhs.base):
                    rhs_copy = self.runtime.create_empty_thunk(
                        rhs.shape,
                        rhs.dtype,
                        inputs=[rhs],
                    )
                    rhs_copy.copy(rhs, deep=False, stacklevel=(stacklevel + 1))
                    rhs = rhs_copy

                view.copy(rhs, deep=False, stacklevel=(stacklevel + 1))
        if self.runtime.shadow_debug:
            self.shadow.set_item(key, rhs.shadow, stacklevel=(stacklevel + 1))
            self.runtime.check_shadow(self, "set_item")

    def reshape(self, newshape, order, stacklevel):
        assert isinstance(newshape, Iterable)
        if order == "A":
            order = "C"

        if order != "C":
            # If we don't have a transform then we need to make a copy
            warnings.warn(
                "legate.numpy has not implemented reshape using Fortran-like "
                "index order and is falling back to canonical numpy. You may "
                "notice significantly decreased performance for this "
                "function call.",
                stacklevel=(stacklevel + 1),
                category=RuntimeWarning,
            )
            numpy_array = self.__numpy_array__(stacklevel=(stacklevel + 1))
            # Force a copy here because we know we can't build a view
            result_array = numpy_array.reshape(newshape, order=order).copy()
            result = self.runtime.get_numpy_thunk(
                result_array, stacklevel=(stacklevel + 1)
            )

            if self.runtime.shadow_debug:
                result.shadow = self.shadow.reshape(
                    newshape, order, stacklevel=(stacklevel + 1)
                )

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

        needs_copy = any(len(src_g) > 1 for src_g, _ in groups)

        tmp_shape = ()
        for src_g, tgt_g in groups:
            if len(src_g) > 1 and len(tgt_g) > 1:
                tmp_shape += (_prod(tgt_g),)
            else:
                tmp_shape += tgt_g

        if needs_copy:
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

            src_array = DeferredArray(
                self.runtime, src, self.dtype, self.scalar
            )
            tgt_array = DeferredArray(
                self.runtime, tgt, self.dtype, self.scalar
            )
            tgt_array.copy(src_array, deep=True, stacklevel=stacklevel + 1)

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
                elif len(src_g) == 1:
                    src = src.delinearize(src_dim, tgt_g)
                    diff = len(tgt_g)
                else:
                    # unreachable
                    assert False

                src_dim += diff

            result = DeferredArray(self.runtime, src, self.dtype, self.scalar)

        if tmp_shape != newshape:
            tgt = result.base
            tgt_dim = 0
            for src_g, tgt_g in groups:
                if len(src_g) > 1 and len(tgt_g) > 1:
                    tgt = tgt.delinearize(tgt_dim, tgt_g)
                tgt_dim += len(tgt_g)

            result = DeferredArray(self.runtime, tgt, self.dtype, self.scalar)

        return result

    def squeeze(self, axis, stacklevel):
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
        result = DeferredArray(self.runtime, result, self.dtype, self.scalar)
        if self.runtime.shadow_debug:
            result.shadow = self.shadow.squeeze(
                axis, stacklevel=stacklevel + 1
            )
        return result

    def swapaxes(self, axis1, axis2, stacklevel=0):
        if self.size == 1 or axis1 == axis2:
            return self
        # Make a new deferred array object and swap the results
        assert axis1 < self.ndim and axis2 < self.ndim

        dims = list(range(self.ndim))
        dims[axis1], dims[axis2] = dims[axis2], dims[axis1]

        result = self.base.transpose(dims)
        result = DeferredArray(self.runtime, result, self.dtype, False)

        if self.runtime.shadow_debug:
            result.shadow = self.shadow.swapaxes(
                axis1, axis2, stacklevel=stacklevel + 1
            )

        return result

    # Convert the source array to the destination array
    @profile
    @auto_convert([1])
    @shadow_debug("convert", [1])
    def convert(self, rhs, stacklevel=0, warn=True, callsite=None):
        lhs_array = self
        rhs_array = rhs
        assert lhs_array.dtype != rhs_array.dtype

        if warn:
            warnings.warn(
                "Legate performing implicit type conversion from "
                + str(rhs_array.dtype)
                + " to "
                + str(lhs_array.dtype),
                category=UserWarning,
                stacklevel=(stacklevel + 1),
            )

        lhs = lhs_array.base
        rhs = rhs_array.base

        if rhs.scalar:
            task_id = NumPyOpCode.SCALAR_CONVERT
        else:
            task_id = NumPyOpCode.CONVERT

        task = self.context.create_task(task_id)
        task.add_output(lhs)
        task.add_input(rhs)
        task.add_dtype_arg(lhs_array.dtype)

        task.add_alignment(lhs, rhs)

        task.execute()

    # Fill the legate array with the value in the numpy array
    @profile
    def _fill(self, value, stacklevel=0, callsite=None):
        if self.size == 1:
            # Handle the 0D case special
            self.base.set_storage(value.storage)
        else:
            assert self.base is not None
            dtype = self.dtype
            # If this is a fill for an arg value, make sure to pass
            # the value dtype so that we get it packed correctly
            if dtype.kind == "V":
                dtype = get_arg_value_dtype(dtype)
            else:
                dtype = None

            task = self.context.create_task(NumPyOpCode.FILL)
            task.add_output(self.base)
            task.add_input(value)

            task.execute()

    @shadow_debug("fill", [])
    def fill(self, numpy_array, stacklevel=0, callsite=None):
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
        self._fill(store, stacklevel=stacklevel + 1, callsite=callsite)

    @profile
    @auto_convert([1, 2])
    @shadow_debug("dot", [1, 2])
    def dot(self, src1, src2, stacklevel=0, callsite=None):
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

            redop = self.runtime.get_scalar_reduction_op_id(UnaryRedCode.SUM)

            task = self.context.create_task(NumPyOpCode.DOT)
            task.add_reduction(lhs_array.base, redop)
            task.add_input(rhs1_array.base)
            task.add_input(rhs2_array.base)

            task.add_alignment(rhs1_array.base, rhs2_array.base)

            task.execute()

            if rhs1_array.dtype == np.float16:
                self.convert(
                    lhs_array,
                    stacklevel=stacklevel + 1,
                    warn=False,
                    callsite=callsite,
                )

        elif rhs1_array.ndim == 1 or rhs2_array.ndim == 1:
            # Matrix-vector or vector-matrix multiply
            assert lhs_array.ndim == 1
            assert rhs1_array.ndim == 2 or rhs2_array.ndim == 2

            left_matrix = rhs1_array.ndim == 2

            if left_matrix and rhs1_array.shape[0] == 1:
                rhs1_array = rhs1_array.get_item(
                    (0, slice(None)), stacklevel + 1
                )
                lhs_array.dot(
                    rhs1_array,
                    rhs2_array,
                    stacklevel=stacklevel + 1,
                    callsite=callsite,
                )
                return
            elif not left_matrix and rhs2_array.shape[1] == 1:
                rhs2_array = rhs2_array.get_item(
                    (slice(None), 0), stacklevel + 1
                )
                lhs_array.dot(
                    rhs1_array,
                    rhs2_array,
                    stacklevel=stacklevel + 1,
                    callsite=callsite,
                )
                return

            # If the inputs are 16-bit floats, we should use 32-bit float
            # for accumulation
            if rhs1_array.dtype == np.float16:
                lhs_array = self.runtime.create_empty_thunk(
                    self.shape, np.dtype(np.float32), inputs=[self]
                )

            # TODO: We should be able to do this in the core
            lhs_array.fill(
                np.array(0, dtype=lhs_array.dtype),
                stacklevel=(stacklevel + 1),
                callsite=callsite,
            )

            if left_matrix:
                rhs1 = rhs1_array.base
                (m, n) = rhs1.shape
                rhs2 = rhs2_array.base.promote(0, m)
                lhs = lhs_array.base.promote(1, n)
            else:
                rhs2 = rhs2_array.base
                (m, n) = rhs2.shape
                rhs1 = rhs1_array.base.promote(1, n)
                lhs = lhs_array.base.promote(0, m)

            redop = self.runtime.get_unary_reduction_op_id(
                UnaryRedCode.SUM, lhs_array.dtype
            )

            task = self.context.create_task(NumPyOpCode.MATVECMUL)
            task.add_reduction(lhs, redop)
            task.add_input(rhs1)
            task.add_input(rhs2)
            task.add_scalar_arg(left_matrix, bool)

            task.add_alignment(lhs, rhs1)
            task.add_alignment(lhs, rhs2)

            task.execute()

            # If we used an accumulation buffer, we should copy the results
            # back to the lhs
            if rhs1_array.dtype == np.float16:
                # Since we're still in the middle of operation, we haven't had
                # a chance to get the shadow array for this intermediate array,
                # so we manually attach a shadow array for it
                if self.runtime.shadow_debug:
                    lhs_array.shadow = self.runtime.to_eager_array(
                        lhs_array,
                        stacklevel + 1,
                    )
                self.convert(
                    lhs_array,
                    stacklevel=stacklevel + 1,
                    warn=False,
                    callsite=callsite,
                )

        elif rhs1_array.ndim == 2 and rhs2_array.ndim == 2:
            # Matrix-matrix multiply
            M = lhs_array.shape[0]
            N = lhs_array.shape[1]
            K = rhs1_array.shape[1]
            assert M == rhs1_array.shape[0]  # Check M
            assert N == rhs2_array.shape[1]  # Check N
            assert K == rhs2_array.shape[0]  # Check K

            if M == 1 and N == 1:
                rhs1_array = rhs1_array.get_item(
                    (0, slice(None)), stacklevel + 1
                )
                rhs2_array = rhs2_array.get_item(
                    (slice(None), 0), stacklevel + 1
                )
                lhs_array.dot(
                    rhs1_array,
                    rhs2_array,
                    stacklevel=stacklevel + 1,
                    callsite=callsite,
                )
                return

            if rhs1_array.dtype == np.float16:
                lhs_array = self.runtime.create_empty_thunk(
                    self.shape, np.dtype(np.float32), inputs=[self]
                )

            # TODO: We should be able to do this in the core
            lhs_array.fill(
                np.array(0, dtype=lhs_array.dtype),
                stacklevel=(stacklevel + 1),
                callsite=callsite,
            )

            lhs = lhs_array.base.promote(1, K)
            rhs1 = rhs1_array.base.promote(2, N)
            rhs2 = rhs2_array.base.promote(0, M)

            redop = self.runtime.get_unary_reduction_op_id(
                UnaryRedCode.SUM, lhs_array.dtype
            )

            task = self.context.create_task(NumPyOpCode.MATMUL)
            task.add_reduction(lhs, redop)
            task.add_input(rhs1)
            task.add_input(rhs2)

            task.add_alignment(lhs, rhs1)
            task.add_alignment(lhs, rhs2)

            task.execute()

            # If we used an accumulation buffer, we should copy the results
            # back to the lhs
            if rhs1_array.dtype == np.float16:
                # Since we're still in the middle of operation, we haven't had
                # a chance to get the shadow array for this intermediate array,
                # so we manually attach a shadow array for it
                if self.runtime.shadow_debug:
                    lhs_array.shadow = self.runtime.to_eager_array(
                        lhs_array,
                        stacklevel + 1,
                    )
                self.convert(
                    lhs_array,
                    stacklevel=stacklevel + 1,
                    warn=False,
                    callsite=callsite,
                )
        else:
            raise NotImplementedError("Need support for tensor contractions")

    # Create or extract a diagonal from a matrix
    @profile
    @auto_convert([1])
    @shadow_debug("diag", [1])
    def diag(self, rhs, extract, k, stacklevel=0, callsite=None):
        if extract:
            matrix_array = rhs
            diag_array = self
        else:
            matrix_array = self
            diag_array = rhs

        assert diag_array.ndim == 1
        assert matrix_array.ndim == 2
        assert diag_array.shape[0] <= min(
            matrix_array.shape[0], matrix_array.shape[1]
        )
        assert rhs.dtype == self.dtype

        # Issue a fill operation to get the output initialized
        if extract:
            diag_array.fill(
                np.array(0, dtype=diag_array.dtype),
                stacklevel=(stacklevel + 1),
                callsite=callsite,
            )
        else:
            matrix_array.fill(
                np.array(0, dtype=matrix_array.dtype),
                stacklevel=(stacklevel + 1),
                callsite=callsite,
            )

        matrix = matrix_array.base
        diag = diag_array.base

        if k > 0:
            matrix = matrix.slice(1, slice(k, None))
        elif k < 0:
            matrix = matrix.slice(0, slice(-k, None))

        if matrix.shape[0] < matrix.shape[1]:
            diag = diag.promote(1, matrix.shape[1])
        else:
            diag = diag.promote(0, matrix.shape[0])

        task = self.context.create_task(NumPyOpCode.DIAG)

        if extract:
            redop = self.runtime.get_unary_reduction_op_id(
                UnaryRedCode.SUM, diag_array.dtype
            )
            task.add_reduction(diag, redop)
            task.add_input(matrix)
        else:
            task.add_output(matrix)
            task.add_input(matrix)
            task.add_input(diag)

        task.add_scalar_arg(extract, bool)

        task.add_alignment(matrix, diag)

        task.execute()

    # Create an identity array with the ones offset from the diagonal by k
    @profile
    @shadow_debug("eye", [])
    def eye(self, k, stacklevel=0, callsite=None):
        assert self.ndim == 2  # Only 2-D arrays should be here
        # First issue a fill to zero everything out
        self.fill(
            np.array(0, dtype=self.dtype),
            stacklevel=(stacklevel + 1),
            callsite=callsite,
        )

        task = self.context.create_task(NumPyOpCode.EYE)
        task.add_output(self.base)
        task.add_scalar_arg(k, ty.int32)

        task.execute()

    @profile
    @shadow_debug("arange", [])
    def arange(self, start, stop, step, stacklevel=0, callsite=None):
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

        task = self.context.create_task(NumPyOpCode.ARANGE)
        task.add_output(self.base)
        task.add_input(create_scalar(start, self.dtype))
        task.add_input(create_scalar(stop, self.dtype))
        task.add_input(create_scalar(step, self.dtype))

        task.execute()

    # Tile the src array onto the destination array
    @profile
    @auto_convert([1])
    @shadow_debug("tile", [1])
    def tile(self, rhs, reps, stacklevel=0, callsite=None):
        src_array = rhs
        dst_array = self
        assert src_array.ndim <= dst_array.ndim
        assert src_array.dtype == dst_array.dtype
        if src_array.size == 1:
            self._fill(
                src_array.base, stacklevel=stacklevel + 1, callsite=callsite
            )
            return

        task = self.context.create_task(NumPyOpCode.TILE)

        task.add_output(self.base)
        task.add_input(rhs.base)

        task.add_broadcast(rhs.base)

        task.execute()

    # Transpose the matrix dimensions
    @profile
    @auto_convert([1])
    @shadow_debug("transpose", [1])
    def transpose(self, rhs, axes, stacklevel=0, callsite=None):
        rhs_array = rhs
        lhs_array = self
        assert lhs_array.dtype == rhs_array.dtype
        assert lhs_array.ndim == rhs_array.ndim
        assert lhs_array.ndim == len(axes)
        lhs_array.base = rhs_array.base.transpose(axes)

    # Perform a bin count operation on the array
    @profile
    @auto_convert([1], ["weights"])
    @shadow_debug("bincount", [1])
    def bincount(self, rhs, stacklevel=0, weights=None, callsite=None):
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

        dst_array.fill(
            np.array(0, dst_array.dtype),
            stacklevel=stacklevel + 1,
            callsite=callsite,
        )

        redop = self.runtime.get_unary_reduction_op_id(
            UnaryRedCode.SUM,
            dst_array.dtype,
        )

        task = self.context.create_task(NumPyOpCode.BINCOUNT)
        task.add_reduction(dst_array.base, redop)
        task.add_input(src_array.base)
        task.add_input(weight_array.base)

        task.add_broadcast(dst_array.base)
        if not weight_array.scalar:
            task.add_alignment(src_array.base, weight_array.base)

        task.execute()

    def nonzero(self, stacklevel=0, callsite=None):
        results = tuple(
            self.runtime.create_unbound_thunk(np.dtype(np.int64))
            for _ in range(self.ndim)
        )

        task = self.context.create_task(NumPyOpCode.NONZERO)

        task.add_input(self.base)
        for result in results:
            task.add_output(result.base)

        task.execute()
        return results

    @profile
    def random(self, gen_code, args, stacklevel=0, callsite=None):
        task = self.context.create_task(NumPyOpCode.RAND)

        task.add_output(self.base)
        task.add_scalar_arg(gen_code.value, ty.int32)
        epoch = self.runtime.get_next_random_epoch()
        task.add_scalar_arg(epoch, ty.uint32)
        task.add_scalar_arg(self.compute_strides(self.shape), (ty.int64,))
        self.add_arguments(task, args)

        task.execute()

        if self.runtime.shadow_debug:
            self.shadow = self.runtime.to_eager_array(self, stacklevel + 1)

    def random_uniform(self, stacklevel, callsite=None):
        assert self.dtype == np.float64
        self.random(
            RandGenCode.UNIFORM,
            [],
            stacklevel=stacklevel + 1,
            callsite=callsite,
        )

    def random_normal(self, stacklevel, callsite=None):
        assert self.dtype == np.float64
        self.random(
            RandGenCode.NORMAL,
            [],
            stacklevel=stacklevel + 1,
            callsite=callsite,
        )

    def random_integer(self, low, high, stacklevel, callsite=None):
        assert self.dtype.kind == "i"
        low = np.array(low, self.dtype)
        high = np.array(high, self.dtype)
        self.random(
            RandGenCode.INTEGER,
            [low, high],
            stacklevel=stacklevel + 1,
            callsite=callsite,
        )

    # Perform the unary operation and put the result in the array
    @profile
    @auto_convert([3])
    @shadow_debug("unary_op", [3])
    def unary_op(
        self, op, op_dtype, src, where, args, stacklevel=0, callsite=None
    ):
        lhs = self.base
        rhs = src._broadcast(lhs.shape)

        if lhs.scalar:
            task_id = NumPyOpCode.SCALAR_UNARY_OP
        else:
            task_id = NumPyOpCode.UNARY_OP

        task = self.context.create_task(task_id)
        task.add_output(lhs)
        task.add_input(rhs)
        task.add_scalar_arg(op.value, ty.int32)
        self.add_arguments(task, args)

        task.add_alignment(lhs, rhs)

        task.execute()

    # Perform a unary reduction operation from one set of dimensions down to
    # fewer
    @profile
    @auto_convert([2])
    @shadow_debug("unary_reduction", [2])
    def unary_reduction(
        self,
        op,
        src,
        where,
        axes,
        keepdims,
        args,
        initial,
        stacklevel=0,
        callsite=None,
    ):
        lhs_array = self
        rhs_array = src
        assert lhs_array.ndim <= rhs_array.ndim
        assert rhs_array.size > 1

        # See if we are doing reduction to a point or another region
        if lhs_array.size == 1:
            assert axes is None or len(axes) == (
                rhs_array.ndim - lhs_array.ndim
            )

            task = self.context.create_task(NumPyOpCode.SCALAR_UNARY_RED)

            redop = self.runtime.get_scalar_reduction_op_id(op)
            task.add_reduction(lhs_array.base, redop)
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
                fill_value = self.runtime.get_reduction_identity(
                    op, rhs_array.dtype
                )
            lhs_array.fill(
                np.array(fill_value, lhs_array.dtype),
                stacklevel=stacklevel + 1,
                callsite=callsite,
            )

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

            task = self.context.create_task(NumPyOpCode.UNARY_RED)

            redop = self.runtime.get_unary_reduction_op_id(op, rhs_array.dtype)

            task.add_input(rhs_array.base)
            task.add_reduction(result, redop)
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
                    stacklevel=stacklevel + 1,
                    callsite=callsite,
                )

    # Perform the binary operation and put the result in the lhs array
    @profile
    @auto_convert([2, 3])
    @shadow_debug("binary_op", [2, 3])
    def binary_op(
        self, op_code, src1, src2, where, args, stacklevel=0, callsite=None
    ):
        lhs = self.base
        rhs1 = src1._broadcast(lhs.shape)
        rhs2 = src2._broadcast(lhs.shape)

        # Populate the Legate launcher
        all_scalar_rhs = rhs1.scalar and rhs2.scalar

        if all_scalar_rhs:
            task_id = NumPyOpCode.SCALAR_BINARY_OP
        else:
            task_id = NumPyOpCode.BINARY_OP

        task = self.context.create_task(task_id)
        task.add_output(lhs)
        task.add_input(rhs1)
        task.add_input(rhs2)
        task.add_scalar_arg(op_code.value, ty.int32)
        self.add_arguments(task, args)

        task.add_alignment(lhs, rhs1)
        task.add_alignment(lhs, rhs2)

        task.execute()

    @profile
    @auto_convert([2, 3])
    @shadow_debug("binary_reduction", [2, 3])
    def binary_reduction(
        self, op, src1, src2, broadcast, args, stacklevel=0, callsite=None
    ):
        lhs = self.base
        rhs1 = src1.base
        rhs2 = src2.base
        assert lhs.scalar

        if broadcast is not None:
            rhs1 = rhs1._broadcast(broadcast)
            rhs2 = rhs2._broadcast(broadcast)

        # Populate the Legate launcher
        all_scalar_rhs = rhs1.scalar and rhs2.scalar

        if all_scalar_rhs:
            task_id = NumPyOpCode.SCALAR_BINARY_OP
        else:
            task_id = NumPyOpCode.BINARY_RED

        redop = self.runtime.get_scalar_reduction_op_id(
            UnaryRedCode.SUM
            if op == BinaryOpCode.NOT_EQUAL
            else UnaryRedCode.PROD
        )
        task = self.context.create_task(task_id)
        task.add_reduction(lhs, redop)
        task.add_input(rhs1)
        task.add_input(rhs2)
        task.add_scalar_arg(op.value, ty.int32)
        self.add_arguments(task, args)

        task.add_alignment(rhs1, rhs2)

        task.execute()

    @profile
    @auto_convert([1, 2, 3])
    @shadow_debug("where", [1, 2, 3])
    def where(self, src1, src2, src3, stacklevel=0, callsite=None):
        lhs = self.base
        rhs1 = src1._broadcast(lhs.shape)
        rhs2 = src2._broadcast(lhs.shape)
        rhs3 = src3._broadcast(lhs.shape)

        # Populate the Legate launcher
        all_scalar_rhs = rhs1.scalar and rhs2.scalar and rhs3.scalar

        if all_scalar_rhs:
            task_id = NumPyOpCode.SCALAR_WHERE
        else:
            task_id = NumPyOpCode.WHERE

        task = self.context.create_task(task_id)
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
