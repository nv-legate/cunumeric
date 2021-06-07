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

from __future__ import absolute_import, division, print_function

import warnings

import numpy as np

from legate.core import *  # noqa F403

from .config import *  # noqa F403
from .launcher import Broadcast, Map, Projection
from .thunk import NumPyThunk
from .utils import get_arg_dtype, get_arg_value_dtype

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

try:
    long  # Python 2
except NameError:
    long = int  # Python 3


def _maybe_apply(f, v, *args, **kwargs):
    if v is None:
        return v
    else:
        return f(v, *args, **kwargs)


def _combine_transforms(f, g):
    if g is None:
        return f
    elif f is None:
        return g
    else:
        return g.compose(f)


def _complex_field_dtype(dtype):
    if dtype == np.complex64:
        return np.dtype(np.float32)
    elif dtype == np.complex128:
        return np.dtype(np.float64)
    elif dtype == np.complex256:
        return np.dtype(np.float128)
    else:
        assert False


class DeferredArrayView(object):
    def __init__(
        self,
        array,
        transform=None,
        offset=None,
        part=None,
        proj_id=0,
        tag=0,
        dtype=None,
    ):
        self._array = array
        self._transform = transform
        self._offset = offset
        self._part = part
        self._proj_id = proj_id
        self._tag = tag
        self._dtype = dtype

    @property
    def scalar(self):
        return self._array.scalar

    @property
    def shape(self):
        return self._array.shape

    @property
    def size(self):
        return self._array.size

    @property
    def dtype(self):
        if self._dtype is None:
            self._dtype = self._array.dtype
        return self._dtype

    @property
    def ndim(self):
        return self._array.ndim

    @property
    def part(self):
        assert not self.scalar
        if self._part is None:
            (part, _, _) = self._array.base.find_or_create_key_partition()
            self._part = part
        return self._part

    @property
    def sharding(self):
        if self.scalar:
            return None, 0, None
        else:
            return self._array.base.find_point_sharding()

    def update_tag(self, tag):
        self._tag = tag

    def copy_key_partition_from(self, src):
        if self.scalar:
            return
        else:
            (
                src_part,
                shardfn,
                shardsp,
            ) = src._array.base.find_or_create_key_partition()
            part = self._array.base.find_or_create_congruent_partition(
                src_part
            )
            self._array.base.set_key_partition(part, shardfn, shardsp)

    def compute_projection(self, redop):
        if self._proj_id is None:
            return Broadcast(redop=redop)
        else:
            return Projection(self.part, self._proj_id, redop=redop)

    def add_to_legate_op(self, op, read_only, read_write=False, redop=None):
        op.add_scalar_arg(self.scalar, bool)
        dim = self.ndim if self._transform is None else self._transform.N
        op.add_scalar_arg(dim, np.int32)
        op.add_dtype_arg(self.dtype)
        if self.scalar:
            if not read_only:
                raise ValueError("Singleton arrays must be read only")
            op.add_future(self._array.base)
        else:
            if redop is None:
                if read_write:
                    add = op.add_inout
                else:
                    add = op.add_input if read_only else op.add_output
            else:
                assert not read_only
                add = op.add_reduction
            add(
                self._array,
                _combine_transforms(
                    self._array.base.transform, self._transform
                ),
                self.compute_projection(redop),
                tag=self._tag,
            )

    def compute_launch_space(self):
        if self.scalar:
            return None
        else:
            return self._array.base.compute_parallel_launch_space()

    @property
    def has_launch_space(self):
        if self.scalar:
            return False
        else:
            return self._array.base.has_parallel_launch_space()

    def find_key_view(self, *views, shape=None):
        if self.has_launch_space:
            return self.compute_launch_space(), self

        shape = self.shape if shape is None else shape
        views = list(filter(lambda v: v.shape == shape, [*views, self]))
        assert len(views) > 0

        for view in views:
            launch_space = view.compute_launch_space()
            if launch_space is not None:
                return launch_space, view

        # If we're here, we haven't found any partitioned region field
        # to use as the key
        return None, views[0]

    def broadcast(self, to_align):
        if self.scalar:
            return self
        else:
            if self is to_align:
                return self
            if not isinstance(to_align, tuple):
                assert isinstance(to_align, DeferredArrayView)
                to_align = to_align._array.shape
            (
                transform,
                offset,
                proj_id,
                mapping_tag,
            ) = self._array.runtime.compute_broadcast_transform(
                to_align, self._array.shape
            )
            new_view = DeferredArrayView(
                self._array,
                transform,
                offset,
                self._part,
                proj_id,
                mapping_tag,
                self._dtype,
            )
            return new_view

    def align_partition(self, key):
        if self is key or self.scalar:
            return self
        else:
            new_part = self._array.base.find_or_create_congruent_partition(
                key.part, self._transform, self._offset
            )
            new_view = DeferredArrayView(
                self._array,
                self._transform,
                self._offset,
                new_part,
                self._proj_id,
                self._tag,
                self._dtype,
            )
            return new_view


class DeferredArray(NumPyThunk):
    """This is a deferred thunk for describing NumPy computations.
    It is backed by either a Legion logical region or a Legion future
    for describing the result of a compuation.

    :meta private:
    """

    def __init__(self, runtime, base, shape, dtype, scalar):
        NumPyThunk.__init__(self, runtime, shape, dtype)
        assert base is not None
        self.base = base  # Either a RegionField or a Future
        self.scalar = scalar

    @property
    def storage(self):
        if isinstance(self.base, Future):
            return self.base
        else:
            return (self.base.region, self.base.field.field_id)

    @property
    def ndim(self):
        return len(self.shape)

    def __numpy_array__(self, stacklevel):
        if self.scalar:
            return np.full(
                self.shape,
                self.get_scalar_array(stacklevel=(stacklevel + 1)),
                dtype=self.dtype,
            )
        elif self.size == 0:
            # Return an empty array with the right number of dimensions
            # and type
            return np.empty(shape=self.shape, dtype=self.dtype)
        else:
            return self.base.get_numpy_array()

    # TODO: We should return a view of the field instead of a copy
    def imag(self, stacklevel, callsite=None):
        dtype = _complex_field_dtype(self.dtype)
        if self.scalar:
            result_field = Future()
        else:
            result_field = self.runtime.allocate_field(self.shape, dtype=dtype)

        result = DeferredArray(
            self.runtime,
            result_field,
            shape=self.shape,
            dtype=dtype,
            scalar=self.scalar,
        )

        result.unary_op(
            UnaryOpCode.IMAG,
            result.dtype,
            self,
            True,
            [],
            stacklevel + 1,
            callsite,
        )

        return result

    # TODO: We should return a view of the field instead of a copy
    def real(self, stacklevel, callsite=None):
        dtype = _complex_field_dtype(self.dtype)
        if self.scalar:
            result_field = Future()
        else:
            result_field = self.runtime.allocate_field(self.shape, dtype=dtype)

        result = DeferredArray(
            self.runtime,
            result_field,
            shape=self.shape,
            dtype=dtype,
            scalar=self.scalar,
        )

        result.unary_op(
            UnaryOpCode.REAL,
            result.dtype,
            self,
            True,
            [],
            stacklevel + 1,
            callsite,
        )

        return result

    def conj(self, stacklevel, callsite=None):
        if self.scalar:
            result_field = Future()
        else:
            result_field = self.runtime.allocate_field(
                self.shape, dtype=self.dtype
            )

        result = DeferredArray(
            self.runtime,
            result_field,
            shape=self.shape,
            dtype=self.dtype,
            scalar=self.scalar,
        )

        result.unary_op(
            UnaryOpCode.CONJ,
            result.dtype,
            self,
            True,
            [],
            stacklevel + 1,
            callsite,
        )

        return result

    # Copy source array to the destination array
    def copy(self, rhs, deep, stacklevel, callsite=None):
        self.unary_op(
            UnaryOpCode.COPY,
            rhs.dtype,
            rhs,
            True,
            [],
            stacklevel + 1,
            callsite,
        )

    def get_scalar_array(self, stacklevel):
        assert self.size == 1
        # Look at the type of the data and figure out how to read this data
        # First four bytes are for the type code, so we need to skip those
        buf = self.base.get_buffer(self.dtype.itemsize + 8)[8:]
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

    def get_item(self, key, stacklevel, view=None, dim_map=None):
        assert self.size > 1
        # Check to see if this is advanced indexing or not
        if self._is_advanced_indexing(key):
            # Create the indexing array
            index_array = self._create_indexing_array(
                key, stacklevel=(stacklevel + 1)
            )
            # Create a new array to be the result
            result_field = self.runtime.allocate_field(
                index_array.shape, dtype=self.dtype
            )
            result = DeferredArray(
                self.runtime,
                result_field,
                shape=index_array.shape,
                dtype=self.dtype,
                scalar=False,
            )
            # Issue the gather copy to the result
            index = index_array.base
            launch_space = index.compute_parallel_launch_space()
            src = self.base
            dst = result_field
            if launch_space is not None:
                # Index copy launch
                if self.ndim == index_array.ndim:
                    # If we have the same dimensionality then normal
                    # partitioning works
                    (
                        src_part,
                        shardfn,
                        shardsp,
                    ) = src.find_or_create_key_partition()
                    src_proj = 0  # identity
                    index_part = index.find_or_create_congruent_partition(
                        src_part
                    )
                else:
                    # Otherwise we need to compute an indirect partition
                    # and functor
                    src_part, src_proj = src.find_or_create_indirect_partition(
                        launch_space
                    )
                    (
                        index_part,
                        shardfn,
                        shardsp,
                    ) = index.find_or_create_key_partition()
                copy = IndexCopy(
                    Rect(launch_space),
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
                if shardsp is not None:
                    copy.set_sharding_space(shardsp)
                # Partition the index array and the destination array
                # the same way
                dst_part = dst.find_or_create_congruent_partition(index_part)
                # Set this as the key partition
                dst.set_key_partition(dst_part, shardfn, shardsp)
                copy.add_src_requirement(
                    src_part, src.field.field_id, src_proj
                )
                copy.add_dst_requirement(
                    dst_part,
                    dst.field.field_id,
                    0,
                    tag=NumPyMappingTag.KEY_REGION_TAG,
                )
                copy.add_src_indirect_requirement(
                    index_part, index.field.field_id, 0
                )
            else:
                # Single copy launch
                shardpt, shardfn, shardsp = index.find_point_sharding()
                copy = Copy(mapper=self.runtime.mapper_id, tag=shardfn)
                if shardpt is not None:
                    copy.set_point(shardpt)
                if shardsp is not None:
                    copy.set_sharding_space(shardsp)
                copy.add_src_requirement(src.region, src.field.field_id)
                copy.add_dst_requirement(dst.region, dst.field.field_id)
                copy.add_src_indirect_requirement(
                    index.region, index.field.field_id
                )
            # Issue the copy to the runtime
            self.runtime.dispatch(copy)
        else:
            if view is None or dim_map is None:
                view, dim_map = self._get_view(key)
            new_shape = self._get_view_shape(view, dim_map)
            # If all the dimensions collapsed then we are just reading a value
            if new_shape != ():
                # Ask the runtime to make a new view onto this logical region
                result = self.runtime.find_or_create_view(
                    self.base, view, dim_map, new_shape, key
                )
                # Handle an unfortunate case where our subview can
                # accidentally collapse the last dimension
                if result.ndim == 0:
                    result.shape = new_shape
            else:  # This is just a value so read it
                use_key = tuple([x.start for x in view])
                src_arg = DeferredArrayView(
                    self, tag=NumPyMappingTag.NO_MEMOIZE_TAG
                )

                task = Map(self.runtime, NumPyOpCode.READ)
                task.add_point(use_key, untyped=True)
                src_arg.add_to_legate_op(task, True)
                future = task.execute_single()
                result = DeferredArray(
                    self.runtime,
                    base=future,
                    shape=(),
                    dtype=self.dtype,
                    scalar=True,
                )
        if self.runtime.shadow_debug:
            result.shadow = self.shadow.get_item(
                key, stacklevel=(stacklevel + 1), view=view, dim_map=dim_map
            )
        return result

    def set_item(self, key, rhs, stacklevel):
        value_array = self.runtime.to_deferred_array(
            rhs, stacklevel=(stacklevel + 1)
        )
        assert self.dtype == value_array.dtype
        # Check to see if this is advanced indexing or not
        if self._is_advanced_indexing(key):
            # Create the indexing array
            index_array = self._create_indexing_array(
                key, stacklevel=(stacklevel + 1)
            )
            if index_array.shape != value_array.shape:
                raise ValueError(
                    "Advanced indexing array does not match source shape"
                )
            # Do the scatter copy
            index = index_array.base
            launch_space = index.compute_parallel_launch_space()
            dst = self.base
            src = value_array.base
            if launch_space is not None:
                # Index copy launch
                if self.ndim == index_array.ndim:
                    # If we have the same dimensionality then normal
                    # partitioning works
                    (
                        dst_part,
                        shardfn,
                        shardsp,
                    ) = dst.find_or_create_key_partition()
                    dst_proj = 0  # identity
                    index_part = index.find_or_create_congruent_partition(
                        dst_part
                    )
                else:
                    # Otherwise we need to compute an indirect partition
                    # and functor
                    dst_part, dst_proj = dst.find_or_create_indirect_partition(
                        launch_space
                    )
                    (
                        index_part,
                        shardfn,
                        shardsp,
                    ) = index.find_or_create_key_partition()
                copy = IndexCopy(
                    Rect(launch_space),
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
                if shardsp is not None:
                    copy.set_sharding_space(shardsp)
                src_part = src.find_or_create_congruent_partition(index_part)
                copy.add_src_requirement(src_part, src.field.field_id, 0)
                copy.add_dst_requirement(
                    dst_part, dst.field.field_id, dst_proj
                )
                copy.add_dst_indirect_requirement(
                    index_part, index.field.field_id, 0
                )
            else:
                # Single copy launch
                point, shardfn, shardsp = index.find_point_sharding()
                copy = Copy(mapper=self.runtime.mapper_id, tag=shardfn)
                if point is not None:
                    copy.set_point(point)
                if shardsp is not None:
                    copy.set_sharding_space(shardsp)
                copy.add_src_requirement(src.region, src.field.field_id)
                copy.add_dst_requirement(dst.region, dst.field.field_id)
                copy.add_dst_indirect_requirement(
                    index.region, index.field.field_id
                )
            # Issue the copy to the runtime
            self.runtime.dispatch(copy)
        elif self.size == 1:
            assert value_array.size == 1
            # Special case of writing a single value
            # We can just copy the future because they are immutable
            self.base = value_array.base
        else:
            # Writing to a view of this array
            view, dim_map = self._get_view(key)
            # See what the shape of the view is
            new_shape = self._get_view_shape(view, dim_map)

            if new_shape == ():
                # We're just writing a single value
                assert value_array.size == 1
                use_key = tuple([x.start for x in view])
                assert len(use_key) == self.ndim
                dst_arg = DeferredArrayView(
                    self, tag=NumPyMappingTag.NO_MEMOIZE_TAG
                )
                value_arg = DeferredArrayView(value_array)

                task = Map(self.runtime, NumPyOpCode.WRITE)
                task.add_point(use_key, untyped=True)
                dst_arg.add_to_legate_op(task, False, read_write=True)
                value_arg.add_to_legate_op(task, True)
                task.execute_single()
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
                if (
                    new_shape == rhs.shape
                    and self.base.field == value_array.base.field
                ):
                    return

                # Get the view for the result
                subview = self.runtime.find_or_create_view(
                    self.base, view, dim_map, new_shape, key
                )
                # Handle an unfortunate case where our subview can
                # accidentally collapse the last dimension
                if subview.ndim == 0:
                    subview.shape = new_shape
                # See if the value is a scalar
                if value_array.size == 1:
                    # Scalar so we can do this with a fill
                    subview.fill(
                        value_array.__numpy_array__(
                            stacklevel=(stacklevel + 1)
                        ),
                        stacklevel=(stacklevel + 1),
                    )
                else:
                    subview.copy(
                        value_array, stacklevel=(stacklevel + 1), deep=False
                    )
        if self.runtime.shadow_debug:
            self.shadow.set_item(key, rhs.shadow, stacklevel=(stacklevel + 1))
            self.runtime.check_shadow(self, "set_item")

    def reshape(self, newshape, order, stacklevel):
        assert isinstance(newshape, tuple)
        # Check to see if we can make an affine mapping that maps points
        # in the new shape into points in the old shape
        transform = None
        if len(newshape) >= self.ndim:
            if order == "C" or order == "A":
                # See if we can group the dimensions from right to left
                # in a way that creates dimensions in the old shape
                # TODO: there are better ways to do this with prime
                # factorization so we can handle weird splits between
                # dimensions
                transform = Transform(self.ndim, len(newshape), False)
                dim = len(newshape) - 1
                for idx in xrange(self.ndim - 1, -1, -1):
                    stride = 1
                    current_dim = self.shape[idx]
                    while stride < current_dim:
                        assert dim >= 0
                        if stride * newshape[dim] <= current_dim:
                            if newshape[dim] > 1:
                                transform.trans[idx][dim] = stride
                                stride *= newshape[dim]
                            dim -= 1
                        else:
                            # We failed, so we're done
                            transform = None
                            break
                    if transform is None:
                        break
            if order == "F" or (order == "A" and transform is None):
                # See if we can group the dimensions from left to right in
                # a way that creates dimensions in the old shape
                # TODO; think about how this interoperates with partitioning
                pass
        if transform is None:
            # If we don't have a transform then we need to make a copy
            warnings.warn(
                "legate.numpy has not implemented reshape/ravel for newshape "
                + str(newshape)
                + " which is a non-affine mapping and is falling back to "
                + "canonical numpy. You may notice significantly decreased "
                + "performance for this function call.",
                stacklevel=(stacklevel + 1),
                category=RuntimeWarning,
            )
            numpy_array = self.__numpy_array__(stacklevel=(stacklevel + 1))
            # Force a copy here because we know we can't build a view
            result_array = numpy_array.reshape(newshape, order=order).copy()
            result = self.runtime.get_numpy_thunk(
                result_array, stacklevel=(stacklevel + 1)
            )
        else:
            # If we have a transform then we can make a view
            affine_transform = AffineTransform(self.ndim, len(newshape), False)
            affine_transform.trans = transform.trans
            result = self.runtime.create_transform_view(
                self.base, newshape, affine_transform
            )
        if self.runtime.shadow_debug:
            result.shadow = self.shadow.reshape(
                newshape, order, stacklevel=(stacklevel + 1)
            )
        return result

    def squeeze(self, axis, stacklevel):
        if axis is None:
            new_shape = ()
            axis = ()
            for idx in xrange(self.ndim):
                if self.shape[idx] != 1:
                    new_shape = new_shape + (self.shape[idx],)
                else:
                    # Record the dropped axis index
                    axis = axis + (idx,)
        elif isinstance(axis, int):
            new_shape = ()
            for idx in xrange(self.ndim):
                if idx == axis:
                    assert self.shape[idx] == 1
                else:
                    new_shape = new_shape + (self.shape[idx],)
            # Convert to a tuple of dropped axes
            axis = (axis,)
        elif isinstance(axis, tuple):
            new_shape = ()
            for idx in xrange(self.ndim):
                if idx in axis:
                    assert self.shape[idx] == 1
                else:
                    new_shape = new_shape + (self.shape[idx],)
        else:
            raise TypeError(
                '"axis" argument for squeeze must be int-like or tuple-like'
            )
        if not self.scalar:
            # Make transform for the lost dimensions
            transform = AffineTransform(self.ndim, len(new_shape), False)
            child_idx = 0
            for parent_idx in xrange(self.ndim):
                # If this is a collapsed dimension then record it
                if parent_idx not in axis:
                    transform.trans[parent_idx, child_idx] = 1
                    child_idx += 1
            assert child_idx == len(new_shape)
            result = self.runtime.create_transform_view(
                self.base, new_shape, transform
            )
        else:
            # Easy case of size 1 array, nothing to do
            result = DeferredArray(
                self.runtime, self.base, new_shape, self.dtype, True
            )
        if self.runtime.shadow_debug:
            result.shadow = self.shadow.squeeze(
                axis, stacklevel=(stacklevel + 1)
            )
        return result

    def swapaxes(self, axis1, axis2, stacklevel):
        if self.size == 1:
            return self
        # Make a new deferred array object and swap the results
        assert axis1 < self.ndim and axis2 < self.ndim
        # Create a new shape and transform for the region field object
        new_shape = ()
        for idx in xrange(self.ndim):
            if idx == axis1:
                new_shape = new_shape + (self.shape[axis2],)
            elif idx == axis2:
                new_shape = new_shape + (self.shape[axis1],)
            else:
                new_shape = new_shape + (self.shape[idx],)
        transform = AffineTransform(self.ndim, self.ndim, True)
        transform.transform[axis1, axis1] = 0
        transform.transform[axis2, axis2] = 0
        transform.transform[axis1, axis2] = 1
        transform.transform[axis2, axis1] = 1
        result = self.runtime.create_transform_view(
            self.base, new_shape, transform
        )
        if self.runtime.shadow_debug:
            result.shadow = self.shadow.swapaxes(
                axis1, axis2, stacklevel=(stacklevel + 1)
            )
        return result

    # Convert the source array to the destination array
    def convert(self, rhs, stacklevel, warn=True, callsite=None):
        lhs_array = self
        rhs_array = self.runtime.to_deferred_array(
            rhs, stacklevel=(stacklevel + 1)
        )
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

        lhs_arg = DeferredArrayView(lhs_array)
        rhs_arg = DeferredArrayView(rhs_array)

        launch_space, key_arg = lhs_arg.find_key_view(rhs_arg)
        key_arg.update_tag(NumPyMappingTag.KEY_REGION_TAG)
        rhs_arg = rhs_arg.broadcast(lhs_arg)

        if launch_space is not None:
            lhs_arg = lhs_arg.align_partition(key_arg)
            rhs_arg = rhs_arg.align_partition(key_arg)
            if lhs_arg is not key_arg:
                lhs_arg.copy_key_partition_from(key_arg)

        if rhs_arg.scalar:
            task_id = NumPyOpCode.SCALAR_CONVERT
        else:
            task_id = NumPyOpCode.CONVERT

        (shardpt, shardfn, shardsp) = key_arg.sharding

        task = Map(self.runtime, task_id, tag=shardfn)
        if not rhs_arg.scalar:
            if launch_space is not None:
                task.add_shape(lhs_arg.shape, lhs_arg.part.tile_shape, 0)
            else:
                task.add_shape(lhs_arg.shape)

            lhs_arg.add_to_legate_op(task, False)
        else:
            task.add_dtype_arg(lhs_array.dtype)
        rhs_arg.add_to_legate_op(task, True)

        if shardsp is not None:
            task.set_sharding_space(shardsp)

        # See if we are doing index space launches or not
        if not rhs_arg.scalar and launch_space is not None:
            task.execute(Rect(launch_space))
        else:
            if shardpt is not None:
                task.set_point(shardpt)
            result = task.execute_single()

            if rhs_arg.scalar:
                lhs_array.base = result

        self.runtime.profile_callsite(stacklevel + 1, True, callsite)
        # See if we are doing shadow debugging
        if self.runtime.shadow_debug:
            self.shadow.convert(
                rhs.shadow, stacklevel=(stacklevel + 1), warn=False
            )
            self.runtime.check_shadow(self, "convert")

    # Fill the legate array with the value in the numpy array
    def _fill(self, value, stacklevel, callsite=None):
        if self.size == 1:
            # Handle the 0D case special
            self.base = value
        else:
            assert self.base is not None
            dst = self.base
            dtype = self.dtype
            # If this is a fill for an arg value, make sure to pass
            # the value dtype so that we get it packed correctly
            if dtype.kind == "V":
                dtype = get_arg_value_dtype(dtype)
            else:
                dtype = None
            launch_space = dst.compute_parallel_launch_space()
            if launch_space is not None:
                (
                    dst_part,
                    shardfn,
                    shardsp,
                ) = dst.find_or_create_key_partition()
                # Need to use a fake dtype for fills with arg values
                lhs = DeferredArrayView(self, part=dst_part, dtype=dtype)
            else:
                shardpt, shardfn, shardsp = dst.find_point_sharding()
                # Need to use a fake dtype for fills with arg values
                lhs = DeferredArrayView(self, dtype=dtype)

            op = Map(self.runtime, NumPyOpCode.FILL, tag=shardfn)
            if launch_space is not None:
                op.add_shape(dst.shape, dst_part.tile_shape, 0)
            else:
                op.add_shape(dst.shape)

            lhs.add_to_legate_op(op, False)
            op.add_future(value)
            if shardsp is not None:
                op.set_sharding_space(shardsp)
            if launch_space is not None:
                op.execute(Rect(launch_space))
            else:
                if shardpt is not None:
                    fill.set_point(shardpt)
                op.execute_single()
            self.runtime.profile_callsite(stacklevel + 1, True, callsite)

    def fill(self, numpy_array, stacklevel, callsite=None):
        assert isinstance(numpy_array, np.ndarray)
        assert numpy_array.size == 1
        assert self.dtype == numpy_array.dtype
        # Have to copy the numpy array because this launch is asynchronous
        # and we need to make sure the application doesn't mutate the value
        # so make a future result, this is immediate so no dependence
        value = self.runtime.create_scalar(numpy_array.data, self.dtype)
        self._fill(value, stacklevel + 1, callsite=callsite)
        # See if we are doing shadow debugging
        if self.size > 1 and self.runtime.shadow_debug:
            self.shadow.fill(numpy_array, stacklevel=(stacklevel + 1))
            self.runtime.check_shadow(self, "fill")

    def dot(self, src1, src2, stacklevel, callsite=None):
        rhs1_array = self.runtime.to_deferred_array(
            src1, stacklevel=(stacklevel + 1)
        )
        rhs2_array = self.runtime.to_deferred_array(
            src2, stacklevel=(stacklevel + 1)
        )
        lhs_array = self

        def to_proj_id(proj):
            return self.runtime.first_proj_id + proj if proj > 0 else proj

        if rhs1_array.ndim == 1 and rhs2_array.ndim == 1:
            # Vector dot product case
            assert lhs_array.size == 1
            assert rhs1_array.shape == rhs2_array.shape or (
                rhs1_array.size == 1 and rhs2_array.size == 1
            )
            rhs1_arg = DeferredArrayView(rhs1_array)
            rhs2_arg = DeferredArrayView(rhs2_array)

            launch_space = rhs1_array.base.compute_parallel_launch_space()
            if launch_space is not None:
                rhs1_arg.update_tag(NumPyMappingTag.KEY_REGION_TAG)
                rhs2_arg.copy_key_partition_from(rhs1_arg)

            (shardpt, shardfn, shardsp) = rhs1_arg.sharding

            task = Map(self.runtime, NumPyOpCode.DOT, tag=shardfn)
            if launch_space is not None:
                task.add_shape(rhs1_arg.shape, rhs1_arg.part.tile_shape, 0)
            else:
                task.add_shape(rhs1_arg.shape)

            rhs1_arg.add_to_legate_op(task, True)
            rhs2_arg.add_to_legate_op(task, True)

            if shardsp is not None:
                task.set_sharding_space(shardsp)

            if launch_space is not None:
                redop = self.runtime.get_scalar_reduction_op_id(
                    UnaryRedCode.SUM
                )
                result = task.execute(Rect(launch_space), redop=redop)
            else:
                if shardpt is not None:
                    task.set_point(shardpt)
                result = task.execute_single()

            lhs_array.base = result
        elif rhs1_array.ndim == 1 or rhs2_array.ndim == 1:
            # Matrix-vector or vector-matrix multiply
            assert lhs_array.ndim == 1
            assert rhs1_array.ndim == 2 or rhs2_array.ndim == 2
            # We're going to do index launches over the matrix so first
            # we need to find the matrix, see if it is the first array
            # or the second
            if rhs1_array.ndim == 2:
                matrix = rhs1_array
                left_matrix = True
            else:
                assert rhs2_array.ndim == 2
                matrix = rhs2_array
                left_matrix = False
            launch_space = self.runtime.compute_parallel_launch_space_by_shape(
                matrix.shape
            )
            if launch_space is not None:
                # Parallel launch space case
                # This optimization involves a transpose which is actually
                # very expensive usually so we're disabling it for now,
                # maybe someone will come up with a use for it later
                # if matrix.shape[0] == matrix.shape[1]:
                #    # A small optimization here for square matrices to get
                #    # better dimension tiling based on whether this is a
                #    # right matrix or a left matrix multiply
                #    if left_matrix:
                #        if launch_space[1] > launch_space[0]:
                #            launch_space = (launch_space[1], launch_space[0])
                #    else:
                #        if launch_space[0] > launch_space[1]:
                #            launch_space = (launch_space[1], launch_space[0])
                # Partition the into the same number of pieces as the matrix
                lhs_part = lhs_array.base.find_or_create_partition(
                    (launch_space[0] if left_matrix else launch_space[1],)
                )
                lhs_proj = (
                    NumPyProjCode.PROJ_2D_1D_X
                    if left_matrix
                    else NumPyProjCode.PROJ_2D_1D_Y
                )
                lhs_tag = NumPyMappingTag.NO_MEMOIZE_TAG
                if rhs1_array.ndim == 1:
                    # Row input, divide rows of matrix
                    rhs1_part = rhs1_array.base.find_or_create_partition(
                        (launch_space[0],)
                    )
                    rhs1_proj = NumPyProjCode.PROJ_2D_1D_X
                    rhs1_tag = NumPyMappingTag.NO_MEMOIZE_TAG
                else:
                    # Matrix input
                    assert rhs1_array.shape[0] > 1
                    rhs1_part = rhs1_array.base.find_or_create_partition(
                        launch_space
                    )
                    rhs1_proj = 0  # Identity projection matrix
                    rhs1_tag = NumPyMappingTag.KEY_REGION_TAG
                if rhs2_array.ndim == 1:
                    # Column input, divide columns of matrix
                    rhs2_part = rhs2_array.base.find_or_create_partition(
                        (launch_space[1],)
                    )
                    rhs2_proj = NumPyProjCode.PROJ_2D_1D_Y
                    rhs2_tag = NumPyMappingTag.NO_MEMOIZE_TAG
                else:
                    assert rhs2_array.shape[1] > 1
                    # Matrix input
                    rhs2_part = rhs2_array.base.find_or_create_partition(
                        launch_space
                    )
                    rhs2_proj = 0  # Identity projection matrix
                    rhs2_tag = NumPyMappingTag.KEY_REGION_TAG

                lhs_arg = DeferredArrayView(
                    lhs_array,
                    part=lhs_part,
                    proj_id=to_proj_id(lhs_proj),
                    tag=lhs_tag,
                )
                rhs1_arg = DeferredArrayView(
                    rhs1_array,
                    part=rhs1_part,
                    proj_id=to_proj_id(rhs1_proj),
                    tag=rhs1_tag,
                )
                rhs2_arg = DeferredArrayView(
                    rhs2_array,
                    part=rhs2_part,
                    proj_id=to_proj_id(rhs2_proj),
                    tag=rhs2_tag,
                )
                needs_reduction = (left_matrix and launch_space[1] > 1) or (
                    not left_matrix and launch_space[0] > 1
                )
            else:
                lhs_arg = DeferredArrayView(lhs_array)
                rhs1_arg = DeferredArrayView(rhs1_array)
                rhs2_arg = DeferredArrayView(rhs2_array)
                needs_reduction = False

            # If the inputs are 16-bit floats, we should use 32-bit float
            # for accumulation
            if needs_reduction and rhs1_arg.dtype == np.float16:
                acc_buffer_dtype = np.dtype(np.float32)
                acc_buffer = self.runtime.allocate_field(
                    lhs_array.shape, acc_buffer_dtype
                )
                acc_array = DeferredArray(
                    self.runtime,
                    acc_buffer,
                    shape=lhs_array.shape,
                    dtype=acc_buffer_dtype,
                    scalar=False,
                )
                acc_part = acc_buffer.find_or_create_partition(
                    (launch_space[0] if left_matrix else launch_space[1],)
                )
                acc_arg = DeferredArrayView(
                    acc_array,
                    part=acc_part,
                    proj_id=to_proj_id(lhs_proj),
                    tag=lhs_tag,
                )

                lhs_array = acc_array
                lhs_arg = acc_arg

            if needs_reduction:
                lhs_array.fill(
                    np.array(0, dtype=lhs_array.dtype),
                    stacklevel=(stacklevel + 1),
                )

            task = Map(self.runtime, NumPyOpCode.MATVECMUL)
            task.add_scalar_arg(needs_reduction, bool)
            if launch_space is not None:
                task.add_shape(lhs_array.shape, lhs_part.tile_shape, lhs_proj)
                task.add_shape(
                    rhs1_array.shape, rhs1_part.tile_shape, rhs1_proj
                )
                task.add_shape(
                    rhs2_array.shape, rhs2_part.tile_shape, rhs2_proj
                )
            else:
                task.add_shape(lhs_array.shape)
                task.add_shape(rhs1_array.shape)
                task.add_shape(rhs2_array.shape)

            if needs_reduction:
                redop = self.runtime.get_unary_reduction_op_id(
                    UnaryRedCode.SUM, lhs_array.dtype
                )
            else:
                redop = None

            lhs_arg.add_to_legate_op(task, False, redop=redop)
            rhs1_arg.add_to_legate_op(task, True)
            rhs2_arg.add_to_legate_op(task, True)

            if launch_space is not None:
                task.execute(Rect(launch_space))
            else:
                task.execute_single()

            # If we used an accumulation buffer, we should copy the results
            # back to the lhs
            if needs_reduction and rhs1_arg.dtype == np.float16:
                self.convert(
                    lhs_array, stacklevel + 1, warn=False, callsite=callsite
                )

        elif rhs1_array.ndim == 2 and rhs2_array.ndim == 2:
            # Matrix-matrix multiply
            M = lhs_array.shape[0]
            N = lhs_array.shape[1]
            K = rhs1_array.shape[1]
            assert M == rhs1_array.shape[0]  # Check M
            assert N == rhs2_array.shape[1]  # Check N
            assert K == rhs2_array.shape[0]  # Check K
            # We need to figure out our strategy for matrix multiple, we can
            # figure this out using a deterministic process of decisions based
            # on the natural tiling of the largest of the three matrices
            lhs_size = M * N
            rhs1_size = M * K
            rhs2_size = K * N
            max_size = max(lhs_size, rhs1_size, rhs2_size)
            if lhs_size == max_size:
                # LHS is biggest
                lhs_launch = lhs_array.base.compute_parallel_launch_space()
                if lhs_launch is not None:
                    # This gives us the tile size for m and n
                    lhs_part = lhs_array.base.find_or_create_partition(
                        lhs_launch
                    )
                    # the choice for k and k_tile here is underconstrained, we
                    # want to be able to recognize inner and outer product like
                    # cases here, but also fall back to normal tiled execution
                    # when it's important, we'll try to make things squar-ish
                    # by making k_tile the min of the m_tile and n_tile sizes
                    m_tile = lhs_part.tile_shape[0]
                    n_tile = lhs_part.tile_shape[1]
                    k_tile = min(m_tile, n_tile)
                    rhs1_launch = (
                        (rhs1_array.shape[0] + m_tile - 1) // m_tile,
                        (rhs1_array.shape[1] + k_tile - 1) // k_tile,
                    )
                    rhs1_part = rhs1_array.base.find_or_create_partition(
                        rhs1_launch, tile_shape=(m_tile, k_tile)
                    )
                    rhs2_launch = (
                        (rhs2_array.shape[0] + k_tile - 1) // k_tile,
                        (rhs2_array.shape[1] + n_tile - 1) // n_tile,
                    )
                    rhs2_part = rhs2_array.base.find_or_create_partition(
                        rhs2_launch, tile_shape=(k_tile, n_tile)
                    )
                else:
                    # No need for any parallelism
                    rhs1_launch = None
            elif rhs1_size == max_size:
                # RHS1 is the biggest
                rhs1_launch = rhs1_array.base.compute_parallel_launch_space()
                if rhs1_launch is not None:
                    # This gives us the tile size for k
                    rhs1_part = rhs1_array.base.find_or_create_partition(
                        rhs1_launch
                    )
                    m_tile = rhs1_part.tile_shape[0]
                    k_tile = rhs1_part.tile_shape[1]
                    # See if we want to make this inner-product like or not
                    if rhs1_launch[0] == 1:
                        n_tile = lhs_array.shape[1]
                        lhs_launch = (1, 1)
                        rhs2_launch = (rhs1_launch[1], 1)
                    else:
                        # Not inner-product like so compute a partitioning of
                        # the output array that will be easy to tile for
                        # reductions
                        lhs_launch = rhs1_launch
                        n_tile = (
                            lhs_array.shape[1] + lhs_launch[1] - 1
                        ) // lhs_launch[1]
                        rhs2_launch = (rhs1_launch[1], rhs1_launch[1])
                    rhs2_part = rhs2_array.base.find_or_create_partition(
                        rhs2_launch, tile_shape=(k_tile, n_tile)
                    )
                    lhs_part = lhs_array.base.find_or_create_partition(
                        lhs_launch, tile_shape=(m_tile, n_tile)
                    )
                else:
                    # No need for any parallelism
                    rhs1_launch = None
            else:
                assert rhs2_size == max_size
                # RHS2 is the biggest
                rhs2_launch = rhs2_array.base.compute_parallel_launch_space()
                if rhs2_launch is not None:
                    # This gives us the tile size for k
                    rhs2_part = rhs2_array.base.find_or_create_partition(
                        rhs2_launch
                    )
                    k_tile = rhs2_part.tile_shape[0]
                    n_tile = rhs2_part.tile_shape[1]
                    # See if we want to make this inner-product like or not
                    if rhs2_launch[1] == 1:
                        m_tile = lhs_array.shape[0]
                        lhs_launch = (1, 1)
                        rhs1_launch = (1, rhs2_launch[0])
                    else:
                        # Not inner-product like so compute the natural
                        # partitioning of the output array so we can get a
                        # partitioning scheme for m
                        lhs_launch = rhs2_launch
                        m_tile = (
                            lhs_array.shape[0] + lhs_launch[0] - 1
                        ) // lhs_launch[0]
                        rhs1_launch = (rhs2_launch[0], rhs2_launch[0])
                    rhs1_part = rhs1_array.base.find_or_create_partition(
                        rhs1_launch, tile_shape=(m_tile, k_tile)
                    )
                    lhs_part = lhs_array.base.find_or_create_partition(
                        lhs_launch, tile_shape=(m_tile, n_tile)
                    )
                else:
                    # No need for any parallelism
                    rhs1_launch = None

            if rhs1_launch is not None:
                # Parallel launch case
                assert rhs2_launch is not None
                assert lhs_launch[0] == rhs1_launch[0]
                assert rhs1_launch[1] == rhs2_launch[0]
                assert rhs2_launch[1] == lhs_launch[1]
                launch_space = (
                    lhs_launch[0],
                    rhs1_launch[1],
                    lhs_launch[1],
                )
                lhs_proj = NumPyProjCode.PROJ_3D_2D_XZ
                rhs1_proj = NumPyProjCode.PROJ_3D_2D_XY
                rhs2_proj = NumPyProjCode.PROJ_3D_2D_YZ
                lhs_tag = (
                    NumPyMappingTag.KEY_REGION_TAG
                    if lhs_size == max_size
                    else NumPyMappingTag.NO_MEMOIZE_TAG
                )
                rhs1_tag = (
                    NumPyMappingTag.KEY_REGION_TAG
                    if rhs1_size == max_size
                    else NumPyMappingTag.NO_MEMOIZE_TAG
                )
                rhs2_tag = (
                    NumPyMappingTag.KEY_REGION_TAG
                    if rhs2_size == max_size
                    else NumPyMappingTag.NO_MEMOIZE_TAG
                )

                lhs_arg = DeferredArrayView(
                    lhs_array,
                    part=lhs_part,
                    proj_id=to_proj_id(lhs_proj),
                    tag=lhs_tag,
                )
                rhs1_arg = DeferredArrayView(
                    rhs1_array,
                    part=rhs1_part,
                    proj_id=to_proj_id(rhs1_proj),
                    tag=rhs1_tag,
                )
                rhs2_arg = DeferredArrayView(
                    rhs2_array,
                    part=rhs2_part,
                    proj_id=to_proj_id(rhs2_proj),
                    tag=rhs2_tag,
                )

                needs_reduction = rhs1_launch[1] > 1

            else:
                launch_space = None
                lhs_arg = DeferredArrayView(lhs_array)
                rhs1_arg = DeferredArrayView(rhs1_array)
                rhs2_arg = DeferredArrayView(rhs2_array)
                needs_reduction = False

            # If the inputs are 16-bit floats, we should use 32-bit float
            # for accumulation
            if needs_reduction and rhs1_arg.dtype == np.float16:
                acc_buffer_dtype = np.dtype(np.float32)
                acc_buffer = self.runtime.allocate_field(
                    lhs_array.shape, acc_buffer_dtype
                )
                acc_array = DeferredArray(
                    self.runtime,
                    acc_buffer,
                    shape=lhs_array.shape,
                    dtype=acc_buffer_dtype,
                    scalar=False,
                )
                acc_part = acc_buffer.find_or_create_partition(
                    lhs_launch, tile_shape=(m_tile, n_tile)
                )
                acc_arg = DeferredArrayView(
                    acc_array,
                    part=acc_part,
                    proj_id=to_proj_id(lhs_proj),
                    tag=lhs_tag,
                )

                lhs_array = acc_array
                lhs_arg = acc_arg

            # If we perform reduction for matrix multiplication,
            # we must zero out the lhs first
            if needs_reduction:
                lhs_array.fill(
                    np.array(0, dtype=lhs_array.dtype),
                    stacklevel=(stacklevel + 1),
                )

            task = Map(self.runtime, NumPyOpCode.MATMUL)
            task.add_scalar_arg(needs_reduction, bool)
            if launch_space is not None:
                task.add_shape(lhs_array.shape, lhs_part.tile_shape, lhs_proj)
                task.add_shape(
                    rhs1_array.shape, rhs1_part.tile_shape, rhs1_proj
                )
                task.add_shape(
                    rhs2_array.shape, rhs2_part.tile_shape, rhs2_proj
                )
            else:
                task.add_shape(lhs_array.shape)
                task.add_shape(rhs1_array.shape)
                task.add_shape(rhs2_array.shape)

            if needs_reduction:
                redop = self.runtime.get_unary_reduction_op_id(
                    UnaryRedCode.SUM, lhs_array.dtype
                )
            else:
                redop = None

            lhs_arg.add_to_legate_op(task, False, redop=redop)
            rhs1_arg.add_to_legate_op(task, True)
            rhs2_arg.add_to_legate_op(task, True)

            if launch_space is not None:
                task.execute(Rect(launch_space))
            else:
                task.execute_single()

            # If we used an accumulation buffer, we should copy the results
            # back to the lhs
            if needs_reduction and rhs1_arg.dtype == np.float16:
                self.convert(
                    lhs_array, stacklevel + 1, warn=False, callsite=callsite
                )
        else:
            raise NotImplementedError("Need support for tensor contractions")

        self.runtime.profile_callsite(stacklevel + 1, True, callsite)
        if self.runtime.shadow_debug:
            self.shadow.dot(
                src1.shadow, src2.shadow, stacklevel=(stacklevel + 1)
            )
            self.runtime.check_shadow(self, op)

    # Create or extract a diagonal from a matrix
    def diag(self, rhs, extract, k, stacklevel, callsite=None):
        if extract:
            matrix_array = self.runtime.to_deferred_array(
                rhs, stacklevel=(stacklevel + 1)
            )
            diag_array = self
        else:
            matrix_array = self
            diag_array = self.runtime.to_deferred_array(
                rhs, stacklevel=(stacklevel + 1)
            )

        assert diag_array.ndim == 1
        assert matrix_array.ndim == 2
        assert diag_array.shape[0] <= min(
            matrix_array.shape[0], matrix_array.shape[1]
        )
        assert rhs.dtype == self.dtype

        launch_space = matrix_array.base.compute_parallel_launch_space()

        if launch_space is not None:
            # Find a partition for diagonal,
            # partition on the smaller tile size since that represents the
            # upper bound on the number of elements needed
            (
                matrix_part,
                shardfn,
                shardsp,
            ) = matrix_array.base.find_or_create_key_partition()

            if matrix_array.shape[0] < matrix_array.shape[1]:
                collapse_dim = 1
                color_space = (launch_space[0],)
                tile_shape = (matrix_part.tile_shape[0],)
                offset = (k if k < 0 else 0,)
                proj = NumPyProjCode.PROJ_2D_1D_X
            else:
                collapse_dim = 0
                color_space = (launch_space[1],)
                tile_shape = (matrix_part.tile_shape[1],)
                offset = (-k if k > 0 else 0,)
                proj = NumPyProjCode.PROJ_2D_1D_Y

            diag_part = diag_array.base.find_or_create_partition(
                color_space,
                tile_shape,
                offset,
            )
            diag_proj = self.runtime.first_proj_id + proj

            matrix_arg = DeferredArrayView(matrix_array, part=matrix_part)
            diag_arg = DeferredArrayView(
                diag_array, part=diag_part, proj_id=diag_proj
            )
        else:
            matrix_arg = DeferredArrayView(matrix_array)
            diag_arg = DeferredArrayView(diag_array)

        matrix_arg.update_tag(NumPyMappingTag.KEY_REGION_TAG)
        if not extract:
            diag_arg.update_tag(NumPyMappingTag.NO_MEMOIZE_TAG)

        (shardpt, shardfn, shardsp) = matrix_arg.sharding

        needs_reduction = (
            extract
            and launch_space is not None
            and launch_space[collapse_dim] > 1
        )

        if needs_reduction:
            # If we need reductions to extract the diagonal,
            # we have to issue a fill operation to get the output initialized.
            diag_array.fill(
                np.array(0, dtype=diag_array.dtype),
                stacklevel=(stacklevel + 1),
            )
        elif not extract:
            # Before we populate the diagonal, we have to issue a fill
            # operation to initialize the matrix with zeros
            matrix_array.fill(
                np.array(0, dtype=matrix_array.dtype),
                stacklevel=(stacklevel + 1),
            )

        task = Map(self.runtime, NumPyOpCode.DIAG, tag=shardfn)
        task.add_scalar_arg(extract, bool)
        task.add_scalar_arg(needs_reduction, bool)
        task.add_scalar_arg(k, np.int32)
        if launch_space is not None:
            task.add_shape(matrix_arg.shape, matrix_arg.part.tile_shape, 0)
        else:
            task.add_shape(matrix_arg.shape)

        if extract:
            if needs_reduction:
                redop = self.runtime.get_unary_reduction_op_id(
                    UnaryRedCode.SUM, diag_arg.dtype
                )
            else:
                redop = None

            diag_arg.add_to_legate_op(task, False, redop=redop)
            matrix_arg.add_to_legate_op(task, True)
        else:
            matrix_arg.add_to_legate_op(task, False, read_write=True)
            diag_arg.add_to_legate_op(task, True)

        if launch_space is not None:
            task.execute(Rect(launch_space))
        else:
            task.execute_single()

        self.runtime.profile_callsite(stacklevel + 1, True, callsite)
        # See if we are doing shadow debugging
        if self.runtime.shadow_debug:
            self.shadow.diag(
                rhs=rhs.shadow,
                extract=extract,
                k=k,
                stacklevel=(stacklevel + 1),
            )
            self.runtime.check_shadow(self, "diag")

    # Create an identity array with the ones offset from the diagonal by k
    def eye(self, k, stacklevel, callsite=None):
        assert self.ndim == 2  # Only 2-D arrays should be here
        # First issue a fill to zero everything out
        self.fill(np.array(0, dtype=self.dtype), stacklevel=(stacklevel + 1))

        lhs_arg = DeferredArrayView(self)
        launch_space, key_arg = lhs_arg.find_key_view()
        key_arg.update_tag(NumPyMappingTag.KEY_REGION_TAG)

        (shardpt, shardfn, shardsp) = key_arg.sharding

        task = Map(self.runtime, NumPyOpCode.EYE, tag=shardfn)

        if launch_space is not None:
            task.add_shape(lhs_arg.shape, lhs_arg.part.tile_shape, 0)
        else:
            task.add_shape(lhs_arg.shape)
        lhs_arg.add_to_legate_op(task, False)
        task.add_scalar_arg(k, np.int32)

        if launch_space is not None:
            task.execute(Rect(launch_space))
        else:
            task.execute_single()

        self.runtime.profile_callsite(stacklevel + 1, True, callsite)
        # See if we are doing shadow debugging
        if self.runtime.shadow_debug:
            self.shadow.eye(k=k, stacklevel=(stacklevel + 1))
            self.runtime.check_shadow(self, "eye")

    def arange(self, start, stop, step, stacklevel, callsite=None):
        assert self.ndim == 1  # Only 1-D arrays should be here
        dst = self.base
        if isinstance(dst, Future):
            # Handle the special case of a single values here
            assert self.shape[0] == 1
            array = np.array(start, dtype=self.dtype)
            dst.set_value(self.runtime.runtime, array.data, array.nbytes)
            # See if we are doing shadow debugging
            if self.runtime.shadow_debug:
                self.shadow.eye(k=k, stacklevel=(stacklevel + 1))
                self.runtime.check_shadow(self, "arange")
            return

        def create_scalar(value, dtype):
            numpy_array = np.array(value, dtype)
            return self.runtime.create_scalar(
                numpy_array.data, numpy_array.dtype
            )

        lhs_arg = DeferredArrayView(self)
        launch_space, key_arg = lhs_arg.find_key_view()
        key_arg.update_tag(NumPyMappingTag.KEY_REGION_TAG)

        (shardpt, shardfn, shardsp) = key_arg.sharding

        task = Map(self.runtime, NumPyOpCode.ARANGE, tag=shardfn)

        if launch_space is not None:
            task.add_shape(lhs_arg.shape, lhs_arg.part.tile_shape, 0)
        else:
            task.add_shape(lhs_arg.shape)
        lhs_arg.add_to_legate_op(task, False)

        task.add_future(create_scalar(start, self.dtype))
        task.add_future(create_scalar(stop, self.dtype))
        task.add_future(create_scalar(step, self.dtype))

        if launch_space is not None:
            task.execute(Rect(launch_space))
        else:
            task.execute_single()

        self.runtime.profile_callsite(stacklevel + 1, True, callsite)
        # See if we are doing shadow debugging
        if self.runtime.shadow_debug:
            self.shadow.eye(k=k, stacklevel=(stacklevel + 1))
            self.runtime.check_shadow(self, "arange")

    # Tile the src array onto the destination array
    def tile(self, rhs, reps, stacklevel, callsite=None):
        src_array = self.runtime.to_deferred_array(
            rhs, stacklevel=(stacklevel + 1)
        )
        dst_array = self
        assert src_array.ndim <= dst_array.ndim
        assert src_array.dtype == dst_array.dtype
        if src_array.size == 1:
            self._fill(src_array.base, stacklevel + 1, callsite=callsite)
            return

        dst_arg = DeferredArrayView(dst_array)
        src_arg = DeferredArrayView(src_array, proj_id=None)

        launch_space = dst_arg.compute_launch_space()

        (shardpt, shardfn, shardsp) = dst_arg.sharding

        task = Map(self.runtime, NumPyOpCode.TILE, tag=shardfn)

        if launch_space is not None:
            task.add_shape(dst_arg.shape, dst_arg.part.tile_shape, 0)
        else:
            task.add_shape(dst_arg.shape)
        task.add_shape(src_arg.shape)
        dst_arg.add_to_legate_op(task, False)
        src_arg.add_to_legate_op(task, True)

        if launch_space is not None:
            task.execute(Rect(launch_space))
        else:
            task.execute_single()

        self.runtime.profile_callsite(stacklevel + 1, True, callsite)
        if self.runtime.shadow_debug:
            self.shadow.tile(rhs.shadow, reps, stacklevel=(stacklevel + 1))
            self.runtime.check_shadow(self, "tile")

    # Transpose the matrix dimensions
    def transpose(self, rhs, axes, stacklevel, callsite=None):
        rhs_array = self.runtime.to_deferred_array(
            rhs, stacklevel=(stacklevel + 1)
        )
        lhs_array = self
        assert lhs_array.dtype == rhs_array.dtype
        assert lhs_array.ndim == rhs_array.ndim
        assert lhs_array.ndim == len(axes)
        # We don't support some cases right now
        if lhs_array.ndim > 2:
            raise NotImplementedError(
                "legate.numpy only supports standard 2-D transposes for now"
            )
        launch_space = lhs_array.base.compute_parallel_launch_space()
        if launch_space is not None:
            lhs_part, _, _ = lhs_array.base.find_or_create_key_partition()

            # Compute the partition using the transposed launch space
            rhs_launch_space = tuple(map(lambda x: launch_space[x], axes))
            rhs_tile_shape = tuple(map(lambda x: lhs_part.tile_shape[x], axes))
            rhs_offset = tuple(map(lambda x: lhs_part.tile_offset[x], axes))
            rhs_part = rhs_array.base.find_or_create_partition(
                rhs_launch_space, rhs_tile_shape, rhs_offset
            )

            lhs_arg = DeferredArrayView(
                lhs_array,
                part=lhs_part,
                tag=NumPyMappingTag.KEY_REGION_TAG,
            )
            rhs_arg = DeferredArrayView(
                rhs_array,
                part=rhs_part,
                proj_id=self.runtime.first_proj_id
                + NumPyProjCode.PROJ_2D_2D_YX,
                tag=NumPyMappingTag.NO_MEMOIZE_TAG,
            )
        else:
            lhs_arg = DeferredArrayView(lhs_array)
            rhs_arg = DeferredArrayView(rhs_array)

        (shardpt, shardfn, shardsp) = lhs_arg.sharding

        task = Map(self.runtime, NumPyOpCode.TRANSPOSE, tag=shardfn)
        if launch_space is not None:
            task.add_shape(lhs_arg.shape, lhs_arg.part.tile_shape, 0)
        else:
            task.add_shape(lhs_arg.shape)
        lhs_arg.add_to_legate_op(task, False)
        rhs_arg.add_to_legate_op(task, True)

        if shardsp is not None:
            task.set_sharding_space(shardsp)

        if launch_space is not None:
            task.execute(Rect(launch_space))
        else:
            if shardpt is not None:
                task.set_point(shardpt)
            task.execute_single()

        self.runtime.profile_callsite(stacklevel + 1, True, callsite)
        # See if we are doing shadow debugging
        if self.runtime.shadow_debug:
            self.shadow.transpose(
                rhs.shadow, axes, stacklevel=(stacklevel + 1)
            )
            self.runtime.check_shadow(self, "transpose")

    # Perform a bin count operation on the array
    def bincount(self, rhs, stacklevel, weights=None, callsite=None):
        weight_array = _maybe_apply(
            self.runtime.to_deferred_array,
            weights,
            stacklevel=stacklevel + 1,
        )
        src_array = self.runtime.to_deferred_array(
            rhs, stacklevel=(stacklevel + 1)
        )
        dst_array = self
        assert src_array.size > 1
        assert dst_array.ndim == 1
        if weight_array is not None:
            assert src_array.shape == weight_array.shape or (
                src_array.size == 1 and weight_array.size == 1
            )

        # We broadcast the reduction output
        dst_arg = DeferredArrayView(dst_array, proj_id=None)
        src_arg = DeferredArrayView(src_array)
        weight_arg = _maybe_apply(DeferredArrayView, weight_array)

        launch_space = src_arg.compute_launch_space()
        needs_reduction = launch_space is not None

        if needs_reduction and weight_arg is not None:
            weight_arg = weight_arg.align_partition(src_arg)

        (shardpt, shardfn, shardsp) = src_arg.sharding

        dst_array.fill(
            np.array(0, dst_arg.dtype), stacklevel + 1, callsite=callsite
        )

        task = Map(self.runtime, NumPyOpCode.BINCOUNT, tag=shardfn)
        task.add_scalar_arg(needs_reduction, bool)
        task.add_scalar_arg(weight_arg is not None, bool)

        if launch_space is not None:
            task.add_shape(src_arg.shape, src_arg.part.tile_shape, 0)
        else:
            task.add_shape(src_arg.shape)

        if needs_reduction:
            redop = self.runtime.get_unary_reduction_op_id(
                UnaryRedCode.SUM,
                dst_arg.dtype,
            )
            dst_arg.add_to_legate_op(task, False, redop=redop)
        else:
            dst_arg.add_to_legate_op(task, False, read_write=True)
        src_arg.add_to_legate_op(task, True)
        if weight_arg is not None:
            weight_arg.add_to_legate_op(task, True)

        if shardsp is not None:
            task.set_sharding_space(shardsp)

        if launch_space is not None:
            task.execute(Rect(launch_space))
        else:
            task.execute_single()

        self.runtime.profile_callsite(stacklevel + 1, True, callsite)
        # See if we are doing shadow debugging
        if self.runtime.shadow_debug:
            self.shadow.bincount(
                rhs.shadow,
                stacklevel=(stacklevel + 1),
                weights=None if weights is None else weights.shadow,
            )
            self.runtime.check_shadow(self, "bincount")

    def count_nonzero(self, stacklevel, axis):
        # Handle the case in which we have a future
        if self.size == 1:
            return ndarray.convert_to_legate_ndarray(
                int(
                    self.get_scalar_array(stacklevel=(stacklevel + 1)).item()
                    != 0
                ),
                stacklevel=(stacklevel + 1),
            )
        return ndarray.perform_unary_reduction(
            UnaryRedCode.COUNT_NONZERO,
            self,
            axis=axis,
            dtype=np.dtype(np.uint64),
            stacklevel=(stacklevel + 1),
            check_types=False,
        )

    def nonzero(self, stacklevel, callsite=None):
        raise NotImplementedError()

    def random(self, gen_code, args, stacklevel, callsite=None):
        lhs_arg = DeferredArrayView(self)
        launch_space, key_arg = lhs_arg.find_key_view()
        key_arg.update_tag(NumPyMappingTag.KEY_REGION_TAG)

        (shardpt, shardfn, shardsp) = key_arg.sharding

        task = Map(self.runtime, NumPyOpCode.RAND, tag=shardfn)

        task.add_scalar_arg(gen_code.value, np.int32)
        if launch_space is not None:
            task.add_shape(lhs_arg.shape, lhs_arg.part.tile_shape, 0)
        else:
            task.add_shape(lhs_arg.shape)
        lhs_arg.add_to_legate_op(task, False)
        epoch = self.runtime.get_next_random_epoch()
        task.add_scalar_arg(epoch, np.uint32)
        task.add_point(self.compute_strides(lhs_arg.shape), untyped=True)
        self.add_arguments(task, args)

        if launch_space is not None:
            task.execute(Rect(launch_space))
        else:
            task.execute_single()

        self.runtime.profile_callsite(stacklevel + 1, True, callsite)

    def random_uniform(self, stacklevel, callsite=None):
        assert self.dtype == np.float64
        # Special case for shadow debugging
        if self.runtime.shadow_debug:
            self.shadow.random_uniform(stacklevel=(stacklevel + 1))
            self.base.attach_numpy_array(self.shadow.array.copy())
            return
        self.random(RandGenCode.UNIFORM, [], stacklevel + 1, callsite)

    def random_normal(self, stacklevel, callsite=None):
        assert self.dtype == np.float64
        # Special case for shadow debugging since it's hard to get data back
        if self.runtime.shadow_debug:
            self.shadow.random_normal(stacklevel=(stacklevel + 1))
            self.base.attach_numpy_array(self.shadow.array.copy())
            return
        self.random(RandGenCode.NORMAL, [], stacklevel + 1, callsite)

    def random_integer(self, low, high, stacklevel, callsite=None):
        assert self.dtype.kind == "i"
        # Special case for shadow debugging since it's hard to get data back
        if self.runtime.shadow_debug:
            self.shadow.random_integer(low, high, stacklevel=(stacklevel + 1))
            self.base.attach_numpy_array(self.shadow.array.copy())
            return
        low = np.array(low, self.dtype)
        high = np.array(high, self.dtype)
        self.random(RandGenCode.INTEGER, [low, high], stacklevel + 1, callsite)

    # Perform the unary operation and put the result in the array
    def unary_op(
        self, op, op_dtype, src, where, args, stacklevel, callsite=None
    ):
        lhs_array = self
        rhs_array = self.runtime.to_deferred_array(
            src, stacklevel=(stacklevel + 1)
        )

        if op == UnaryOpCode.GETARG:
            dtype = get_arg_value_dtype(rhs_array.dtype)
        else:
            dtype = None
        rhs_arg = DeferredArrayView(rhs_array, dtype=dtype)
        if rhs_array is lhs_array:
            lhs_arg = rhs_arg
        else:
            lhs_arg = DeferredArrayView(lhs_array)

        launch_space, key_arg = lhs_arg.find_key_view(rhs_arg)
        key_arg.update_tag(NumPyMappingTag.KEY_REGION_TAG)
        rhs_arg = rhs_arg.broadcast(lhs_arg)

        if launch_space is not None:
            lhs_arg = lhs_arg.align_partition(key_arg)
            rhs_arg = rhs_arg.align_partition(key_arg)
            if lhs_arg is not key_arg:
                lhs_arg.copy_key_partition_from(key_arg)

        if rhs_arg.scalar:
            task_id = NumPyOpCode.SCALAR_UNARY_OP
        else:
            task_id = NumPyOpCode.UNARY_OP

        (shardpt, shardfn, shardsp) = key_arg.sharding

        task = Map(self.runtime, task_id, tag=shardfn)
        task.add_scalar_arg(op.value, np.int32)

        if not lhs_arg.scalar:
            if launch_space is not None:
                task.add_shape(lhs_arg.shape, lhs_arg.part.tile_shape, 0)
            else:
                task.add_shape(lhs_arg.shape)
            lhs_arg.add_to_legate_op(task, False)

        rhs_arg.add_to_legate_op(task, True)

        self.add_arguments(task, args)

        if shardsp is not None:
            task.set_sharding_space(shardsp)

        # See if we are doing index space launches or not
        if not rhs_arg.scalar and launch_space is not None:
            task.execute(Rect(launch_space))
        else:
            if shardpt is not None:
                task.set_point(shardpt)
            result = task.execute_single()

            if rhs_arg.scalar:
                lhs_arg._array._fill(result, stacklevel + 1, callsite)

        self.runtime.profile_callsite(stacklevel + 1, True, callsite)
        if self.runtime.shadow_debug:
            self.shadow.unary_op(
                op,
                op_dtype,
                src.shadow,
                where if not isinstance(where, NumPyThunk) else where.shadow,
                args,
                stacklevel=(stacklevel + 1),
            )
            self.runtime.check_shadow(self, op)

    # Perform a unary reduction operation from one set of dimensions down to
    # fewer
    def unary_reduction(
        self,
        op,
        src,
        where,
        axes,
        keepdims,
        args,
        initial,
        stacklevel,
        callsite=None,
    ):
        lhs_array = self
        rhs_array = self.runtime.to_deferred_array(
            src, stacklevel=(stacklevel + 1)
        )
        assert lhs_array.ndim <= rhs_array.ndim
        assert rhs_array.size > 1
        rhs = rhs_array.base

        # See if we are doing reduction to a point or another region
        if lhs_array.size == 1:
            assert axes is None or len(axes) == (
                rhs_array.ndim - lhs_array.ndim
            )
            rhs_arg = DeferredArrayView(rhs_array)
            launch_space, key_arg = rhs_arg.find_key_view()
            key_arg.update_tag(NumPyMappingTag.KEY_REGION_TAG)
            (shardpt, shardfn, shardsp) = key_arg.sharding

            task = Map(self.runtime, NumPyOpCode.SCALAR_UNARY_RED, tag=shardfn)

            task.add_scalar_arg(op, np.int32)
            if launch_space is not None:
                task.add_shape(rhs_arg.shape, rhs_arg.part.tile_shape, 0)
            else:
                task.add_shape(rhs_arg.shape)
            rhs_arg.add_to_legate_op(task, True)

            if shardpt is not None:
                task.set_point(shardpt)
            if shardsp is not None:
                task.set_sharding_space(shardsp)
            self.add_arguments(task, args)

            if launch_space is not None:
                result = task.execute(
                    Rect(launch_space),
                    redop=self.runtime.get_scalar_reduction_op_id(op),
                )
            else:
                result = task.execute_single()

            lhs_array.base = result

        else:
            argred = op in (UnaryRedCode.ARGMAX, UnaryRedCode.ARGMIN)

            if argred:
                argred_dtype = get_arg_dtype(rhs_array.dtype)
                lhs_field = self.runtime.allocate_field(
                    self.shape,
                    dtype=argred_dtype,
                )
                lhs_array = DeferredArray(
                    self.runtime,
                    lhs_field,
                    shape=self.shape,
                    dtype=argred_dtype,
                    scalar=self.scalar,
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
            # Iterate over all the dimension(s) being collapsed and build a
            # temporary field that we will reduce down to the final value
            if len(axes) > 1:
                raise NotImplementedError(
                    "Need support for reducing multiple dimensions"
                )
            axis = axes[0]
            # Compute the reduction transform
            transform, proj_id = self.runtime.get_reduction_transform(
                rhs_array.shape, axes
            )
            # Compute the launch space
            launch_space = rhs.compute_parallel_launch_space()
            # Then we launch the reduction task(s)
            if launch_space is not None:
                # Index space launch case
                rhs_part, shardfn, shardsp = rhs.find_or_create_key_partition()
                # Get the result partition to use
                if lhs_array.ndim == rhs_array.ndim:
                    offset = np.zeros(rhs_array.ndim, dtype=np.dtype(np.int64))
                    # Handle the keepdims case
                    # Put ones in all the keep dim axes
                    for ax in axes:
                        offset[ax] = 1
                else:
                    offset = np.zeros(lhs_array.ndim, dtype=np.dtype(np.int64))
                result_part = result.find_or_create_congruent_partition(
                    rhs_part, transform, offset
                )

                needs_reduction = launch_space[axis] != 1
                # Fake the dtype when this is an arg reduction.
                # Otherwise, we would need to double the number of type codes.
                lhs_arg = DeferredArrayView(
                    lhs_array,
                    part=result_part,
                    transform=transform,
                    proj_id=proj_id,
                    tag=NumPyMappingTag.NO_MEMOIZE_TAG,
                    dtype=rhs_array.dtype if argred else None,
                )
                rhs_arg = DeferredArrayView(
                    rhs_array,
                    part=rhs_part,
                    tag=NumPyMappingTag.KEY_REGION_TAG,
                )

                shardpt, shardfn, shardsp = rhs_arg.sharding

                task = Map(self.runtime, NumPyOpCode.UNARY_RED, tag=shardfn)
                task.add_scalar_arg(needs_reduction, bool)
                task.add_scalar_arg(axis, np.int32)
                task.add_scalar_arg(op, np.int32)
                task.add_shape(rhs_arg.shape, rhs_part.tile_shape, 0)

                if needs_reduction:
                    redop = self.runtime.get_unary_reduction_op_id(
                        op, rhs_array.dtype
                    )
                    lhs_arg.add_to_legate_op(task, False, redop=redop)
                    rhs_arg.add_to_legate_op(task, True)
                else:
                    lhs_arg.add_to_legate_op(task, False, read_write=True)
                    rhs_arg.add_to_legate_op(task, True)

                if shardpt is not None:
                    task.set_point(shardpt)
                if shardsp is not None:
                    task.set_sharding_space(shardsp)
                if args is not None:
                    self.add_arguments(task, args)
                if initial is not None:
                    task.add_future(initial_future)

                task.execute(Rect(launch_space))

            else:
                # Fake the dtype when this is an arg reduction.
                # Otherwise, we would need to double the number of type codes.
                lhs_arg = DeferredArrayView(
                    lhs_array,
                    transform=transform,
                    proj_id=proj_id,
                    tag=NumPyMappingTag.NO_MEMOIZE_TAG,
                    dtype=rhs_array.dtype if argred else None,
                )
                rhs_arg = DeferredArrayView(
                    rhs_array,
                    tag=NumPyMappingTag.KEY_REGION_TAG,
                )

                shardpt, shardfn, shardsp = rhs_arg.sharding

                task = Map(self.runtime, NumPyOpCode.UNARY_RED, tag=shardfn)
                task.add_scalar_arg(False, bool)  # needs_reduction
                task.add_scalar_arg(axis, np.int32)
                task.add_scalar_arg(op, np.int32)
                task.add_shape(rhs_arg.shape)

                lhs_arg.add_to_legate_op(task, False, read_write=True)
                rhs_arg.add_to_legate_op(task, True)

                if shardpt is not None:
                    task.set_point(shardpt)
                if shardsp is not None:
                    task.set_sharding_space(shardsp)
                if args is not None:
                    self.add_arguments(task, args)
                if initial is not None:
                    task.add_future(initial_future)

                task.execute_single()

            if argred:
                self.unary_op(
                    UnaryOpCode.GETARG,
                    self.dtype,
                    lhs_array,
                    True,
                    [],
                    stacklevel + 1,
                    callsite,
                )

        self.runtime.profile_callsite(stacklevel + 1, True, callsite)
        if self.runtime.shadow_debug:
            self.shadow.unary_reduction(
                op,
                redop,
                rhs.shadow,
                where.shadow,
                axes,
                keepdims,
                args,
                initial,
                stacklevel=(stacklevel + 1),
            )
            self.runtime.check_shadow(self, op)

    # Perform the binary operation and put the result in the lhs array
    def binary_op(
        self, op_code, src1, src2, where, args, stacklevel, callsite=None
    ):
        rhs1_array = self.runtime.to_deferred_array(
            src1, stacklevel=(stacklevel + 1)
        )
        rhs2_array = self.runtime.to_deferred_array(
            src2, stacklevel=(stacklevel + 1)
        )
        lhs_array = self

        rhs1_arg = DeferredArrayView(rhs1_array)
        rhs2_arg = DeferredArrayView(rhs2_array)

        if lhs_array.base is rhs1_array.base:
            lhs_arg = rhs1_arg
        elif lhs_array.base is rhs2_array.base:
            lhs_arg = rhs2_arg
        else:
            lhs_arg = DeferredArrayView(lhs_array)

        # Align and broadcast region arguments if necessary
        launch_space, key_arg = lhs_arg.find_key_view(rhs1_arg, rhs2_arg)
        key_arg.update_tag(NumPyMappingTag.KEY_REGION_TAG)
        rhs1_arg = rhs1_arg.broadcast(lhs_arg)
        rhs2_arg = rhs2_arg.broadcast(lhs_arg)

        if launch_space is not None:
            lhs_arg = lhs_arg.align_partition(key_arg)
            rhs1_arg = rhs1_arg.align_partition(key_arg)
            rhs2_arg = rhs2_arg.align_partition(key_arg)
            if lhs_arg is not key_arg:
                lhs_arg.copy_key_partition_from(key_arg)

        # Populate the Legate launcher
        all_scalar_rhs = rhs1_arg.scalar and rhs2_arg.scalar

        if all_scalar_rhs:
            task_id = NumPyOpCode.SCALAR_BINARY_OP
        else:
            task_id = NumPyOpCode.BINARY_OP

        (shardpt, shardfn, shardsp) = key_arg.sharding

        op = Map(self.runtime, task_id, tag=shardfn)
        op.add_scalar_arg(op_code.value, np.int32)

        if not all_scalar_rhs:
            if launch_space is not None:
                op.add_shape(lhs_arg.shape, lhs_arg.part.tile_shape, 0)
            else:
                op.add_shape(lhs_arg.shape)
            lhs_arg.add_to_legate_op(op, False)

        rhs1_arg.add_to_legate_op(op, True)
        rhs2_arg.add_to_legate_op(op, True)
        self.add_arguments(op, args)

        if shardsp is not None:
            op.set_sharding_space(shardsp)

        # See if we are doing index space launches or not
        if not all_scalar_rhs and launch_space is not None:
            op.execute(Rect(launch_space))
        else:
            if shardpt is not None:
                op.set_point(shardpt)
            result = op.execute_single()

            # If the result is a scalar, we need to fill the lhs with it.
            # Note that the fill function makes an alias to the scalar,
            # when the lhs is a singleton array.
            if all_scalar_rhs:
                lhs_arg._array._fill(result, stacklevel + 1, callsite)

        # TODO: We should be able to do this automatically via decorators
        self.runtime.profile_callsite(stacklevel + 1, True, callsite)
        if self.runtime.shadow_debug:
            self.shadow.binary_op(
                op_code,
                src1.shadow,
                src2.shadow,
                where if not isinstance(where, NumPyThunk) else where.shadow,
                args,
                stacklevel=(stacklevel + 1),
            )
            self.runtime.check_shadow(self, op_code)

    def binary_reduction(
        self, op, src1, src2, broadcast, args, stacklevel, callsite=None
    ):

        rhs1_array = self.runtime.to_deferred_array(
            src1, stacklevel=(stacklevel + 1)
        )
        rhs2_array = self.runtime.to_deferred_array(
            src2, stacklevel=(stacklevel + 1)
        )
        lhs_array = self
        assert lhs_array.size == 1

        rhs1_arg = DeferredArrayView(rhs1_array)
        rhs2_arg = DeferredArrayView(rhs2_array)

        # Align and broadcast region arguments if necessary
        launch_space, key_arg = rhs1_arg.find_key_view(
            rhs2_arg, shape=broadcast
        )
        key_arg.update_tag(NumPyMappingTag.KEY_REGION_TAG)
        if broadcast is not None:
            rhs1_arg = rhs1_arg.broadcast(broadcast)
            rhs2_arg = rhs2_arg.broadcast(broadcast)

        if launch_space is not None:
            rhs1_arg = rhs1_arg.align_partition(key_arg)
            rhs2_arg = rhs2_arg.align_partition(key_arg)

        # Populate the Legate launcher
        all_scalar_rhs = rhs1_arg.scalar and rhs2_arg.scalar

        if all_scalar_rhs:
            task_id = NumPyOpCode.SCALAR_BINARY_OP
        else:
            task_id = NumPyOpCode.BINARY_RED

        (shardpt, shardfn, shardsp) = key_arg.sharding

        task = Map(self.runtime, task_id, tag=shardfn)
        task.add_scalar_arg(op.value, np.int32)

        if not all_scalar_rhs:
            if launch_space is not None:
                task.add_shape(key_arg.shape, key_arg.part.tile_shape, 0)
            else:
                task.add_shape(key_arg.shape)

        rhs1_arg.add_to_legate_op(task, True)
        rhs2_arg.add_to_legate_op(task, True)
        self.add_arguments(task, args)

        if shardsp is not None:
            task.set_sharding_space(shardsp)

        # See if we are doing index space launches or not
        if launch_space is not None:
            if op == BinaryOpCode.NOT_EQUAL:
                redop = UnaryRedCode.SUM
            else:
                redop = UnaryRedCode.PROD
            redop_id = self.runtime.get_scalar_reduction_op_id(redop)
            result = task.execute(Rect(launch_space), redop=redop_id)
        else:
            if shardpt is not None:
                task.set_point(shardpt)
            result = task.execute_single()

        lhs_array.base = result

        self.runtime.profile_callsite(stacklevel + 1, True, callsite)
        if self.runtime.shadow_debug:
            self.shadow.binary_reduction(
                op,
                src1.shadow,
                src2.shadow,
                broadcast,
                args,
                stacklevel=(stacklevel + 1),
            )
            self.runtime.check_shadow(self, op)

    def where(self, src1, src2, src3, stacklevel, callsite=None):
        rhs1_array = self.runtime.to_deferred_array(
            src1, stacklevel=(stacklevel + 1)
        )
        rhs2_array = self.runtime.to_deferred_array(
            src2, stacklevel=(stacklevel + 1)
        )
        rhs3_array = self.runtime.to_deferred_array(
            src3, stacklevel=(stacklevel + 1)
        )
        lhs_array = self

        rhs1_arg = DeferredArrayView(rhs1_array)
        rhs2_arg = DeferredArrayView(rhs2_array)
        rhs3_arg = DeferredArrayView(rhs3_array)
        lhs_arg = DeferredArrayView(lhs_array)

        # Align and broadcast region arguments if necessary
        launch_space, key_arg = lhs_arg.find_key_view(
            rhs1_arg, rhs2_arg, rhs3_arg
        )
        key_arg.update_tag(NumPyMappingTag.KEY_REGION_TAG)
        rhs1_arg = rhs1_arg.broadcast(lhs_arg)
        rhs2_arg = rhs2_arg.broadcast(lhs_arg)
        rhs3_arg = rhs3_arg.broadcast(lhs_arg)

        if launch_space is not None:
            lhs_arg = lhs_arg.align_partition(key_arg)
            rhs1_arg = rhs1_arg.align_partition(key_arg)
            rhs2_arg = rhs2_arg.align_partition(key_arg)
            rhs3_arg = rhs3_arg.align_partition(key_arg)
            if lhs_arg is not key_arg:
                lhs_arg.copy_key_partition_from(key_arg)

        # Populate the Legate launcher
        all_scalar_rhs = (
            rhs1_arg.scalar and rhs2_arg.scalar and rhs3_arg.scalar
        )

        if all_scalar_rhs:
            task_id = NumPyOpCode.SCALAR_WHERE
        else:
            task_id = NumPyOpCode.WHERE

        (shardpt, shardfn, shardsp) = key_arg.sharding

        task = Map(self.runtime, task_id, tag=shardfn)
        if not all_scalar_rhs:
            if launch_space is not None:
                task.add_shape(lhs_arg.shape, lhs_arg.part.tile_shape, 0)
            else:
                task.add_shape(lhs_arg.shape)
            lhs_arg.add_to_legate_op(task, False)
        rhs1_arg.add_to_legate_op(task, True)
        rhs2_arg.add_to_legate_op(task, True)
        rhs3_arg.add_to_legate_op(task, True)

        if shardsp is not None:
            task.set_sharding_space(shardsp)

        # See if we are doing index space launches or not
        if not all_scalar_rhs and launch_space is not None:
            task.execute(Rect(launch_space))
        else:
            if shardpt is not None:
                op.set_point(shardpt)
            result = task.execute_single()
            if lhs_array.scalar:
                lhs_array.base = result

        self.runtime.profile_callsite(stacklevel + 1, True, callsite)
        if self.runtime.shadow_debug:
            self.shadow.where(
                src1.shadow,
                src2.shadow,
                src3.shadow,
                stacklevel=(stacklevel + 1),
            )
            self.runtime.check_shadow(self, op)

    # A helper method for attaching arguments
    def add_arguments(self, op, args):
        args = [] if args is None else args
        op.add_scalar_arg(len(args), np.int32)
        for numpy_array in args:
            assert numpy_array.size == 1
            scalar = self.runtime.create_scalar(
                numpy_array.data, numpy_array.dtype
            )
            op.add_future(scalar)

    @staticmethod
    def compute_strides(shape):
        stride = 1
        result = ()
        for dim in reversed(shape):
            result = (stride,) + result
            stride *= dim
        return result
