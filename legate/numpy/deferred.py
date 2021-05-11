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
from .launcher import Map, Projection
from .thunk import NumPyThunk
from .utils import calculate_volume

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

try:
    long  # Python 2
except NameError:
    long = int  # Python 3


def _combine_transforms(f, g):
    if g is None:
        return f
    elif f is None:
        return g
    else:
        return g.compose(f)


class DeferredArrayView(object):
    def __init__(self, array, transform=None, part=None, proj_id=0, tag=0):
        self._array = array
        self._transform = transform
        self._part = part
        self._proj_id = proj_id
        self._tag = tag

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
    def part(self):
        assert not self.scalar
        if self._part is None:
            (part, _, _) = self._array.base.find_or_create_key_partition()
            self._part = part
        return self._part

    @property
    def transform(self):
        return self._array.shape

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

    def add_to_legate_op(self, op, read_only):
        if self.scalar:
            if not read_only:
                raise ValueError("Singleton arrays must be read only")
            op.add_future(self._array.base)
        else:
            add = op.add_input if read_only else op.add_output
            add(
                self._array,
                _combine_transforms(
                    self._array.base.transform, self._transform
                ),
                Projection(self.part, self._proj_id),
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

    def find_key_view(self, *views):
        if self.scalar:
            return None, self
        elif self.has_launch_space:
            return self.compute_launch_space(), self

        for view in views:
            if view is self or view.shape != self.shape:
                continue
            launch_space = view.compute_launch_space()
            if launch_space is not None:
                return launch_space, view

        # If we're here, we haven't found any partitioned region field
        # to use as the key
        launch_space = self.compute_launch_space()
        return launch_space, self

    def broadcast(self, to_align):
        if self.scalar:
            return self, None
        else:
            (
                transform,
                offset,
                proj_id,
                mapping_tag,
            ) = self._array.runtime.compute_broadcast_transform(
                to_align._array.shape, self._array.shape
            )
            new_view = DeferredArrayView(
                self._array,
                transform,
                self._part,
                proj_id,
                mapping_tag,
            )
            return new_view, offset

    def align_partition(self, key, offset=None):
        if self is key or self.scalar:
            return self
        else:
            new_part = self._array.base.find_or_create_congruent_partition(
                key.part, self._transform, offset
            )
            new_view = DeferredArrayView(
                self._array,
                self._transform,
                new_part,
                self._proj_id,
                self._tag,
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

    def imag(self, stacklevel):
        raise NotImplementedError("Implement in derived classes")

    def real(self, stacklevel):
        raise NotImplementedError("Implement in derived classes")

    # Copy source array to the destination array
    def copy(self, rhs, deep, stacklevel, callsite=None):
        dst_array = self
        src_array = self.runtime.to_deferred_array(
            rhs, stacklevel=(stacklevel + 1)
        )
        assert src_array.ndim == dst_array.ndim
        assert src_array.shape == dst_array.shape or (
            src_array.size == 1 and dst_array.size == 1
        )
        if isinstance(src_array.base, Future):
            # A slightly strange case, but an easy one since futures are
            # immutable
            dst_array.base = src_array.base
            return
        src = src_array.base
        dst = dst_array.base
        # Sometimes we get asked to do this in case of in-place updates
        # in which case both src and dst will have the same field so
        # there is nothing for us to do
        if src.field is dst.field:
            return
        # Check to see if we already have a target launch space for
        # the destination meaning it has been partitioned already
        if dst.has_parallel_launch_space():
            launch_space = dst.compute_parallel_launch_space()
            src_parallel = False
        else:
            launch_space = src.compute_parallel_launch_space()
            src_parallel = True
        # Check to see if they are the same or there are no transforms
        if (
            src.region is dst.region
            or src.region.index_space is dst.region.index_space
            or (src.transform is None and dst.transform is None)
        ):
            # In this case we can just do a normal Legion copy
            if launch_space is not None:
                if src_parallel:
                    (
                        src_part,
                        shardfn,
                        shardsp,
                    ) = src.find_or_create_key_partition()
                    dst_part = dst.find_or_create_congruent_partition(src_part)
                    dst.set_key_partition(dst_part, shardfn, shardsp)
                else:
                    (
                        dst_part,
                        shardfn,
                        shardsp,
                    ) = dst.find_or_create_key_partition()
                    src_part = src.find_or_create_congruent_partition(dst_part)
                copy = IndexCopy(
                    Rect(launch_space),
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
                if shardsp is not None:
                    copy.set_sharding_space(shardsp)
                copy.add_src_requirement(src_part, src.field.field_id, 0)
                copy.add_dst_requirement(
                    dst_part,
                    dst.field.field_id,
                    0,
                    tag=NumPyMappingTag.KEY_REGION_TAG,
                )
                self.runtime.dispatch(copy)
            else:
                if src_parallel:
                    shardpt, shardfn, shardsp = src.find_point_sharding()
                else:
                    shardpt, shardfn, shardsp = dst.find_point_sharding()
                copy = Copy(mapper=self.runtime.mapper_id, tag=shardfn)
                if shardpt is not None:
                    copy.set_point(shardpt)
                if shardsp is not None:
                    copy.set_sharding_space(shardsp)
                copy.add_src_requirement(src.region, src.field.field_id)
                copy.add_dst_requirement(dst.region, dst.field.field_id)
                self.runtime.dispatch(copy)
        else:
            # We have to do an explicit copy task to handle the general
            # transform/gather/scatter case
            argbuf = BufferBuilder()
            if launch_space is not None:
                if src_parallel:
                    (
                        src_part,
                        shardfn,
                        shardsp,
                    ) = src.find_or_create_key_partition()
                    dst_part = dst.find_or_create_congruent_partition(src_part)
                    dst.set_key_partition(dst_part, shardfn, shardsp)
                else:
                    (
                        dst_part,
                        shardfn,
                        shardsp,
                    ) = dst.find_or_create_key_partition()
                    src_part = src.find_or_create_congruent_partition(dst_part)
                self.pack_shape(
                    argbuf, src_array.shape, src_part.tile_shape, 0
                )
            else:
                self.pack_shape(argbuf, src_array.shape)
            argbuf.pack_accessor(dst.field.field_id, dst.transform)
            argbuf.pack_accessor(src.field.field_id, src.transform)
            if launch_space is not None:
                task = IndexTask(
                    self.runtime.get_unary_task_id(
                        NumPyOpCode.COPY,
                        argument_type=src_array.dtype,
                        result_type=dst_array.dtype,
                    ),
                    Rect(launch_space),
                    self.runtime.empty_argmap,
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
                if shardsp is not None:
                    task.set_sharding_space(shardsp)
                assert dst_part.tile_shape == src_part.tile_shape
                task.add_write_requirement(
                    dst_part,
                    dst.field.field_id,
                    0,
                    tag=NumPyMappingTag.KEY_REGION_TAG,
                )
                task.add_read_requirement(src_part, src.field.field_id, 0)
                self.runtime.dispatch(task)
            else:
                if src_parallel:
                    shardpt, shardfn, shardsp = src.find_point_sharding()
                else:
                    shardpt, shardfn, shardsp = dst.find_point_sharding()
                task = Task(
                    self.runtime.get_unary_task_id(
                        NumPyOpCode.COPY,
                        argument_type=src_array.dtype,
                        result_type=dst_array.dtype,
                    ),
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
                if shardpt is not None:
                    task.set_point(shardpt)
                if shardsp is not None:
                    task.set_sharding_space(shardsp)
                task.add_write_requirement(dst.region, dst.field.field_id)
                task.add_read_requirement(src.region, src.field.field_id)
                self.runtime.dispatch(task)
        self.runtime.profile_callsite(stacklevel + 1, True, callsite)
        if self.runtime.shadow_debug:
            self.shadow.copy(rhs.shadow, stacklevel=(stacklevel + 1))
            self.runtime.check_shadow(self, "copy")

    def get_scalar_array(self, stacklevel):
        assert self.size == 1
        # Look at the type of the data and figure out how to read this data
        # First four bytes are for the type code, so we need to skip those
        buf = self.base.get_buffer(self.dtype.itemsize + 4)[4:]
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
                shardpt, sharfn, shardsp = index.find_point_sharding()
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
                src = self.base
                # Construct our buffer for task arguments
                argbuf = BufferBuilder()
                argbuf.pack_dimension(self.ndim)
                argbuf.pack_accessor(src.field.field_id, src.transform)
                argbuf.pack_point(use_key)
                # Now we can make the Task and add the future and region
                # arguments
                task = Task(
                    self.runtime.get_unary_task_id(
                        NumPyOpCode.READ,
                        argument_type=self.dtype,
                        result_type=self.dtype,
                    ),
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                )
                task.add_read_requirement(
                    src.region,
                    src.field.field_id,
                    tag=NumPyMappingTag.NO_MEMOIZE_TAG,
                )
                future = self.runtime.dispatch(task)
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
                point, shardfn, shardsp = index.find_pont_sharding()
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
                dst = self.base
                # Create the arguments to the task
                argbuf = BufferBuilder()
                argbuf.pack_dimension(self.ndim)
                argbuf.pack_accessor(dst.field.field_id, dst.transform)
                argbuf.pack_point(use_key)
                # No need for dependence since this is immediate
                argbuf.pack_value(
                    value_array.get_scalar_array(stacklevel + 1),
                    value_array.dtype.type,
                )
                # Now we can make the Task and add the region requirements
                task = Task(
                    self.runtime.get_nullary_task_id(
                        NumPyOpCode.WRITE, result_type=self.dtype
                    ),
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                )
                task.add_read_write_requirement(
                    dst.region,
                    dst.field.field_id,
                    tag=NumPyMappingTag.NO_MEMOIZE_TAG,
                )
                self.runtime.dispatch(task)
            else:
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
        src_array = self.runtime.to_deferred_array(
            rhs, stacklevel=(stacklevel + 1)
        )
        dst_array = self
        dst_dtype = dst_array.dtype
        assert src_array.dtype != dst_dtype
        assert src_array.shape == dst_array.shape or (
            src_array.size == 1 and dst_array.size == 1
        )
        src = src_array.base
        dst = dst_array.base
        if isinstance(dst, Future):
            # Simple case with futures
            # No warnings in this case since it's easy to convert futures
            assert isinstance(src, Future)
            task = Task(
                self.runtime.get_unary_task_id(
                    NumPyOpCode.CONVERT,
                    argument_type=src_array.dtype,
                    result_type=dst_dtype,
                    variant_code=NumPyVariantCode.SCALAR,
                ),
                mapper=self.runtime.mapper_id,
            )
            task.add_future(src)
            self.base = self.runtime.dispatch(task)
        else:
            if warn:
                warnings.warn(
                    "Legate performing implicit type conversion from "
                    + str(src_array.dtype)
                    + " to "
                    + str(dst_dtype),
                    category=UserWarning,
                    stacklevel=(stacklevel + 1),
                )
            # Check to see if we already have a target launch space for
            # the destination meaning it has been partitioned already
            if dst.has_parallel_launch_space():
                launch_space = dst.compute_parallel_launch_space()
                src_parallel = False
            else:
                launch_space = src.compute_parallel_launch_space()
                src_parallel = True
            argbuf = BufferBuilder()
            if launch_space is not None:
                if src_parallel:
                    (
                        src_part,
                        shardfn,
                        shardsp,
                    ) = src.find_or_create_key_partition()
                    dst_part = dst.find_or_create_congruent_partition(src_part)
                    dst.set_key_partition(dst_part, shardfn, shardsp)
                else:
                    (
                        dst_part,
                        shardfn,
                        shardsp,
                    ) = dst.find_or_create_key_partition()
                    src_part = src.find_or_create_congruent_partition(dst_part)
                self.pack_shape(
                    argbuf, src_array.shape, src_part.tile_shape, 0
                )
            else:
                self.pack_shape(argbuf, src_array.shape)
            argbuf.pack_accessor(dst.field.field_id, dst.transform)
            argbuf.pack_accessor(src.field.field_id, src.transform)
            if launch_space is not None:
                task = IndexTask(
                    self.runtime.get_unary_task_id(
                        NumPyOpCode.CONVERT,
                        result_type=dst_dtype,
                        argument_type=src_array.dtype,
                    ),
                    Rect(launch_space),
                    self.runtime.empty_argmap,
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
                if shardsp is not None:
                    task.set_sharding_space(shardsp)
                task.add_write_requirement(
                    dst_part,
                    dst.field.field_id,
                    0,
                    tag=NumPyMappingTag.KEY_REGION_TAG,
                )
                # src_part obtained above
                task.add_read_requirement(src_part, src.field.field_id, 0)
                self.runtime.dispatch(task)
            else:
                if src_parallel:
                    shardpt, shardfn, shardsp = src.find_point_sharding()
                else:
                    shardpt, shardfn, shardsp = dst.find_point_sharding()
                task = Task(
                    self.runtime.get_unary_task_id(
                        NumPyOpCode.CONVERT,
                        result_type=dst_dtype,
                        argument_type=src_array.dtype,
                    ),
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
                if shardpt is not None:
                    task.set_point(shardpt)
                if shardsp is not None:
                    task.set_sharding_space(shardsp)
                task.add_write_requirement(dst.region, dst.field.field_id)
                task.add_read_requirement(src.region, src.field.field_id)
                self.runtime.dispatch(task)
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
            launch_space = dst.compute_parallel_launch_space()
            if launch_space is not None:
                (
                    dst_part,
                    shardfn,
                    shardsp,
                ) = dst.find_or_create_key_partition()
                lhs = DeferredArrayView(self, part=dst_part)
            else:
                shardpt, shardfn, shardsp = dst.find_point_sharding()
                lhs = DeferredArrayView(self)

            op = Map(self.runtime, NumPyOpCode.FILL, tag=shardfn)
            if launch_space is not None:
                op.add_shape(dst.shape, dst_part.tile_shape, 0)
            else:
                op.add_shape(dst.shape)

            lhs.add_to_legate_op(op, False)
            op.add_future(value)
            if shardsp is not None:
                fill.set_sharding_space(shardsp)
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
        rhs1 = rhs1_array.base
        rhs2 = rhs2_array.base
        if rhs1_array.ndim == 1 and rhs2_array.ndim == 1:
            # Vector dot product case
            assert lhs_array.size == 1
            assert rhs1_array.shape == rhs2_array.shape or (
                rhs1_array.size == 1 and rhs2_array.size == 1
            )
            launch_space = rhs1.compute_parallel_launch_space()
            argbuf = BufferBuilder()
            if launch_space is not None:
                (
                    rhs1_part,
                    shardfn,
                    shardsp,
                ) = rhs1.find_or_create_key_partition()
                self.pack_shape(
                    argbuf, rhs1_array.shape, rhs1_part.tile_shape, 0
                )
            else:
                self.pack_shape(argbuf, rhs1_array.shape)
            argbuf.pack_accessor(rhs1.field.field_id, rhs1.transform)
            argbuf.pack_accessor(rhs2.field.field_id, rhs2.transform)
            if launch_space is not None:
                # Index space launch case
                task = IndexTask(
                    self.runtime.get_binary_task_id(
                        NumPyOpCode.DOT,
                        result_type=lhs_array.dtype,
                        first_argument_type=lhs_array.dtype,
                        second_argument_type=lhs_array.dtype,
                        variant_code=NumPyVariantCode.REDUCTION,
                    ),
                    Rect(launch_space),
                    self.runtime.empty_argmap,
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
                if shardsp is not None:
                    task.set_sharding_space(shardsp)
                # rhs1_part is computed above
                task.add_read_requirement(
                    rhs1_part,
                    rhs1.field.field_id,
                    0,
                    tag=NumPyMappingTag.KEY_REGION_TAG,
                )
                rhs2_part = rhs2.find_or_create_congruent_partition(rhs1_part)
                task.add_read_requirement(rhs2_part, rhs2.field.field_id, 0)
                redop = self.runtime.get_reduction_op_id(
                    NumPyOpCode.DOT, lhs_array.dtype
                )
                result = self.runtime.dispatch(task, redop=redop)
            else:
                # Single task launch case
                shardpt, shardfn, shardsp = rhs1.find_point_sharding()
                task = Task(
                    self.runtime.get_binary_task_id(
                        NumPyOpCode.DOT,
                        result_type=lhs_array.dtype,
                        first_argument_type=lhs_array.dtype,
                        second_argument_type=lhs_array.dtype,
                        variant_code=NumPyVariantCode.REDUCTION,
                    ),
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
                if shardpt is not None:
                    task.set_point(shardpt)
                if shardsp is not None:
                    task.set_sharding_space(shardsp)
                task.add_read_requirement(rhs1.region, rhs1.field.field_id)
                task.add_read_requirement(rhs2.region, rhs2.field.field_id)
                result = self.runtime.dispatch(task)
            lhs_array.base = result
        elif rhs1_array.ndim == 1 or rhs2_array.ndim == 1:
            # Matrix-vector or vector-matrix multiply
            assert lhs_array.ndim == 1
            assert rhs1_array.ndim == 2 or rhs2_array.ndim == 2
            result = lhs_array.base
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
                result_part = result.find_or_create_partition(
                    (launch_space[0] if left_matrix else launch_space[1],)
                )
                result_proj = self.runtime.first_proj_id + (
                    NumPyProjCode.PROJ_2D_1D_X
                    if left_matrix
                    else NumPyProjCode.PROJ_2D_1D_Y
                )
                lhs_tag = NumPyMappingTag.NO_MEMOIZE_TAG
                if rhs1_array.ndim == 1:
                    # Row input, divide rows of matrix
                    rhs1_part = rhs1.find_or_create_partition(
                        (launch_space[0],)
                    )
                    rhs1_proj = (
                        self.runtime.first_proj_id + NumPyProjCode.PROJ_2D_1D_X
                    )
                    rhs1_tag = NumPyMappingTag.NO_MEMOIZE_TAG
                else:
                    # Matrix input
                    assert rhs1_array.shape[0] > 1
                    rhs1_part = rhs1.find_or_create_partition(launch_space)
                    rhs1_proj = 0  # Identity projection matrix
                    rhs1_tag = NumPyMappingTag.KEY_REGION_TAG
                if rhs2_array.ndim == 1:
                    # Column input, divide columns of matrix
                    rhs2_part = rhs2.find_or_create_partition(
                        (launch_space[1],)
                    )
                    rhs2_proj = (
                        self.runtime.first_proj_id + NumPyProjCode.PROJ_2D_1D_Y
                    )
                    rhs2_tag = NumPyMappingTag.NO_MEMOIZE_TAG
                else:
                    assert rhs2_array.shape[1] > 1
                    # Matrix input
                    rhs2_part = rhs2.find_or_create_partition(launch_space)
                    rhs2_proj = 0  # Identity projection matrix
                    rhs2_tag = NumPyMappingTag.KEY_REGION_TAG
                convert = lhs_array.dtype.type == np.float16
                # Do an index space launch to compute the intermediates
                # If there are no reductions then we can put the output
                # directly in the output field, otherwise we need to make
                # a partial_field to store the intermediaries for reduciton
                if (left_matrix and launch_space[1] == 1) or (
                    not left_matrix and launch_space[0] == 1
                ):
                    # No reduction so write directly to outputs
                    argbuf = BufferBuilder()
                    if convert:
                        argbuf.pack_bool(False)  # no partial reduction
                    argbuf.pack_dimension(-1)  # No extra dimensions
                    self.pack_shape(
                        argbuf,
                        lhs_array.shape,
                        result_part.tile_shape,
                        result_proj,
                    )
                    argbuf.pack_accessor(
                        result.field.field_id, result.transform
                    )
                    self.pack_shape(
                        argbuf,
                        rhs1_array.shape,
                        rhs1_part.tile_shape,
                        rhs1_proj,
                    )
                    argbuf.pack_accessor(rhs1.field.field_id, rhs1.transform)
                    self.pack_shape(
                        argbuf,
                        rhs2_array.shape,
                        rhs2_part.tile_shape,
                        rhs2_proj,
                    )
                    argbuf.pack_accessor(rhs2.field.field_id, rhs2.transform)
                    task = IndexTask(
                        self.runtime.get_binary_task_id(
                            NumPyOpCode.DOT,
                            result_type=lhs_array.dtype,
                            first_argument_type=rhs1_array.dtype,
                            second_argument_type=rhs2_array.dtype,
                        ),
                        Rect(launch_space),
                        self.runtime.empty_argmap,
                        argbuf.get_string(),
                        argbuf.get_size(),
                        mapper=self.runtime.mapper_id,
                    )
                    task.add_write_requirement(
                        result_part, result.field.field_id, result_proj
                    )
                    task.add_read_requirement(
                        rhs1_part, rhs1.field.field_id, rhs1_proj, tag=rhs1_tag
                    )
                    task.add_read_requirement(
                        rhs2_part, rhs2.field.field_id, rhs2_proj, tag=rhs2_tag
                    )
                    self.runtime.dispatch(task)
                else:
                    # This is the reduction case so we're going to need a
                    # temporary field
                    # Figure out how big the reduction field is initially
                    reduction_shape = (
                        lhs_array.shape[0],
                        (launch_space[1] if left_matrix else launch_space[0]),
                    )
                    reduction_field = self.runtime.allocate_field(
                        reduction_shape,
                        result.field.dtype
                        if not convert
                        else np.dtype(np.float32),
                    )
                    reduction_part = reduction_field.find_or_create_partition(
                        (
                            launch_space[0]
                            if left_matrix
                            else launch_space[1],
                            reduction_shape[1],
                        )
                    )
                    argbuf = BufferBuilder()
                    if convert:
                        argbuf.pack_bool(True)  # partial reduction
                    argbuf.pack_dimension(
                        1 if left_matrix else 0
                    )  # The dimension with the reduction field
                    self.pack_shape(
                        argbuf,
                        lhs_array.shape,
                        result_part.tile_shape,
                        result_proj,
                    )
                    argbuf.pack_accessor(
                        reduction_field.field.field_id,
                        reduction_field.transform,
                    )
                    self.pack_shape(
                        argbuf,
                        rhs1_array.shape,
                        rhs1_part.tile_shape,
                        rhs1_proj,
                    )
                    argbuf.pack_accessor(rhs1.field.field_id, rhs1.transform)
                    self.pack_shape(
                        argbuf,
                        rhs2_array.shape,
                        rhs2_part.tile_shape,
                        rhs2_proj,
                    )
                    argbuf.pack_accessor(rhs2.field.field_id, rhs2.transform)
                    task = IndexTask(
                        self.runtime.get_binary_task_id(
                            NumPyOpCode.DOT,
                            result_type=lhs_array.dtype,
                            first_argument_type=lhs_array.dtype,
                            second_argument_type=lhs_array.dtype,
                        ),
                        Rect(launch_space),
                        self.runtime.empty_argmap,
                        argbuf.get_string(),
                        argbuf.get_size(),
                        mapper=self.runtime.mapper_id,
                    )
                    task.add_write_requirement(
                        reduction_part,
                        reduction_field.field.field_id,
                        0
                        if left_matrix
                        else self.runtime.first_proj_id
                        + NumPyProjCode.PROJ_2D_2D_YX,
                    )
                    task.add_read_requirement(
                        rhs1_part, rhs1.field.field_id, rhs1_proj, tag=rhs1_tag
                    )
                    task.add_read_requirement(
                        rhs2_part, rhs2.field.field_id, rhs2_proj, tag=rhs2_tag
                    )
                    self.runtime.dispatch(task)
                    # Now we need to launch the reduction tree(s)
                    RADIX = self.runtime.radix
                    assert reduction_shape[1] > 1
                    # Figure out what the natural partitioning of the output
                    # vector is so we can do reductions to it
                    launch_space = result.compute_parallel_launch_space()
                    if launch_space is None:
                        result_part = result.find_or_create_partition((1,))
                        launch_space = (1, reduction_shape[1])
                    else:
                        result_part = result.find_or_create_partition(
                            launch_space
                        )
                        launch_space = (launch_space[0], reduction_shape[1])
                    result_proj = (
                        self.runtime.first_proj_id + NumPyProjCode.PROJ_2D_1D_X
                    )
                    sharding_space = self.runtime.find_or_create_index_space(
                        launch_space
                    )
                    reduction_part = reduction_field.find_or_create_partition(
                        launch_space
                    )
                    radix_generation = 1
                    # Now we need to launch the tasks for the reductions across
                    # nodes
                    while reduction_shape[1] > 1:
                        argbuf = BufferBuilder()
                        argbuf.pack_dimension(RADIX)
                        new_reduction_shape = (
                            lhs_array.shape[0],
                            (reduction_shape[1] + RADIX - 1) // RADIX,
                        )
                        local_launch_space = (
                            launch_space[0],
                            new_reduction_shape[1],
                        )
                        # Perform index task launches to do the reductions
                        if new_reduction_shape[1] == 1 and not convert:
                            # If this is the last reduction we can put it right
                            # in the output array assuming we don't need a
                            # conversion
                            new_reduction_field = result
                            new_reduction_part = result_part
                            new_reduction_proj = result_proj
                            argbuf.pack_dimension(
                                -1
                            )  # No collapse out dimension
                        else:
                            new_reduction_field = self.runtime.allocate_field(
                                new_reduction_shape,
                                reduction_field.field.dtype,
                            )
                            new_reduction_part = new_reduction_field.find_or_create_partition(  # noqa E501
                                local_launch_space
                            )
                            new_reduction_proj = 0  # identity projection
                            argbuf.pack_dimension(
                                1
                            )  # Collapsing out domension 1
                        argbuf.pack_dimension(1)  # Collapsing in dimension 1
                        self.pack_shape(
                            argbuf,
                            lhs_array.shape,
                            result_part.tile_shape,
                            result_proj,
                        )
                        argbuf.pack_accessor(
                            new_reduction_field.field.field_id,
                            new_reduction_field.transform,
                        )
                        for idx in xrange(RADIX):
                            argbuf.pack_accessor(
                                reduction_field.field.field_id,
                                reduction_field.transform,
                            )
                        task = IndexTask(
                            self.runtime.get_binary_task_id(
                                NumPyOpCode.SUM_RADIX,
                                result_type=reduction_field.field.dtype,
                                first_argument_type=reduction_field.field.dtype,  # noqa E501
                                second_argument_type=reduction_field.field.dtype,  # noqa E501
                            ),
                            Rect(local_launch_space),
                            self.runtime.empty_argmap,
                            argbuf.get_string(),
                            argbuf.get_size(),
                            mapper=self.runtime.mapper_id,
                            tag=(
                                NumPyMappingTag.RADIX_GEN_TAG
                                & (radix_generation << RADIX_GEN_SHIFT)
                            )
                            | (
                                NumPyMappingTag.RADIX_DIM_TAG
                                & (1 << RADIX_DIM_SHIFT)
                            ),
                        )
                        task.set_sharding_space(sharding_space)
                        task.add_write_requirement(
                            new_reduction_part,
                            new_reduction_field.field.field_id,
                            new_reduction_proj,
                            tag=NumPyMappingTag.KEY_REGION_TAG,
                            flags=legion.LEGION_COMPLETE_PROJECTION_WRITE_FLAG,
                        )
                        for idx in xrange(RADIX):
                            task.add_read_requirement(
                                reduction_part,
                                reduction_field.field.field_id,
                                self.runtime.get_radix_projection_functor_id(
                                    2, 1, RADIX, idx
                                ),
                            )
                        self.runtime.dispatch(task)
                        reduction_shape = new_reduction_shape
                        reduction_field = new_reduction_field
                        reduction_part = new_reduction_part
                        radix_generation += 1
                    # If we need a conversion back from 16-bit do that now
                    if convert:
                        self.convert_float32_to_float16(
                            reduction_field, lhs_array, 1
                        )
            else:
                # Sequential GEMV case
                argbuf = BufferBuilder()
                # 16-bit floats don't need any special support in this case
                if lhs_array.dtype.type == np.float16:
                    argbuf.pack_bool(
                        False
                    )  # No need for a partial accumulator
                argbuf.pack_dimension(-1)  # No collapsing dimension here
                self.pack_shape(argbuf, lhs_array.shape)
                argbuf.pack_accessor(result.field.field_id, result.transform)
                self.pack_shape(argbuf, rhs1_array.shape)
                argbuf.pack_accessor(rhs1.field.field_id, rhs1.transform)
                self.pack_shape(argbuf, rhs2_array.shape)
                argbuf.pack_accessor(rhs2.field.field_id, rhs2.transform)
                # This is the easy in-place GEMV case
                task = Task(
                    self.runtime.get_binary_task_id(
                        NumPyOpCode.DOT,
                        result_type=lhs_array.dtype,
                        first_argument_type=rhs1_array.dtype,
                        second_argument_type=rhs2_array,
                    ),
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                )
                task.add_write_requirement(
                    result.region, result.field.field_id
                )
                task.add_read_requirement(rhs1.region, rhs1.field.field_id)
                task.add_read_requirement(rhs2.region, rhs2.field.field_id)
                self.runtime.dispatch(task)
        elif rhs1_array.ndim == 2 and rhs2_array.ndim == 2:
            # Matrix-matrix multiply
            M = lhs_array.shape[0]
            N = lhs_array.shape[1]
            K = rhs1_array.shape[1]
            assert M == rhs1_array.shape[0]  # Check M
            assert N == rhs2_array.shape[1]  # Check N
            assert K == rhs2_array.shape[0]  # Check K
            # Get our result field
            result = lhs_array.base
            # We need to figure out our strategy for matrix multiple, we can
            # figure this out using a deterministic process of decisions based
            # on the natural tiling of the largest of the three matrices
            lhs_size = M * N
            rhs1_size = M * K
            rhs2_size = K * N
            if lhs_size > rhs1_size and lhs_size > rhs2_size:
                # LHS is biggest
                lhs_launch = result.compute_parallel_launch_space()
                if lhs_launch is not None:
                    # This gives us the tile size for m and n
                    result_part = result.find_or_create_partition(lhs_launch)
                    # the choice for k and k_tile here is underconstrained, we
                    # want to be able to recognize inner and outer product like
                    # cases here, but also fall back to normal tiled execution
                    # when it's important, we'll try to make things squar-ish
                    # by making k_tile the min of the m_tile and n_tile sizes
                    m_tile = result_part.tile_shape[0]
                    n_tile = result_part.tile_shape[1]
                    k_tile = min(m_tile, n_tile)
                    rhs1_launch = (
                        (rhs1_array.shape[0] + m_tile - 1) // m_tile,
                        (rhs1_array.shape[1] + k_tile - 1) // k_tile,
                    )
                    rhs1_part = rhs1.find_or_create_partition(
                        rhs1_launch, tile_shape=(m_tile, k_tile)
                    )
                    rhs2_launch = (
                        (rhs2_array.shape[0] + k_tile - 1) // k_tile,
                        (rhs2_array.shape[1] + n_tile - 1) // n_tile,
                    )
                    rhs2_part = rhs2.find_or_create_partition(
                        rhs2_launch, tile_shape=(k_tile, n_tile)
                    )
                else:
                    # No need for any parallelism
                    rhs1_launch = None
            elif rhs1_size >= lhs_size and rhs1_size > rhs2_size:
                # RHS1 is the biggest
                rhs1_launch = rhs1.compute_parallel_launch_space()
                if rhs1_launch is not None:
                    # This gives us the tile size for k
                    rhs1_part = rhs1.find_or_create_partition(rhs1_launch)
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
                    rhs2_part = rhs2.find_or_create_partition(
                        rhs2_launch, tile_shape=(k_tile, n_tile)
                    )
                    result_part = result.find_or_create_partition(
                        lhs_launch, tile_shape=(m_tile, n_tile)
                    )
                else:
                    # No need for any parallelism
                    rhs1_launch = None
            else:
                assert rhs2_size >= lhs_size and rhs2_size >= rhs1_size
                # RHS2 is the biggest
                rhs2_launch = rhs2.compute_parallel_launch_space()
                if rhs2_launch is not None:
                    # This gives us the tile size for k
                    rhs2_part = rhs2.find_or_create_partition(rhs2_launch)
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
                    rhs1_part = rhs1.find_or_create_partition(
                        rhs1_launch, tile_shape=(m_tile, k_tile)
                    )
                    result_part = result.find_or_create_partition(
                        lhs_launch, tile_shape=(m_tile, n_tile)
                    )
                else:
                    # No need for any parallelism
                    rhs1_launch = None
            if rhs1_launch is not None:
                # Parallel launch case
                assert rhs2_launch is not None
                # Figure out if we're doing a matrix multiply that is going to
                # require a reduction or not. If we need a reduction it will be
                # more complicated because we'll have to build reduciton tree
                # tasks
                if rhs1_launch[1] == 1:
                    ###############################################
                    # Non-Reduction case
                    ###############################################
                    assert rhs2_launch[0] == 1
                    argbuf = BufferBuilder()
                    if lhs_array.dtype.type == np.float16:
                        argbuf.pack_bool(
                            False
                        )  # No need for 32-bit accumulate ever
                    argbuf.pack_dimension(-1)  # No collapse dimension
                    self.pack_shape(
                        argbuf, lhs_array.shape, result_part.tile_shape, 0
                    )
                    argbuf.pack_accessor(
                        result.field.field_id, result.transform
                    )
                    self.pack_shape(
                        argbuf,
                        rhs1_array.shape,
                        rhs1_part.tile_shape,
                        self.runtime.first_proj_id
                        + NumPyProjCode.PROJ_2D_2D_X,
                    )
                    argbuf.pack_accessor(rhs1.field.field_id, rhs1.transform)
                    self.pack_shape(
                        argbuf,
                        rhs2_array.shape,
                        rhs2_part.tile_shape,
                        self.runtime.first_proj_id
                        + NumPyProjCode.PROJ_2D_2D_Y,
                    )
                    argbuf.pack_accessor(rhs2.field.field_id, rhs2.transform)
                    task = IndexTask(
                        self.runtime.get_binary_task_id(
                            NumPyOpCode.DOT,
                            result_type=lhs_array.dtype,
                            first_argument_type=rhs1_array.dtype,
                            second_argument_type=rhs2_array.dtype,
                        ),
                        Rect(lhs_launch),
                        self.runtime.empty_argmap,
                        argbuf.get_string(),
                        argbuf.get_size(),
                        mapper=self.runtime.mapper_id,
                    )
                    task.add_write_requirement(
                        result_part,
                        result.field.field_id,
                        0,
                        tag=NumPyMappingTag.KEY_REGION_TAG
                        if lhs_size > rhs1_size and lhs_size > rhs2_size
                        else NumPyMappingTag.NO_MEMOIZE_TAG,
                    )
                    task.add_read_requirement(
                        rhs1_part,
                        rhs1.field.field_id,
                        self.runtime.first_proj_id
                        + NumPyProjCode.PROJ_2D_2D_X,
                        tag=NumPyMappingTag.KEY_REGION_TAG
                        if rhs1_size >= lhs_size and rhs1_size > rhs2_size
                        else NumPyMappingTag.NO_MEMOIZE_TAG,
                    )
                    task.add_read_requirement(
                        rhs2_part,
                        rhs2.field.field_id,
                        self.runtime.first_proj_id
                        + NumPyProjCode.PROJ_2D_2D_Y,
                        tag=NumPyMappingTag.KEY_REGION_TAG
                        if rhs2_size >= lhs_size and rhs2_size >= rhs1_size
                        else NumPyMappingTag.NO_MEMOIZE_TAG,
                    )
                    self.runtime.dispatch(task)
                else:
                    ###############################################
                    # Reduction case
                    ###############################################
                    convert = lhs_array.dtype.type == np.float16
                    assert lhs_launch[0] == rhs1_launch[0]
                    assert rhs1_launch[1] == rhs2_launch[0]
                    assert rhs2_launch[1] == lhs_launch[1]
                    launch_space = (
                        lhs_launch[0],
                        rhs1_launch[1],
                        lhs_launch[1],
                    )
                    reduction_shape = (
                        lhs_array.shape[0],
                        rhs1_launch[1],
                        lhs_array.shape[1],
                    )
                    reduction_field = self.runtime.allocate_field(
                        reduction_shape,
                        result.field.dtype
                        if not convert
                        else np.dtype(np.float32),
                    )
                    reduction_part = reduction_field.find_or_create_partition(
                        launch_space
                    )
                    result_proj = (
                        self.runtime.first_proj_id
                        + NumPyProjCode.PROJ_3D_2D_XZ
                    )
                    rhs1_proj = (
                        self.runtime.first_proj_id
                        + NumPyProjCode.PROJ_3D_2D_XY
                    )
                    rhs2_proj = (
                        self.runtime.first_proj_id
                        + NumPyProjCode.PROJ_3D_2D_YZ
                    )
                    if lhs_size > rhs1_size and lhs_size > rhs2_size:
                        lhs_tag = NumPyMappingTag.KEY_REGION_TAG
                        rhs1_tag = NumPyMappingTag.NO_MEMOIZE_TAG
                        rhs2_tag = NumPyMappingTag.NO_MEMOIZE_TAG
                    elif rhs1_size > lhs_size and rhs1_size > rhs2_size:
                        lhs_tag = NumPyMappingTag.NO_MEMOIZE_TAG
                        rhs1_tag = NumPyMappingTag.KEY_REGION_TAG
                        rhs2_tag = NumPyMappingTag.NO_MEMOIZE_TAG
                    else:
                        assert rhs2_size >= lhs_size and rhs2_size >= rhs1_size
                        lhs_tag = NumPyMappingTag.NO_MEMOIZE_TAG
                        rhs1_tag = NumPyMappingTag.NO_MEMOIZE_TAG
                        rhs2_tag = NumPyMappingTag.KEY_REGION_TAG
                    argbuf = BufferBuilder()
                    if convert:
                        argbuf.pack_bool(True)
                    argbuf.pack_dimension(1)  # The dimension being collapsed
                    self.pack_shape(
                        argbuf,
                        lhs_array.shape,
                        result_part.tile_shape,
                        result_proj,
                    )
                    argbuf.pack_accessor(
                        reduction_field.field.field_id,
                        reduction_field.transform,
                    )
                    self.pack_shape(
                        argbuf,
                        rhs1_array.shape,
                        rhs1_part.tile_shape,
                        rhs1_proj,
                    )
                    argbuf.pack_accessor(rhs1.field.field_id, rhs1.transform)
                    self.pack_shape(
                        argbuf,
                        rhs2_array.shape,
                        rhs2_part.tile_shape,
                        rhs2_proj,
                    )
                    argbuf.pack_accessor(rhs2.field.field_id, rhs2.transform)
                    task = IndexTask(
                        self.runtime.get_binary_task_id(
                            NumPyOpCode.DOT,
                            result_type=lhs_array.dtype,
                            first_argument_type=lhs_array.dtype,
                            second_argument_type=lhs_array.dtype,
                        ),
                        Rect(launch_space),
                        self.runtime.empty_argmap,
                        argbuf.get_string(),
                        argbuf.get_size(),
                        mapper=self.runtime.mapper_id,
                    )
                    task.add_write_requirement(
                        reduction_part,
                        reduction_field.field.field_id,
                        0,
                        tag=lhs_tag,
                    )
                    task.add_read_requirement(
                        rhs1_part, rhs1.field.field_id, rhs1_proj, tag=rhs1_tag
                    )
                    task.add_read_requirement(
                        rhs2_part, rhs2.field.field_id, rhs2_proj, tag=rhs2_tag
                    )
                    self.runtime.dispatch(task)
                    # Now that we've created the reduction field we need to
                    # actually reduce it down to the output array
                    RADIX = self.runtime.radix
                    assert reduction_shape[1] > 1
                    # this will be the sharding space for all the index
                    # launches
                    sharding_space = self.runtime.find_or_create_index_space(
                        launch_space
                    )
                    result_part = result.find_or_create_partition(lhs_launch)
                    radix_generation = 1
                    # Launch reduction tasks until we've reduced down to the
                    # output
                    while reduction_shape[1] > 1:
                        argbuf = BufferBuilder()
                        argbuf.pack_dimension(RADIX)
                        new_reduction_shape = (
                            lhs_array.shape[0],
                            (reduction_shape[1] + RADIX - 1) // RADIX,
                            lhs_array.shape[1],
                        )
                        local_launch_space = (
                            lhs_launch[0],
                            new_reduction_shape[1],
                            lhs_launch[1],
                        )
                        if new_reduction_shape[1] == 1 and not convert:
                            # If this is the last reduction we can put it right
                            # in the output array assuming we don't need a
                            # conversion
                            new_reduction_field = result
                            new_reduction_part = result_part
                            new_reduction_proj = result_proj
                            argbuf.pack_dimension(
                                -1
                            )  # No collapse out dimension
                        else:
                            new_reduction_field = self.runtime.allocate_field(
                                new_reduction_shape,
                                reduction_field.field.dtype,
                            )
                            new_reduction_part = new_reduction_field.find_or_create_partition(  # noqa E501
                                local_launch_space
                            )
                            new_reduction_proj = 0  # identity projection
                            argbuf.pack_dimension(
                                1
                            )  # Collapsing out dimension 1
                        argbuf.pack_dimension(1)  # Collapsing in dimension 1
                        self.pack_shape(
                            argbuf,
                            lhs_array.shape,
                            result_part.tile_shape,
                            result_proj,
                        )
                        argbuf.pack_accessor(
                            new_reduction_field.field.field_id,
                            new_reduction_field.transform,
                        )
                        for idx in xrange(RADIX):
                            argbuf.pack_accessor(
                                reduction_field.field.field_id,
                                reduction_field.transform,
                            )
                        task = IndexTask(
                            self.runtime.get_binary_task_id(
                                NumPyOpCode.SUM_RADIX,
                                result_type=reduction_field.field.dtype,
                                first_argument_type=reduction_field.field.dtype,  # noqa E501
                                second_argument_type=reduction_field.field.dtype,  # noqa E501
                            ),
                            Rect(local_launch_space),
                            self.runtime.empty_argmap,
                            argbuf.get_string(),
                            argbuf.get_size(),
                            mapper=self.runtime.mapper_id,
                            tag=(
                                NumPyMappingTag.RADIX_GEN_TAG
                                & (radix_generation << RADIX_GEN_SHIFT)
                            )
                            | (
                                NumPyMappingTag.RADIX_DIM_TAG
                                & (1 << RADIX_DIM_SHIFT)
                            ),
                        )
                        task.set_sharding_space(sharding_space)
                        task.add_write_requirement(
                            new_reduction_part,
                            new_reduction_field.field.field_id,
                            new_reduction_proj,
                            tag=NumPyMappingTag.KEY_REGION_TAG,
                            flags=legion.LEGION_COMPLETE_PROJECTION_WRITE_FLAG,
                        )
                        for idx in xrange(RADIX):
                            task.add_read_requirement(
                                reduction_part,
                                reduction_field.field.field_id,
                                self.runtime.get_radix_projection_functor_id(
                                    3, 1, RADIX, idx
                                ),
                            )
                        self.runtime.dispatch(task)
                        reduction_shape = new_reduction_shape
                        reduction_field = new_reduction_field
                        reduction_part = new_reduction_part
                        radix_generation += 1
                    # If we need a conversion back from 16-bit do that now
                    if convert:
                        self.convert_float32_to_float16(
                            reduction_field, lhs_array, 1
                        )
            else:
                # Single matrix multiply, pretty straight-forward
                argbuf = BufferBuilder()
                # 16-bit floats don't need any special support in this case
                if lhs_array.dtype.type == np.float16:
                    argbuf.pack_bool(
                        False
                    )  # No need for a partial accumulator
                argbuf.pack_dimension(-1)  # No extra dimensions
                self.pack_shape(argbuf, lhs_array.shape)
                argbuf.pack_accessor(result.field.field_id, result.transform)
                self.pack_shape(argbuf, rhs1_array.shape)
                argbuf.pack_accessor(rhs1.field.field_id, rhs1.transform)
                self.pack_shape(argbuf, rhs2_array.shape)
                argbuf.pack_accessor(rhs2.field.field_id, rhs2.transform)
                # This is the easy in-place GEMM case
                task = Task(
                    self.runtime.get_binary_task_id(
                        NumPyOpCode.DOT,
                        result_type=lhs_array.dtype,
                        first_argument_type=rhs1_array.dtype,
                        second_argument_type=rhs2_array.dtype,
                    ),
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                )
                task.add_write_requirement(
                    result.region, result.field.field_id
                )
                task.add_read_requirement(rhs1.region, rhs1.field.field_id)
                task.add_read_requirement(rhs2.region, rhs2.field.field_id)
                self.runtime.dispatch(task)
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
        assert rhs.dtype == self.dtype
        if extract:
            matrix_array = self.runtime.to_deferred_array(
                rhs, stacklevel=(stacklevel + 1)
            )
            diag_array = self
            assert diag_array.ndim == 1
            assert matrix_array.ndim == 2
            assert diag_array.shape[0] <= min(
                matrix_array.shape[0], matrix_array.shape[1]
            )
            assert diag_array.dtype == matrix_array.dtype
            if matrix_array.shape[0] < matrix_array.shape[1]:
                offset = k if k < 0 else 0
            else:
                offset = -k if k > 0 else 0
            if offset != 0:
                raise NotImplementedError(
                    "Need support for offset!=0 in diag extract"
                )
            diag = diag_array.base
            matrix = matrix_array.base
            launch_space = matrix.compute_parallel_launch_space()
            if launch_space is not None:
                # Index space launch case Find a partition for diagonal,
                # partition on the smaller tile size since that represents the
                # upper bound on the number of elements needed
                (
                    matrix_part,
                    shardfn,
                    shardsp,
                ) = matrix.find_or_create_key_partition()
                if matrix_array.shape[0] < matrix_array.shape[1]:
                    assert matrix_array.shape[0] == diag_array.shape[0]
                    reduction_shape = (diag_array.shape[0], launch_space[1])
                    # Check to see if we even need a reduction field
                    if launch_space[1] > 1:
                        collapse_dim = 1
                        collapse_index = 1
                        reduction_field = self.runtime.allocate_field(
                            reduction_shape, diag_array.dtype
                        )
                        reduction_part = reduction_field.find_or_create_partition(  # noqa E501
                            launch_space,
                            tile_shape=(matrix_part.tile_shape[0], 1),
                            offset=(matrix_part.tile_offset[0], 0),
                        )
                        reduction_proj = 0
                    else:
                        # Only partitioning is on the other dimension so we can
                        # put our output directly into the diag array
                        collapse_dim = -1
                        collapse_index = -1
                        reduction_field = diag
                        reduction_part = diag.find_or_create_partition(
                            (launch_space[0],),
                            tile_shape=(matrix_part.tile_shape[0],),
                            offset=(matrix_part.tile_offset[0],),
                        )
                        reduction_proj = (
                            self.runtime.first_proj_id
                            + NumPyProjCode.PROJ_2D_1D_X
                        )
                else:
                    assert matrix_array.shape[1] == diag_array.shape[0]
                    reduction_shape = (diag_array.shape[0], launch_space[0])
                    # Check to see if we even need a reduction field
                    if launch_space[0] > 1:
                        collapse_dim = 1
                        collapse_index = 0
                        reduction_field = self.runtime.allocate_field(
                            reduction_shape, diag_array.dtype
                        )
                        reduction_part = reduction_field.find_or_create_partition(  # noqa E501
                            (launch_space[1], launch_space[0]),
                            tile_shape=(matrix_part.tile_shape[1], 1),
                            offset=(matrix_part.tile_offset[1], 0),
                        )
                        reduction_proj = (
                            self.runtime.first_proj_id
                            + NumPyProjCode.PROJ_2D_2D_YX
                        )
                    else:
                        # Only partitioning is on the other dimension so we can
                        # put our output directly into the diag array
                        collapse_dim = -1
                        collapse_index = -1
                        reduction_field = diag
                        reduction_part = diag.find_or_create_partition(
                            (launch_space[1],),
                            tile_shape=(matrix_part.tile_shape[1],),
                            offset=(matrix_part.tile_offset[1],),
                        )
                        reduction_proj = (
                            self.runtime.first_proj_id
                            + NumPyProjCode.PROJ_2D_1D_Y
                        )
                # Do an index space launch over the matrix to fill in the
                # reduction field
                argbuf = BufferBuilder()
                argbuf.pack_32bit_int(k)
                self.pack_shape(
                    argbuf,
                    matrix_array.shape,
                    matrix_part.tile_shape,
                    proj=0,
                    pack_dim=False,
                )
                argbuf.pack_dimension(
                    collapse_dim
                )  # Dimension that we're collapsing
                argbuf.pack_dimension(
                    collapse_index
                )  # Index of the collapsing dimension
                argbuf.pack_accessor(
                    reduction_field.field.field_id, reduction_field.transform
                )
                argbuf.pack_accessor(matrix.field.field_id, matrix.transform)
                task = IndexTask(
                    self.runtime.get_nullary_task_id(
                        NumPyOpCode.DIAG, result_type=diag_array.dtype
                    ),
                    Rect(launch_space),
                    self.runtime.empty_argmap,
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
                if shardsp is not None:
                    task.set_sharding_space(shardsp)
                task.add_write_requirement(
                    reduction_part,
                    reduction_field.field.field_id,
                    reduction_proj,
                )
                task.add_read_requirement(
                    matrix_part,
                    matrix.field.field_id,
                    0,  # identity
                    tag=NumPyMappingTag.KEY_REGION_TAG,
                )
                self.runtime.dispatch(task)
                if reduction_shape[1] > 1:
                    # Now build reduction trees until we get to the bottom
                    RADIX = self.runtime.radix
                    # We want the output vector partitioned in the natural way
                    # for it
                    launch_space = diag.compute_parallel_launch_space()
                    if launch_space is None:
                        diag_part = diag.find_or_create_partition((1,))
                        launch_space = (1, reduction_shape[1])
                    else:
                        diag_part = diag.find_or_create_partition(launch_space)
                        launch_space = (launch_space[0], reduction_shape[1])
                    diag_proj = (
                        self.runtime.first_proj_id + NumPyProjCode.PROJ_2D_1D_X
                    )
                    sharding_space = self.runtime.find_or_create_index_space(
                        launch_space
                    )
                    reduction_part = reduction_field.find_or_create_partition(
                        launch_space
                    )
                    radix_generation = 1
                    while reduction_shape[1] > 1:
                        argbuf = BufferBuilder()
                        argbuf.pack_dimension(RADIX)
                        new_reduction_shape = (
                            diag_array.shape[0],
                            (reduction_shape[1] + RADIX - 1) // RADIX,
                        )
                        local_launch_space = (
                            launch_space[0],
                            new_reduction_shape[1],
                        )
                        if new_reduction_shape[1] == 1:
                            # We can put the result directly into the output
                            # since this is the last reduction
                            new_reduction_field = diag
                            new_reduction_part = diag_part
                            new_reduction_proj = diag_proj
                            argbuf.pack_dimension(
                                -1
                            )  # No collapse out dimension
                        else:
                            new_reduction_field = self.runtime.allocate_field(
                                new_reduction_shape,
                                reduction_field.field.dtype,
                            )
                            new_reduction_part = new_reduction_field.find_or_create_partition(  # noqa E501
                                local_launch_space
                            )
                            new_reduction_proj = 0  # identity projection
                            argbuf.pack_dimension(
                                1
                            )  # collapsing out dimension 1
                        argbuf.pack_dimension(1)  # Collapsing in dimension 1
                        self.pack_shape(
                            argbuf,
                            diag_array.shape,
                            diag_part.tile_shape,
                            diag_proj,
                        )
                        argbuf.pack_accessor(
                            new_reduction_field.field.field_id,
                            new_reduction_field.transform,
                        )
                        for idx in xrange(RADIX):
                            argbuf.pack_accessor(
                                reduction_field.field.field_id,
                                reduction_field.transform,
                            )
                        task = IndexTask(
                            self.runtime.get_binary_task_id(
                                NumPyOpCode.SUM_RADIX,
                                result_type=reduction_field.field.dtype,
                                first_argument_type=reduction_field.field.dtype,  # noqa E501
                                second_argument_type=reduction_field.field.dtype,  # noqa E501
                            ),
                            Rect(local_launch_space),
                            self.runtime.empty_argmap,
                            argbuf.get_string(),
                            argbuf.get_size(),
                            mapper=self.runtime.mapper_id,
                            tag=(
                                NumPyMappingTag.RADIX_GEN_TAG
                                & (radix_generation << RADIX_GEN_SHIFT)
                            )
                            | (
                                NumPyMappingTag.RADIX_DIM_TAG
                                & (1 << RADIX_DIM_SHIFT)
                            ),
                        )
                        task.set_sharding_space(sharding_space)
                        task.add_write_requirement(
                            new_reduction_part,
                            new_reduction_field.field.field_id,
                            new_reduction_proj,
                            tag=NumPyMappingTag.KEY_REGION_TAG,
                            flags=legion.LEGION_COMPLETE_PROJECTION_WRITE_FLAG,
                        )
                        for idx in xrange(RADIX):
                            task.add_read_requirement(
                                reduction_part,
                                reduction_field.field.field_id,
                                self.runtime.get_radix_projection_functor_id(
                                    2, 1, RADIX, idx
                                ),
                            )
                        self.runtime.dispatch(task)
                        reduction_shape = new_reduction_shape
                        reduction_field = new_reduction_field
                        reduction_part = new_reduction_part
                        radix_generation += 1
            else:
                # Sequential diag case
                argbuf = BufferBuilder()
                argbuf.pack_32bit_int(k)
                self.pack_shape(argbuf, matrix_array.shape, pack_dim=False)
                argbuf.pack_dimension(-1)  # No collapse dimension
                argbuf.pack_dimension(-1)  # No collapse index
                argbuf.pack_accessor(diag.field.field_id, diag.transform)
                argbuf.pack_accessor(matrix.field.field_id, matrix.transform)
                shardpt, shardfn, shardsp = matrix.find_point_sharding()
                task = Task(
                    self.runtime.get_nullary_task_id(
                        NumPyOpCode.DIAG, result_type=diag_array.dtype
                    ),
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
                if shardpt is not None:
                    task.set_point(shardpt)
                if shardsp is not None:
                    task.set_sharding_space(shardsp)
                task.add_write_requirement(diag.region, diag.field.field_id)
                task.add_read_requirement(matrix.region, matrix.field.field_id)
                self.runtime.dispatch(task)
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
            assert diag_array.dtype == matrix_array.dtype
            argbuf = BufferBuilder()
            argbuf.pack_32bit_int(k)
            diag = diag_array.base
            matrix = matrix_array.base
            launch_space = matrix.compute_parallel_launch_space()
            if launch_space is not None:
                (
                    matrix_part,
                    shardfn,
                    shardsp,
                ) = matrix.find_or_create_key_partition()
                self.pack_shape(
                    argbuf,
                    matrix_array.shape,
                    matrix_part.tile_shape,
                    proj=0,
                    pack_dim=False,
                )
                # Then find a partition for diagonal, partition on the smaller
                # tile size since that represents the upper bound on the number
                # of elements needed
                assert len(matrix_part.tile_shape) == 2
                if matrix_part.tile_shape[0] < matrix_part.tile_shape[1]:
                    # If we have a negative k, use it as an offset
                    if k < 0:
                        diag_part = diag.find_or_create_partition(
                            (launch_space[0],),
                            tile_shape=(matrix_part.tile_shape[0],),
                            offset=(matrix_part.tile_offset[0] + k,),
                        )
                    else:
                        diag_part = diag.find_or_create_partition(
                            (launch_space[0],),
                            tile_shape=(matrix_part.tile_shape[0],),
                            offset=(matrix_part.tile_offset[0],),
                        )
                    diag_proj = (
                        self.runtime.first_proj_id + NumPyProjCode.PROJ_2D_1D_X
                    )
                else:
                    # If we have a positive k, use it as a negative offset
                    if k > 0:
                        diag_part = diag.find_or_create_partition(
                            (launch_space[1],),
                            tile_shape=(matrix_part.tile_shape[1],),
                            offset=(matrix_part.tile_offset[1] + -k,),
                        )
                    else:
                        diag_part = diag.find_or_create_partition(
                            (launch_space[1],),
                            tile_shape=(matrix_part.tile_shape[1],),
                            offset=(matrix_part.tile_offset[1],),
                        )
                    diag_proj = (
                        self.runtime.first_proj_id + NumPyProjCode.PROJ_2D_1D_Y
                    )
            else:
                self.pack_shape(argbuf, matrix_array.shape, pack_dim=False)
            argbuf.pack_accessor(diag.field.field_id, diag.transform)
            argbuf.pack_accessor(matrix.field.field_id, matrix.transform)
            if launch_space is not None:
                task = IndexTask(
                    self.runtime.get_nullary_task_id(
                        NumPyOpCode.DIAG, result_type=diag_array.dtype
                    ),
                    Rect(launch_space),
                    self.runtime.empty_argmap,
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
                if shardsp is not None:
                    task.set_sharding_space(shardsp)
                task.add_read_requirement(
                    diag_part,
                    diag.field.field_id,
                    diag_proj,
                    tag=NumPyMappingTag.NO_MEMOIZE_TAG,
                )
                task.add_read_write_requirement(
                    matrix_part,
                    matrix.field.field_id,
                    0,
                    tag=NumPyMappingTag.KEY_REGION_TAG,
                )
            else:
                shardpt, shardfn, shardsp = matrix.find_point_sharding()
                task = Task(
                    self.runtime.get_nullary_task_id(
                        NumPyOpCode.DIAG, result_type=diag_array.dtype
                    ),
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
                if shardpt is not None:
                    task.set_point(shardpt)
                if shardsp is not None:
                    task.set_sharding_space(shardsp)
                task.add_read_requirement(diag.region, diag.field.field_id)
                task.add_read_write_requirement(
                    matrix.region, matrix.field.field_id
                )
            # Before we issue the task, we have to issue a fill operation
            # to initialize the matrix with zeros
            matrix_array.fill(
                np.array(0, dtype=matrix_array.dtype),
                stacklevel=(stacklevel + 1),
            )
            self.runtime.dispatch(task)
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
        dst = self.base
        # First issue a fill to zero everything out
        self.fill(np.array(0, dtype=self.dtype), stacklevel=(stacklevel + 1))
        launch_space = dst.compute_parallel_launch_space()
        # Then issue a task to fill in the diagonal
        argbuf = BufferBuilder()
        if launch_space is not None:
            dst_part, shardfn, shardsp = dst.find_or_create_key_partition()
            self.pack_shape(
                argbuf, self.shape, dst_part.tile_shape, proj=0, pack_dim=False
            )
        else:
            self.pack_shape(argbuf, self.shape, pack_dim=False)
        argbuf.pack_accessor(dst.field.field_id, dst.transform)
        argbuf.pack_32bit_int(k)
        if launch_space is not None:
            # We do the full index space launch, but we the implementation
            # will quickly filter out certain points
            task = IndexTask(
                self.runtime.get_nullary_task_id(
                    NumPyOpCode.EYE, result_type=self.dtype
                ),
                Rect(launch_space),
                self.runtime.empty_argmap,
                argbuf.get_string(),
                argbuf.get_size(),
                mapper=self.runtime.mapper_id,
                tag=shardfn,
            )
            if shardsp is not None:
                task.set_sharding_space(shardsp)
            task.add_read_write_requirement(
                dst_part,
                dst.field.field_id,
                0,
                tag=NumPyMappingTag.KEY_REGION_TAG,
            )
            self.runtime.dispatch(task)
        else:
            shardpt, shardfn, shardsp = dst.find_point_sharding()
            task = Task(
                self.runtime.get_nullary_task_id(
                    NumPyOpCode.EYE, result_type=self.dtype
                ),
                argbuf.get_string(),
                argbuf.get_size(),
                mapper=self.runtime.mapper_id,
                tag=shardfn,
            )
            if shardpt is not None:
                task.set_point(shardpt)
            if shardsp is not None:
                task.set_sharding_space(shardsp)
            task.add_read_write_requirement(dst.region, dst.field.field_id)
            self.runtime.dispatch(task)
        self.runtime.profile_callsite(stacklevel + 1, True, callsite)
        # See if we are doing shadow debugging
        if self.runtime.shadow_debug:
            self.shadow.eye(k=k, stacklevel=(stacklevel + 1))
            self.runtime.check_shadow(self, "eye")

    def arange(self, start, stop, step, stacklevel, callsite=None):
        assert self.ndim == 1  # Only 1-D arrays should be here
        dst = self.base
        launch_space = dst.compute_parallel_launch_space()
        argbuf = BufferBuilder()
        if launch_space is not None:
            dst_part, shardfn, shardsp = dst.find_or_create_key_partition()
            self.pack_shape(
                argbuf, self.shape, dst_part.tile_shape, proj=0, pack_dim=False
            )
        else:
            self.pack_shape(argbuf, self.shape, pack_dim=False)
        argbuf.pack_accessor(dst.field.field_id)

        def create_future(value):
            array = np.array(value, dtype=self.dtype)
            return self.runtime.create_future(array.data, array.nbytes)

        start = create_future(start)
        stop = create_future(stop)
        step = create_future(step)
        if launch_space is not None:
            task = IndexTask(
                self.runtime.get_nullary_task_id(
                    NumPyOpCode.ARANGE, result_type=self.dtype
                ),
                Rect(launch_space),
                self.runtime.empty_argmap,
                argbuf.get_string(),
                argbuf.get_size(),
                mapper=self.runtime.mapper_id,
                tag=shardfn,
            )
            if shardsp is not None:
                task.set_sharding_space(shardsp)
            task.add_write_requirement(
                dst_part,
                dst.field.field_id,
                0,
                tag=NumPyMappingTag.KEY_REGION_TAG,
            )
            task.add_future(start)
            task.add_future(stop)
            task.add_future(step)
            self.runtime.dispatch(task)
        else:
            shardpt, shardfn, shardsp = dst.find_point_sharding()
            task = Task(
                self.runtime.get_nullary_task_id(
                    NumPyOpCode.ARANGE, result_type=self.dtype
                ),
                argbuf.get_string(),
                argbuf.get_size(),
                mapper=self.runtime.mapper_id,
                tag=shardfn,
            )
            if shardpt is not None:
                task.set_point(shardpt)
            if shardsp is not None:
                task.set_sharding_space(shardsp)
            task.add_write_requirement(dst.region, dst.field.field_id)
            task.add_future(start)
            task.add_future(stop)
            task.add_future(step)
            self.runtime.dispatch(task)
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
        if dst_array.size == 1:
            # Easy case, we can just copy the future
            dst_array.base = src_array.base
            return
        src = src_array.base
        dst = dst_array.base
        launch_space = dst.compute_parallel_launch_space()
        if src_array.size == 1:
            # Another easy case, we can just do a fill with the future
            if launch_space is not None:
                dst_part, shardfn, shardsp = dst.find_or_create_key_partition()
                fill = IndexFill(
                    dst_part,
                    0,
                    dst.region.get_root(),
                    dst.field.field_id,
                    src,
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
                if shardsp is not None:
                    fill.set_sharding_space(shardsp)
            else:
                shardpt, shardfn, shardsp = dst.find_point_sharding()
                fill = Fill(
                    dst.region,
                    dst.region.get_root(),
                    dst.field.field_id,
                    src,
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
                if shardpt is not None:
                    fill.set_point(shardpt)
                if shardsp is not None:
                    fill.set_sharding_space(shardsp)
            self.runtime.dispatch(fill)
        else:
            # The common case where we have to do the tiling
            # First pack the dst shape
            argbuf = BufferBuilder()
            if launch_space is not None:
                dst_part, shardfn, shardsp = dst.find_or_create_key_partition()
                self.pack_shape(
                    argbuf, dst_array.shape, dst_part.tile_shape, 0
                )
            else:
                self.pack_shape(argbuf, dst_array.shape)
            argbuf.pack_accessor(dst.field.field_id, None)
            # Then pack the src dim if dst.ndim > 1
            if dst_array.ndim > 1:
                argbuf.pack_dimension(src_array.ndim)
            argbuf.pack_point(src_array.shape)
            argbuf.pack_accessor(src.field.field_id, None)
            if launch_space is not None:
                task = IndexTask(
                    self.runtime.get_unary_task_id(
                        NumPyOpCode.TILE,
                        result_type=dst_array.dtype,
                        argument_type=dst_array.dtype,
                    ),
                    Rect(launch_space),
                    self.runtime.empty_argmap,
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
                if shardsp is not None:
                    task.set_sharding_space(shardsp)
                task.add_write_requirement(
                    dst_part,
                    dst.field.field_id,
                    0,
                    tag=NumPyMappingTag.KEY_REGION_TAG,
                )
                # Pack the whole source region, we'll assume it is small
                # Projection 0 on regions are just the region itself
                task.add_read_requirement(
                    src.region,
                    src.field.field_id,
                    0,
                    tag=NumPyMappingTag.NO_MEMOIZE_TAG,
                )
                self.runtime.dispatch(task)
            else:
                shardpt, shardfn, shardsp = dst.find_point_sharding()
                task = Task(
                    self.runtime.get_unary_task_id(
                        NumPyOpCode.TILE,
                        result_type=dst_array.dtype,
                        argument_type=dst_array.dtype,
                    ),
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
                if shardpt is not None:
                    task.set_point(shardpt)
                if shardsp is not None:
                    task.set_sharding_space(shardsp)
                task.add_write_requirement(dst.region, dst.field.field_id)
                task.add_read_requirement(src.region, src.field.field_id)
                self.runtime.dispatch(task)
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
        src = rhs_array.base
        dst = lhs_array.base
        launch_space = dst.compute_parallel_launch_space()
        # Now launch the task
        argbuf = BufferBuilder()
        if launch_space is not None:
            dst_part, shardfn, shardsp = dst.find_or_create_key_partition()
            self.pack_shape(argbuf, lhs_array.shape, dst_part.tile_shape, 0)
        else:
            self.pack_shape(argbuf, lhs_array.shape)
        argbuf.pack_accessor(dst.field.field_id, dst.transform)
        argbuf.pack_accessor(src.field.field_id, src.transform)
        if launch_space is not None:
            task = IndexTask(
                self.runtime.get_unary_task_id(
                    NumPyOpCode.TRANSPOSE,
                    result_type=lhs_array.dtype,
                    argument_type=rhs_array.dtype,
                ),
                Rect(launch_space),
                self.runtime.empty_argmap,
                argbuf.get_string(),
                argbuf.get_size(),
                mapper=self.runtime.mapper_id,
                tag=shardfn,
            )
            if shardsp is not None:
                task.set_sharding_space(shardsp)
            task.add_write_requirement(
                dst_part,
                dst.field.field_id,
                0,
                tag=NumPyMappingTag.KEY_REGION_TAG,
            )
            # Compute the partition using the transposed launch space
            src_launch_space = tuple(map(lambda x: launch_space[x], axes))
            src_tile_shape = tuple(map(lambda x: dst_part.tile_shape[x], axes))
            src_offset = tuple(map(lambda x: dst_part.tile_offset[x], axes))
            src_part = src.find_or_create_partition(
                src_launch_space, src_tile_shape, src_offset
            )
            # TODO: change the hard-coded projection function for 2D
            task.add_read_requirement(
                src_part,
                src.field.field_id,
                self.runtime.first_proj_id + NumPyProjCode.PROJ_2D_2D_YX,
                tag=NumPyMappingTag.NO_MEMOIZE_TAG,
            )
            self.runtime.dispatch(task)
        else:
            shardpt, shardfn, shardsp = dst.find_point_sharding()
            task = Task(
                self.runtime.get_unary_task_id(
                    NumPyOpCode.TRANSPOSE,
                    result_type=lhs_array.dtype,
                    argument_type=rhs_array.dtype,
                ),
                argbuf.get_string(),
                argbuf.get_size(),
                mapper=self.runtime.mapper_id,
                tag=shardfn,
            )
            if shardpt is not None:
                task.set_point(shardpt)
            if shardsp is not None:
                task.set_sharding_space(shardsp)
            task.add_write_requirement(dst.region, dst.field.field_id)
            task.add_read_requirement(src.region, src.field.field_id)
            self.runtime.dispatch(task)
        self.runtime.profile_callsite(stacklevel + 1, True, callsite)
        # See if we are doing shadow debugging
        if self.runtime.shadow_debug:
            self.shadow.transpose(
                rhs.shadow, axes, stacklevel=(stacklevel + 1)
            )
            self.runtime.check_shadow(self, "transpose")

    # Perform a bin count operation on the array
    def bincount(self, rhs, stacklevel, weights=None, callsite=None):
        weight_array = (
            None
            if weights is None
            else self.runtime.to_deferred_array(
                weights, stacklevel=(stacklevel + 1)
            )
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
        src = src_array.base
        dst = dst_array.base
        if weight_array is not None:
            weights = weight_array.base
        launch_space = src.compute_parallel_launch_space()
        if launch_space is not None:
            src_part, shardfn, shardsp = src.find_or_create_key_partition()
            if src_array.ndim > 1:
                raise NotImplementedError(
                    "Need support for higher dimensions in index bincount"
                )
            assert len(launch_space) == 1
            # Construct a reduction array
            reduction_shape = dst_array.shape + launch_space
            reduction_field = self.runtime.allocate_field(
                reduction_shape, dst_array.dtype
            )
            reduction_part = reduction_field.find_or_create_partition(
                (1, launch_space[0]), tile_shape=(reduction_shape[0], 1)
            )
            reduction_proj = (
                self.runtime.first_proj_id + NumPyProjCode.PROJ_1D_2D_Y
            )
            argbuf = BufferBuilder()
            argbuf.pack_dimension(1)  # Dimension that we're collapsing
            argbuf.pack_dimension(0)  # Dimension index
            self.pack_shape(argbuf, dst_array.shape, pack_dim=False)
            argbuf.pack_accessor(
                reduction_field.field.field_id, reduction_field.transform
            )
            self.pack_shape(
                argbuf, src_array.shape, src_part.tile_shape, proj=0
            )
            argbuf.pack_accessor(src.field.field_id, src.transform)
            if weight_array is not None:
                argbuf.pack_accessor(weights.field.field_id, weights.transform)
                task = IndexTask(
                    self.runtime.get_weighted_bincount_task_id(
                        src_array.dtype, weight_array.dtype
                    ),
                    Rect(launch_space),
                    self.runtime.empty_argmap,
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
            else:
                task = IndexTask(
                    self.runtime.get_unary_task_id(
                        NumPyOpCode.BINCOUNT,
                        argument_type=src_array.dtype,
                        result_type=dst_array.dtype,
                    ),
                    Rect(launch_space),
                    self.runtime.empty_argmap,
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
            if shardsp is not None:
                task.set_sharding_space(shardsp)
            task.add_write_requirement(
                reduction_part, reduction_field.field.field_id, reduction_proj
            )
            task.add_read_requirement(
                src_part,
                src.field.field_id,
                0,  # identity projection
                tag=NumPyMappingTag.KEY_REGION_TAG,
            )
            if weight_array is not None:
                weight_part = weights.find_or_create_partition(launch_space)
                task.add_read_requirement(
                    weight_part, weights.field.field_id, 0
                )
            self.runtime.dispatch(task)
            assert reduction_shape[1] > 1
            # Now build reduction trees until we get to the bottom
            RADIX = self.runtime.radix
            # We want the output vector partitioned in the natural way for it
            launch_space = dst.compute_parallel_launch_space()
            if launch_space is None:
                dst_part = dst.find_or_create_partition((1,))
                launch_space = (1, reduction_shape[1])
            else:
                dst_part = dst.find_or_create_partition(launch_space)
                launch_space = (launch_space[0], reduction_shape[1])
            dst_proj = self.runtime.first_proj_id + NumPyProjCode.PROJ_2D_1D_X
            sharding_space = self.runtime.find_or_create_index_space(
                launch_space
            )
            reduction_part = reduction_field.find_or_create_partition(
                launch_space
            )
            radix_generation = 1
            while reduction_shape[1] > 1:
                argbuf = BufferBuilder()
                argbuf.pack_dimension(RADIX)
                new_reduction_shape = (
                    dst_array.shape[0],
                    (reduction_shape[1] + RADIX - 1) // RADIX,
                )
                local_launch_space = (launch_space[0], new_reduction_shape[1])
                if new_reduction_shape[1] == 1:
                    # We can put the result directly into the output
                    # since this is the last reduction
                    new_reduction_field = dst
                    new_reduction_part = dst_part
                    new_reduction_proj = dst_proj
                    argbuf.pack_dimension(-1)  # No collapse out dimension
                else:
                    new_reduction_field = self.runtime.allocate_field(
                        new_reduction_shape, reduction_field.field.dtype
                    )
                    new_reduction_part = new_reduction_field.find_or_create_partition(  # noqa E501
                        local_launch_space
                    )
                    new_reduction_proj = 0  # identity projection
                    argbuf.pack_dimension(1)  # collapsing out dimension 1
                argbuf.pack_dimension(1)  # Collapsing in dimension 1
                self.pack_shape(
                    argbuf, dst_array.shape, dst_part.tile_shape, dst_proj
                )
                argbuf.pack_accessor(
                    new_reduction_field.field.field_id,
                    new_reduction_field.transform,
                )
                for idx in xrange(RADIX):
                    argbuf.pack_accessor(
                        reduction_field.field.field_id,
                        reduction_field.transform,
                    )
                task = IndexTask(
                    self.runtime.get_binary_task_id(
                        NumPyOpCode.SUM_RADIX,
                        result_type=reduction_field.field.dtype,
                        first_argument_type=reduction_field.field.dtype,
                        second_argument_type=reduction_field.field.dtype,
                    ),
                    Rect(local_launch_space),
                    self.runtime.empty_argmap,
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                    tag=(
                        NumPyMappingTag.RADIX_GEN_TAG
                        & (radix_generation << RADIX_GEN_SHIFT)
                    )
                    | (NumPyMappingTag.RADIX_DIM_TAG & (1 << RADIX_DIM_SHIFT)),
                )
                task.set_sharding_space(sharding_space)
                task.add_write_requirement(
                    new_reduction_part,
                    new_reduction_field.field.field_id,
                    new_reduction_proj,
                    tag=NumPyMappingTag.KEY_REGION_TAG,
                    flags=legion.LEGION_COMPLETE_PROJECTION_WRITE_FLAG,
                )
                for idx in xrange(RADIX):
                    task.add_read_requirement(
                        reduction_part,
                        reduction_field.field.field_id,
                        self.runtime.get_radix_projection_functor_id(
                            2, 1, RADIX, idx
                        ),
                    )
                self.runtime.dispatch(task)
                reduction_shape = new_reduction_shape
                reduction_field = new_reduction_field
                reduction_part = new_reduction_part
                radix_generation += 1
        else:
            # Sequential bincount case
            argbuf = BufferBuilder()
            argbuf.pack_dimension(-1)  # Not collapsing any dimensions here
            argbuf.pack_dimension(-1)  # dimension index
            self.pack_shape(argbuf, dst_array.shape, pack_dim=False)
            argbuf.pack_accessor(dst.field.field_id, dst.transform)
            self.pack_shape(argbuf, src_array.shape)
            argbuf.pack_accessor(src.field.field_id, src.transform)
            shardpt, shardfn, shardsp = src.find_point_sharding()
            if weight_array is not None:
                argbuf.pack_accessor(weights.field.field_id, weights.transform)
                task = Task(
                    self.runtime.get_weighted_bincount_task_id(
                        src_array.dtype, weight_array.dtype
                    ),
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
            else:
                task = Task(
                    self.runtime.get_unary_task_id(
                        NumPyOpCode.BINCOUNT,
                        result_type=dst_array.dtype,
                        argument_type=src_array.dtype,
                    ),
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
            if shardpt is not None:
                task.set_point(shardpt)
            if shardsp is not None:
                task.set_sharding_space(shardsp)
            task.add_write_requirement(dst.region, dst.field.field_id)
            task.add_read_requirement(src.region, src.field.field_id)
            if weight_array is not None:
                task.add_read_requirement(
                    weights.region, weights.field.field_id
                )
            self.runtime.dispatch(task)
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
            NumPyOpCode.COUNT_NONZERO,
            NumPyOpCode.SUM_RADIX,
            self,
            axis=axis,
            dtype=np.dtype(np.uint64),
            stacklevel=(stacklevel + 1),
            check_types=False,
        )

    def nonzero(self, stacklevel, callsite=None):
        # First we need to count how many non-zero elements there are and then
        # we can make arrays for recording the indexes in each dimension

        # This case should be handled in the frontend
        assert self.ndim > 0
        # We cannot call into the upper abstraction layer's count_nonzero, so
        # we do a simplified version here
        dst_nonzero = self.runtime.create_empty_thunk(
            shape=(1,), dtype=np.dtype(np.uint64), inputs=(self,)
        )
        dst_nonzero.unary_reduction(
            NumPyOpCode.COUNT_NONZERO,
            NumPyOpCode.SUM_RADIX,
            self,
            stacklevel=(stacklevel + 1),
            axes=None,
            keepdims=False,
            where=True,
            initial=None,
            args=None,
        )
        num_nonzero = int(
            dst_nonzero.get_scalar_array(stacklevel=(stacklevel + 1))
        )
        if num_nonzero == 0:
            result = ()
            for _ in xrange(self.ndim):
                result += (
                    self.runtime.create_empty_thunk(
                        shape=(0,), dtype=np.dtype(np.uint64), inputs=(self,)
                    ),
                )
            return result
        launch_space = self.base.compute_parallel_launch_space()
        out_shape = (self.ndim, num_nonzero)
        # TODO: A better way to create the output array?
        dst_array = self.runtime.to_deferred_array(
            self.runtime.create_empty_thunk(
                shape=out_shape, dtype=np.dtype(np.uint64), inputs=(self,)
            ),
            stacklevel=(stacklevel + 1),
        )
        dst = dst_array.base
        rhs = self.base
        if launch_space is not None:  # Index task launch
            # First get the amount of nonzeros for each point in the
            # launch_space
            nonzeros_dist_shape = (calculate_volume(launch_space),)
            nonzeros_dist_field = self.runtime.allocate_field(
                nonzeros_dist_shape, np.dtype(np.uint64)
            )
            nonzeros_dist_part = nonzeros_dist_field.find_or_create_partition(
                nonzeros_dist_shape
            )
            launch_part, shardfn, shardsp = rhs.find_or_create_key_partition()
            argbuf = BufferBuilder()
            self.pack_shape(argbuf, self.shape, launch_part.tile_shape, 0)
            argbuf.pack_accessor(rhs.field.field_id, rhs.transform)
            self.pack_shape(
                argbuf,
                nonzeros_dist_shape,
                nonzeros_dist_part.tile_shape,
                self.runtime.first_proj_id + NumPyProjCode.PROJ_ND_1D_C_ORDER,
            )
            argbuf.pack_accessor(
                nonzeros_dist_field.field.field_id,
                nonzeros_dist_field.transform,
            )
            index_task = IndexTask(
                self.runtime.get_unary_task_id(
                    NumPyOpCode.COUNT_NONZERO_REDUC,
                    result_type=np.dtype(np.uint64),
                    argument_type=self.dtype,
                ),
                Rect(launch_space),
                self.runtime.empty_argmap,
                argbuf.get_string(),
                argbuf.get_size(),
                mapper=self.runtime.mapper_id,
                tag=shardfn,
            )
            if shardsp is not None:
                task.set_sharding_space(shardsp)
            index_task.add_read_requirement(
                launch_part,
                self.base.field.field_id,
                0,
                tag=NumPyMappingTag.KEY_REGION_TAG,
            )
            index_task.add_write_requirement(
                nonzeros_dist_part,
                nonzeros_dist_field.field.field_id,
                self.runtime.first_proj_id + NumPyProjCode.PROJ_ND_1D_C_ORDER,
                tag=NumPyMappingTag.NO_MEMOIZE_TAG,
            )
            self.runtime.dispatch(index_task)

            argbuf = BufferBuilder()
            self.pack_shape(argbuf, nonzeros_dist_shape, pack_dim=False)
            argbuf.pack_accessor(
                nonzeros_dist_field.field.field_id,
                nonzeros_dist_field.transform,
            )
            task = Task(
                self.runtime.get_unary_task_id(
                    NumPyOpCode.INCLUSIVE_SCAN,
                    result_type=nonzeros_dist_field.field.dtype,
                    argument_type=nonzeros_dist_field.field.dtype,
                ),
                argbuf.get_string(),
                argbuf.get_size(),
                mapper=self.runtime.mapper_id,
            )
            task.add_read_write_requirement(
                nonzeros_dist_field.region, nonzeros_dist_field.field.field_id
            )
            self.runtime.dispatch(task)

            nonzeros_ranges_field = self.runtime.allocate_field(
                nonzeros_dist_shape,
                np.dtype((np.void, ffi.sizeof("legion_rect_2d_t"))),
            )
            argbuf = BufferBuilder()
            argbuf.pack_32bit_int(self.ndim - 1)
            self.pack_shape(argbuf, nonzeros_dist_shape)
            argbuf.pack_accessor(
                nonzeros_dist_field.field.field_id,
                nonzeros_dist_field.transform,
            )
            argbuf.pack_accessor(
                nonzeros_ranges_field.field.field_id,
                nonzeros_ranges_field.transform,
            )
            task = Task(
                self.runtime.get_unary_task_id(
                    NumPyOpCode.CONVERT_TO_RECT,
                    result_type=np.dtype(np.uint64),
                    argument_type=nonzeros_dist_field.field.dtype,
                ),
                argbuf.get_string(),
                argbuf.get_size(),
                mapper=self.runtime.mapper_id,
            )
            task.add_read_requirement(
                nonzeros_dist_field.region, nonzeros_dist_field.field.field_id
            )
            task.add_write_requirement(
                nonzeros_ranges_field.region,
                nonzeros_ranges_field.field.field_id,
            )
            self.runtime.dispatch(task)

            nonzeros_projection = (
                nonzeros_ranges_field.find_or_create_partition(  # noqa E501
                    nonzeros_dist_shape
                )
            )
            functor = PartitionByImageRange(
                nonzeros_ranges_field.region,
                nonzeros_projection,
                nonzeros_ranges_field.field.field_id,
                self.runtime.mapper_id,
            )
            index_partition = IndexPartition(
                self.runtime.context,
                self.runtime.runtime,
                dst.region.index_space,
                nonzeros_projection.color_space,
                functor,
                kind=legion.DISJOINT_COMPLETE_KIND,
            )
            dst_partition = dst.region.get_child(index_partition)

            argbuf = BufferBuilder()
            self.pack_shape(argbuf, self.shape, launch_part.tile_shape, 0)
            argbuf.pack_accessor(rhs.field.field_id, rhs.transform)
            argbuf.pack_accessor(dst.field.field_id, dst.transform)
            index_task = IndexTask(
                self.runtime.get_unary_task_id(
                    NumPyOpCode.NONZERO,
                    result_type=np.dtype(np.uint64),
                    argument_type=self.dtype,
                ),
                Rect(launch_space),
                self.runtime.empty_argmap,
                argbuf.get_string(),
                argbuf.get_size(),
                mapper=self.runtime.mapper_id,
                tag=shardfn,
            )
            index_task.add_read_requirement(
                launch_part,
                self.base.field.field_id,
                0,
                tag=NumPyMappingTag.KEY_REGION_TAG,
            )
            index_task.add_write_requirement(
                dst_partition,
                dst.field.field_id,
                self.runtime.first_proj_id + NumPyProjCode.PROJ_ND_1D_C_ORDER,
                tag=NumPyMappingTag.NO_MEMOIZE_TAG,
            )
            self.runtime.dispatch(index_task)
        else:  # Single task launch
            argbuf = BufferBuilder()
            self.pack_shape(argbuf, self.shape)
            argbuf.pack_accessor(rhs.field.field_id, rhs.transform)
            # Shape of the destination is not packed since it can be irregular
            # in multiple launch space, and it needs to be unpacked
            # dynamically.  self.pack_shape(argbuf, dst.shape, pack_dim=False)
            argbuf.pack_accessor(dst.field.field_id, dst.transform)
            shardpt, shardfn, shardsp = rhs.find_point_sharding()
            task = Task(
                self.runtime.get_unary_task_id(
                    NumPyOpCode.NONZERO,
                    argument_type=self.dtype,
                    result_type=self.dtype,
                ),
                argbuf.get_string(),
                argbuf.get_size(),
                mapper=self.runtime.mapper_id,
                tag=shardfn,
            )
            if shardpt is not None:
                task.set_point(shardpt)
            if shardsp is not None:
                task.set_sharding_space(shardsp)
            task.add_read_requirement(
                rhs.region,
                rhs.field.field_id,
                tag=NumPyMappingTag.NO_MEMOIZE_TAG,
            )
            task.add_write_requirement(
                dst.region,
                dst.field.field_id,
                tag=NumPyMappingTag.NO_MEMOIZE_TAG,
            )
            self.runtime.dispatch(task)
        result = ()
        for idx in xrange(self.ndim):
            result += (
                dst_array.get_item(
                    (idx, slice(None, None, None)), stacklevel=(stacklevel + 1)
                ),
            )
        return result

    def sort(self, rhs, stacklevel, callsite=None):
        assert lhs_array.ndim == 1
        assert lhs_array.dtype == rhs_array.dtype
        rhs = rhs_array.get_thunk()
        lhs = lhs_array.get_thunk(check=False)
        rhs_launch_space = rhs.compute_parallel_launch_space()
        # If no index space launch this is easy just launch the single sort
        # task
        if rhs_launch_space is None:
            # See if we need to flatten the array or can just do a normal copy
            if rhs_array.ndim > 1:
                raise NotImplementedError("Need flatten copy")
            else:
                # Copy the data to the output array
                copy = Copy(mapper=self.runtime.mapper_id)
                copy.add_src_requirement(rhs.region, rhs.field.field_id)
                copy.add_dst_requirement(lhs.region, lhs.field.field_id)
                self.runtime.dispatch(copy)
            argbuf = BufferBuilder()
            self.pack_shape(argbuf, lhs_array.shape, pack_dim=False)
            arbguf.pack_accessor(lhs.field.field_id, rhs.transform)
            task = Task(
                self.runtime.get_unary_task_id(
                    NumPyOpCode.SORT,
                    result_type=lhs_array.dtype,
                    argument_type=rhs_array.dtype,
                ),
                argbuf.get_string(),
                argbuf.get_size(),
                mapper=self.runtime.mapper_id,
            )
            task.add_read_write_requirement(lhs.region, lhs.field.field_id)
            self.runtime.dispatch(task)
            self.runtime.profile_callsite(stacklevel + 1, True, callsite)
            return
        # First compute the min and the max futures for establishing our
        # buckets
        argbuf = BufferBuilder()
        rhs_part = rhs.find_or_create_partition(rhs_launch_space)
        self.pack_shape(argbuf, rhs_array.shape, rhs_part.tile_shape, 0)
        argbuf.pack_accessor(rhs.field.field_id, rhs.transform)
        task = IndexTask(
            self.runtime.get_unary_task_id(
                NumPyOpCode.MIN,
                result_type=rhs_array.dtype,
                argument_type=rhs_array.dtype,
                variant_code=NumPyVariantCode.REDUCTION,
            ),
            Rect(rhs_launch_space),
            self.runtime.empty_argmap,
            argbuf.get_string(),
            argbuf.get_size(),
            mapper=self.runtime.mapper_id,
        )
        task.add_read_requirement(
            rhs_part, rhs.field.field_id, 0, tag=NumPyMappingTag.KEG_REGION_TAG
        )
        task = IndexTask(
            self.runtime.get_unary_task_id(
                NumPyOpCode.MAX,
                result_type=rhs_array.dtype,
                argument_type=rhs_array.dtype,
                variant_code=NumPyVariantCode.REDUCTION,
            ),
            Rect(rhs_launch_space),
            self.runtime.empty_argmap,
            argbuf.get_string(),
            argbuf.get_size(),
            mapper=self.runtime.mapper_id,
        )
        task.add_read_requirement(
            rhs_part, rhs.field.field_id, 0, tag=NumPyMappingTag.KEG_REGION_TAG
        )
        raise NotImplementedError("finish sort")

    def random_uniform(self, stacklevel, callsite=None):
        lhs_array = self
        assert lhs_array.dtype.type == np.float64
        # Special case for shadow debugging
        if self.runtime.shadow_debug:
            self.shadow.random_uniform(stacklevel=(stacklevel + 1))
            self.base.attach_numpy_array(self.shadow.array.copy())
            return
        result = lhs_array.base
        launch_space = result.compute_parallel_launch_space()
        argbuf = BufferBuilder()
        argbuf.pack_32bit_uint(self.runtime.get_next_random_epoch())
        if launch_space is not None:
            (
                result_part,
                shardfn,
                shardsp,
            ) = result.find_or_create_key_partition()
            self.pack_shape(argbuf, lhs_array.shape, result_part.tile_shape, 0)
        else:
            self.pack_shape(argbuf, lhs_array.shape)
        argbuf.pack_point(self.compute_strides(lhs_array.shape))
        argbuf.pack_accessor(result.field.field_id, result.transform)
        if launch_space is not None:
            task = IndexTask(
                self.runtime.get_nullary_task_id(
                    NumPyOpCode.RAND_UNIFORM, result_type=lhs_array.dtype
                ),
                Rect(launch_space),
                self.runtime.empty_argmap,
                argbuf.get_string(),
                argbuf.get_size(),
                mapper=self.runtime.mapper_id,
                tag=shardfn,
            )
            if shardsp is not None:
                task.set_sharding_space(shardsp)
            task.add_write_requirement(
                result_part,
                result.field.field_id,
                0,
                tag=NumPyMappingTag.KEY_REGION_TAG,
            )
            self.runtime.dispatch(task)
        else:
            shardpt, shardfn, shardsp = result.find_point_sharding()
            task = Task(
                self.runtime.get_nullary_task_id(
                    NumPyOpCode.RAND_UNIFORM, result_type=lhs_array.dtype
                ),
                argbuf.get_string(),
                argbuf.get_size(),
                mapper=self.runtime.mapper_id,
                tag=shardfn,
            )
            if shardpt is not None:
                task.set_point(shardpt)
            if shardsp is not None:
                task.set_sharding_space(shardsp)
            task.add_write_requirement(result.region, result.field.field_id)
            self.runtime.dispatch(task)
        self.runtime.profile_callsite(stacklevel + 1, True, callsite)

    def random_normal(self, stacklevel, callsite=None):
        lhs_array = self
        assert lhs_array.dtype.type == np.float64
        # Special case for shadow debugging since it's hard to get data back
        if self.runtime.shadow_debug:
            self.shadow.random_normal(stacklevel=(stacklevel + 1))
            self.base.attach_numpy_array(self.shadow.array.copy())
            return
        result = lhs_array.base
        launch_space = result.compute_parallel_launch_space()
        argbuf = BufferBuilder()
        argbuf.pack_32bit_uint(self.runtime.get_next_random_epoch())
        if launch_space is not None:
            (
                result_part,
                shardfn,
                shardsp,
            ) = result.find_or_create_key_partition()
            self.pack_shape(argbuf, lhs_array.shape, result_part.tile_shape, 0)
        else:
            self.pack_shape(argbuf, lhs_array.shape)
        argbuf.pack_point(self.compute_strides(lhs_array.shape))
        argbuf.pack_accessor(result.field.field_id, result.transform)
        if launch_space is not None:
            task = IndexTask(
                self.runtime.get_nullary_task_id(
                    NumPyOpCode.RAND_NORMAL, result_type=lhs_array.dtype
                ),
                Rect(launch_space),
                self.runtime.empty_argmap,
                argbuf.get_string(),
                argbuf.get_size(),
                mapper=self.runtime.mapper_id,
                tag=shardfn,
            )
            if shardsp is not None:
                task.set_sharding_space(shardsp)
            task.add_write_requirement(
                result_part,
                result.field.field_id,
                0,
                tag=NumPyMappingTag.KEY_REGION_TAG,
            )
            self.runtime.dispatch(task)
        else:
            shardpt, shardfn, shardsp = result.find_point_sharding()
            task = Task(
                self.runtime.get_nullary_task_id(
                    NumPyOpCode.RAND_NORMAL, result_type=lhs_array.dtype
                ),
                argbuf.get_string(),
                argbuf.get_size(),
                mapper=self.runtime.mapper_id,
                tag=shardfn,
            )
            if shardpt is not None:
                task.set_point(shardpt)
            if shardsp is not None:
                task.set_sharding_space(shardsp)
            task.add_write_requirement(result.region, result.field.field_id)
            self.runtime.dispatch(task)
        self.runtime.profile_callsite(stacklevel + 1, True, callsite)

    def random_integer(self, low, high, stacklevel, callsite=None):
        lhs_array = self
        assert lhs_array.dtype.kind == "i"
        # Special case for shadow debugging since it's hard to get data back
        if self.runtime.shadow_debug:
            self.shadow.random_integer(low, high, stacklevel=(stacklevel + 1))
            self.base.attach_numpy_array(self.shadow.array.copy())
            return
        result = lhs_array.base
        launch_space = result.compute_parallel_launch_space()
        argbuf = BufferBuilder()
        argbuf.pack_32bit_uint(self.runtime.get_next_random_epoch())
        argbuf.pack_value(low, lhs_array.dtype)
        argbuf.pack_value(high, lhs_array.dtype)
        if launch_space is not None:
            (
                result_part,
                shardfn,
                shardsp,
            ) = result.find_or_create_key_partition()
            self.pack_shape(argbuf, lhs_array.shape, result_part.tile_shape, 0)
        else:
            self.pack_shape(argbuf, lhs_array.shape)
        argbuf.pack_point(self.compute_strides(lhs_array.shape))
        argbuf.pack_accessor(result.field.field_id, result.transform)
        if launch_space is not None:
            task = IndexTask(
                self.runtime.get_nullary_task_id(
                    NumPyOpCode.RAND_INTEGER, result_type=lhs_array.dtype
                ),
                Rect(launch_space),
                self.runtime.empty_argmap,
                argbuf.get_string(),
                argbuf.get_size(),
                mapper=self.runtime.mapper_id,
                tag=shardfn,
            )
            if shardsp is not None:
                task.set_sharding_space(shardsp)
            task.add_write_requirement(
                result_part,
                result.field.field_id,
                0,
                tag=NumPyMappingTag.KEY_REGION_TAG,
            )
            self.runtime.dispatch(task)
        else:
            shardpt, shardfn, shardsp = result.find_point_sharding()
            task = Task(
                self.runtime.get_nullary_task_id(
                    NumPyOpCode.RAND_INTEGER, result_type=lhs_array.dtype
                ),
                argbuf.get_string(),
                argbuf.get_size(),
                mapper=self.runtime.mapper_id,
                tag=shardfn,
            )
            if shardpt is not None:
                task.set_point(shardpt)
            if shardsp is not None:
                task.set_sharding_space(shardsp)
            task.add_write_requirement(result.region, result.field.field_id)
            self.runtime.dispatch(task)
        self.runtime.profile_callsite(stacklevel + 1, True, callsite)

    # Perform the unary operation and put the result in the array
    def unary_op(
        self, op, op_dtype, src, where, args, stacklevel, callsite=None
    ):
        lhs_array = self
        rhs_array = self.runtime.to_deferred_array(
            src, stacklevel=(stacklevel + 1)
        )
        rhs = rhs_array.base
        # If we haven't computed a parallel launch space yet for
        # the destination array and the shapes are the same then
        # we can try to reuse the same partitioning from the src array
        if lhs_array.size > 1:
            if (
                not self.base.has_parallel_launch_space()
                and lhs_array.shape == rhs_array.shape
            ):
                launch_space = rhs.compute_parallel_launch_space()
                rhs_parallel = True
            else:
                launch_space = self.base.compute_parallel_launch_space()
                rhs_parallel = False
        else:
            launch_space = None
        if rhs_array.size == 1:
            # Scalar version of the task
            assert isinstance(rhs, Future)
            task = Task(
                self.runtime.get_unary_task_id(
                    op,
                    argument_type=op_dtype,
                    result_type=lhs_array.dtype,
                    variant_code=NumPyVariantCode.SCALAR,
                ),
                mapper=self.runtime.mapper_id,
            )
            task.add_future(rhs)
            if args is not None:
                self.add_arguments(task, args)
            future = self.runtime.dispatch(task)
            # See if the output is also a scalar or not
            if lhs_array.size > 1:
                # Output is not a scalar so do a fill broadcast
                result = lhs_array.base
                if launch_space is not None:
                    (
                        lhs_part,
                        shardfn,
                        shardsp,
                    ) = result.find_or_create_key_partition()
                    fill = IndexFill(
                        lhs_part,
                        0,
                        result.region.get_root(),
                        result.field.field_id,
                        future,
                        mapper=self.runtime.mapper_id,
                        tag=shardfn,
                    )
                    if shardsp is not None:
                        fill.set_sharding_space(shardsp)
                else:
                    (
                        shardpt,
                        shardfn,
                        shardsp,
                    ) = result.find_point_sharding()
                    fill = Fill(
                        result.region,
                        result.region.get_root(),
                        result.field.field_id,
                        future,
                        mapper=self.runtime.mapper_id,
                        tag=shardfn,
                    )
                    if shardpt is not None:
                        fill.set_point(shardpt)
                    if shardsp is not None:
                        fill.set_sharding_space(shardsp)
                self.runtime.dispatch(fill)
            else:
                # Output is a scalar so we can just save the result
                lhs_array.base = future
        else:
            assert not isinstance(rhs, Future)
            # Normal/broadcast version of this operation
            result = lhs_array.base
            # Compute a transform if we need one
            (
                transform,
                offset,
                proj_id,
                mapping_tag,
            ) = self.runtime.compute_broadcast_transform(
                lhs_array.shape, rhs_array.shape
            )
            argbuf = BufferBuilder()
            assert lhs_array.ndim >= rhs_array.ndim
            if launch_space is not None:
                if rhs_parallel:
                    (
                        rhs_part,
                        shardfn,
                        shardsp,
                    ) = rhs.find_or_create_key_partition()
                    lhs_part = result.find_or_create_congruent_partition(
                        rhs_part
                    )
                    result.set_key_partition(lhs_part, shardfn, shardsp)
                else:
                    (
                        lhs_part,
                        shardfn,
                        shardsp,
                    ) = result.find_or_create_key_partition()
                self.pack_shape(
                    argbuf, lhs_array.shape, lhs_part.tile_shape, proj_id
                )
            else:
                self.pack_shape(argbuf, lhs_array.shape)
            argbuf.pack_accessor(result.field.field_id, result.transform)
            if rhs is not result:
                # Check to see if we have a transform for input region
                self.pack_transform_accessor(argbuf, rhs, transform)
            if launch_space is not None:
                # Index space launch to do this operation in parallel
                # First construct the index launch space
                result_type = lhs_array.dtype
                variant_code = (
                    NumPyVariantCode.INPLACE
                    if rhs is result
                    else NumPyVariantCode.NORMAL
                )
                task_id = self.runtime.get_unary_task_id(
                    op,
                    argument_type=op_dtype,
                    result_type=result_type,
                    variant_code=variant_code,
                )
                task = IndexTask(
                    task_id,
                    Rect(launch_space),
                    self.runtime.empty_argmap,
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
                if shardsp is not None:
                    task.set_sharding_space(shardsp)
                # lhs_part is computed above
                assert len(launch_space) == lhs_array.ndim
                # See if we're doing this in-place
                if rhs is result:
                    # We're doing this in-place so only one region requirement
                    task.add_read_write_requirement(
                        lhs_part,
                        result.field.field_id,
                        proj_id,
                        tag=mapping_tag + NumPyMappingTag.KEY_REGION_TAG,
                    )
                else:
                    # Use the identity projection region requirement
                    task.add_write_requirement(
                        lhs_part,
                        result.field.field_id,
                        proj_id,
                        tag=NumPyMappingTag.KEY_REGION_TAG,
                    )
                    if not rhs_parallel:
                        rhs_part = rhs.find_or_create_congruent_partition(
                            lhs_part, transform, offset
                        )
                    # Shapes are the same so we can use the identity
                    # projection
                    task.add_read_requirement(
                        rhs_part,
                        rhs.field.field_id,
                        proj_id,
                        tag=mapping_tag,
                    )
                if args is not None:
                    self.add_arguments(task, args)
                self.runtime.dispatch(task)
            else:
                # Single task launch since we only need one piece
                variant_code = (
                    NumPyVariantCode.INPLACE
                    if rhs is result
                    else NumPyVariantCode.NORMAL
                )
                task_id = self.runtime.get_unary_task_id(
                    op,
                    argument_type=op_dtype,
                    result_type=lhs_array.dtype,
                    variant_code=variant_code,
                )
                if rhs_parallel:
                    shardpt, shardfn, shardsp = rhs.find_point_sharding()
                else:
                    shardpt, shardfn, shardsp = result.find_point_sharding()
                task = Task(
                    task_id,
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
                if shardpt is not None:
                    task.set_point(shardpt)
                if shardsp is not None:
                    task.set_sharding_space(shardsp)
                # See if we're doing this in-place
                if rhs is result:
                    task.add_read_write_requirement(
                        result.region, result.field.field_id
                    )
                else:
                    task.add_write_requirement(
                        result.region, result.field.field_id
                    )
                    task.add_read_requirement(rhs.region, rhs.field.field_id)
                if args is not None:
                    self.add_arguments(task, args)
                self.runtime.dispatch(task)
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
        redop,
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
        # See if this is an arg reduction
        argred = op == NumPyOpCode.ARGMIN or op == NumPyOpCode.ARGMAX
        if argred:
            assert lhs_array.dtype == np.int64
        rhs = rhs_array.base
        if initial is not None:
            initial_array = np.array(initial, dtype=lhs_array.dtype)
            if initial_array.size != 1:
                raise ValueError(
                    '"initial" value for reduction must be a scalar'
                )
            initial_future = self.runtime.create_future(
                initial_array.data, initial_array.nbytes
            )
        if isinstance(rhs, Future):
            if initial is not None:
                assert rhs_array.dtype == lhs_array.dtype
                # If we had an initial value then we need to combine these
                task = Task(
                    self.runtime.get_binary_task_id(
                        op,
                        argument_type=rhs_array.dtype,
                        result_type=lhs_array.dtype,
                        variant_code=NumPyVariantCode.SCALAR,
                    ),
                    mapper=self.runtime.mapper_id,
                )
                task.add_future(rhs)
                task.add_future(initial_future)
                lhs_array.base = self.runtime.dispatch(task)
            else:
                lhs_array.base = rhs
            return
        task_dtype = rhs_array.dtype
        result_dtype = lhs_array.dtype
        # See if we are doing reduction to a point or another region
        if lhs_array.size == 1:
            assert axes is None or len(axes) == (
                rhs_array.ndim - lhs_array.ndim
            )
            # Reduction to a single value, parallelize over the reduce space
            launch_space = rhs.compute_parallel_launch_space()
            argbuf = BufferBuilder()
            if launch_space is not None:
                rhs_part, shardfn, shardsp = rhs.find_or_create_key_partition()
                self.pack_shape(
                    argbuf, rhs_array.shape, rhs_part.tile_shape, 0
                )
            else:
                self.pack_shape(argbuf, rhs_array.shape)
            argbuf.pack_accessor(rhs.field.field_id, rhs.transform)
            if launch_space is not None:
                task = IndexTask(
                    self.runtime.get_unary_task_id(
                        op,
                        result_type=result_dtype,
                        argument_type=task_dtype,
                        variant_code=NumPyVariantCode.REDUCTION,
                    ),
                    Rect(launch_space),
                    self.runtime.empty_argmap,
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
                if shardsp is not None:
                    task.set_sharding_space(shardsp)
                # rhs_part is computed above
                task.add_read_requirement(
                    rhs_part,
                    rhs.field.field_id,
                    0,
                    tag=NumPyMappingTag.KEY_REGION_TAG,
                )
                if args is not None:
                    self.add_arguments(task, args)
                result = self.runtime.dispatch(
                    task,
                    redop=self.runtime.get_reduction_op_id(
                        op, lhs_array.dtype
                    ),
                )
            else:
                shardpt, shardfn, shardsp = rhs.find_point_sharding()
                task = Task(
                    self.runtime.get_unary_task_id(
                        op,
                        argument_type=task_dtype,
                        result_type=result_dtype,
                        variant_code=NumPyVariantCode.REDUCTION,
                    ),
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
                if shardpt is not None:
                    task.set_point(shardpt)
                if shardsp is not None:
                    task.set_sharding_space(shardsp)
                task.add_read_requirement(rhs.region, rhs.field.field_id)
                if args is not None:
                    self.add_arguments(task, args)
                result = self.runtime.dispatch(task)
            # If this is an argred task, we need to convert from
            # the argred type back to the actual type of the result
            if argred:
                assert initial is None
                task = Task(
                    self.runtime.get_nullary_task_id(
                        NumPyOpCode.GETARG,
                        result_type=result_type,
                        variant_code=NumPyVariantCode.SCALAR,
                    ),
                    mapper=self.runtime.mapper_id,
                )
                task.add_future(result)
                result = self.runtime.dispatch(task)
            elif initial is not None:
                # If we had an initial value then we need to do an extra step
                # to combine that with the actual result of the
                task = Task(
                    self.runtime.get_binary_task_id(
                        op,
                        argumnet_type=lhs_array.dtype,
                        result_type=lhs_array.dtype,
                        variant_code=NumPyVariantCode.SCALAR,
                    ),
                    mapper=self.runtime.mapper_id,
                )
                task.add_future(result)
                task.add_future(initial_future)
                result = self.runtime.dispatch(task)
            lhs_array.base = result
        else:
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
                rhs_array.shape, lhs_array.shape, axes
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
                if launch_space[axis] == 1 and not argred:
                    # No temporary field needed since we can do all the
                    # reductions in each point
                    argbuf = BufferBuilder()
                    argbuf.pack_dimension(axis)
                    argbuf.pack_dimension(-1)  # No extra dimension
                    self.pack_shape(
                        argbuf,
                        lhs_array.shape,
                        result_part.tile_shape,
                        proj_id,
                    )
                    argbuf.pack_accessor(
                        result.field.field_id, result.transform
                    )
                    self.pack_shape(
                        argbuf, rhs_array.shape, rhs_part.tile_shape, 0
                    )
                    # Use the result field here
                    if result.transform is not None:
                        # Transform from the rhs space back to our space
                        to_pack = transform.compose(result.transform)
                    else:
                        to_pack = transform
                    argbuf.pack_accessor(result.field.field_id, to_pack)
                    argbuf.pack_accessor(rhs.field.field_id, rhs.transform)
                    task = IndexTask(
                        self.runtime.get_unary_task_id(
                            op,
                            argument_type=task_dtype,
                            result_type=task_dtype,
                        ),
                        Rect(launch_space),
                        self.runtime.empty_argmap,
                        argbuf.get_string(),
                        argbuf.get_size(),
                        mapper=self.runtime.mapper_id,
                        tag=shardfn,
                    )
                    if shardsp is not None:
                        task.set_sharding_space(shardsp)
                    task.add_write_requirement(
                        result_part,
                        result.field.field_id,
                        proj_id,
                        tag=NumPyMappingTag.NO_MEMOIZE_TAG,
                        flags=legion.LEGION_COMPLETE_PROJECTION_WRITE_FLAG,
                    )
                    task.add_read_requirement(
                        rhs_part,
                        rhs.field.field_id,
                        0,
                        tag=NumPyMappingTag.KEY_REGION_TAG,
                    )
                    if args is not None:
                        self.add_arguments(task, args)
                    if initial is not None:
                        task.add_future(initial_future)
                    self.runtime.dispatch(task)
                else:
                    # We need to make a temporary field for reductions
                    reduction_shape = ()
                    for idx in xrange(rhs_array.ndim):
                        if idx == axis:
                            reduction_shape += (launch_space[idx],)
                        else:
                            reduction_shape += (rhs_array.shape[idx],)
                    if argred:
                        # Argreds have a combination dtype
                        argred_dtype = np.dtype(
                            [("f1", np.int64), ("f2", rhs_array.dtype)],
                            align=True,
                        )
                        reduction_field = self.runtime.allocate_field(
                            reduction_shape, argred_dtype
                        )
                    else:
                        reduction_field = self.runtime.allocate_field(
                            reduction_shape, lhs_array.dtype
                        )
                    reduction_part = reduction_field.find_or_create_partition(
                        launch_space
                    )
                    argbuf = BufferBuilder()
                    argbuf.pack_dimension(axis)
                    argbuf.pack_dimension(
                        axis
                    )  # Extra dimension to reduce over
                    self.pack_shape(
                        argbuf,
                        lhs_array.shape,
                        result_part.tile_shape,
                        proj_id,
                    )
                    argbuf.pack_accessor(
                        reduction_field.field.field_id,
                        reduction_field.transform,
                    )
                    self.pack_shape(
                        argbuf, rhs_array.shape, rhs_part.tile_shape, 0
                    )
                    # Use the result field here
                    if reduction_field.transform is not None:
                        # Transform from the rhs space back to our space
                        to_pack = transform.compose(reduction_field.transform)
                    else:
                        to_pack = transform
                    argbuf.pack_accessor(
                        reduction_field.field.field_id, to_pack
                    )
                    argbuf.pack_accessor(rhs.field.field_id, rhs.transform)
                    task = IndexTask(
                        self.runtime.get_unary_task_id(
                            op,
                            result_type=task_dtype,
                            argument_type=task_dtype,
                        ),
                        Rect(launch_space),
                        self.runtime.empty_argmap,
                        argbuf.get_string(),
                        argbuf.get_size(),
                        mapper=self.runtime.mapper_id,
                        tag=shardfn,
                    )
                    if shardsp is not None:
                        task.set_sharding_space(shardsp)
                    task.add_write_requirement(
                        reduction_part,
                        reduction_field.field.field_id,
                        0,
                        tag=NumPyMappingTag.NO_MEMOIZE_TAG,
                        flags=legion.LEGION_COMPLETE_PROJECTION_WRITE_FLAG,
                    )
                    task.add_read_requirement(
                        rhs_part,
                        rhs.field.field_id,
                        0,
                        tag=NumPyMappingTag.KEY_REGION_TAG,
                    )
                    if args is not None:
                        self.add_arguments(task, args)
                    if initial is not None:
                        task.add_future(initial_future)
                    self.runtime.dispatch(task)
                    # Now we need to launch the reduction tree(s)
                    # Figure out which kind of projection functor to use
                    if len(lhs_array.shape) == 1:
                        # 2D -> 1D
                        assert axis == 0 or axis == 1
                        if axis == 0:
                            if keepdims:
                                result_proj = NumPyProjCode.PROJ_2D_2D_Y
                            else:
                                result_proj = NumPyProjCode.PROJ_2D_1D_Y
                        else:
                            if keepdims:
                                result_proj = NumPyProjCode.PROJ_2D_2D_X
                            else:
                                result_proj = NumPyProjCode.PROJ_2D_1D_X
                    elif len(lhs_array.shape) == 2:
                        # 3D -> 2D
                        assert axis >= 0 and axis <= 2
                        if axis == 0:
                            if keepdims:
                                result_proj = NumPyProjCode.PROJ_3D_3D_0YZ
                            else:
                                result_proj = NumPyProjCode.PROJ_3D_2D_YZ
                        elif axis == 1:
                            if keepdims:
                                result_proj = NumPyProjCode.PROJ_3D_3D_X0Z
                            else:
                                result_proj = NumPyProjCode.PROJ_3D_2D_XZ
                        else:
                            if keepdims:
                                result_proj = NumPyProjCode.PROJ_3D_3D_XY0
                            else:
                                result_proj = NumPyProjCode.PROJ_3D_2D_XY
                    else:
                        raise NotImplementedError(
                            "Need support for more dims in unary reduction"
                        )
                    result_proj += self.runtime.first_proj_id
                    if reduction_shape[axis] > 1:
                        if initial is not None:
                            # If we had a non-trivial initial value for
                            # sum or prod then we can't count initial
                            # values more than once which is not
                            # something that we're currently handling
                            if op == NumPyOpCode.SUM and initial != 0:
                                raise NotImplementedError(
                                    "Need reduction tree support with "
                                    "non-trivial initial value"
                                )
                            if op == NumPyOpCode.PROD and initial != 1:
                                raise NotImplementedError(
                                    "Need reduction tree support with "
                                    "non-trivial initial value"
                                )
                        RADIX = self.runtime.radix
                        # Figure out what the natural partitioning of the
                        # output vector is so we can do reductions to it
                        lhs_launch = result.compute_parallel_launch_space()
                        if lhs_launch is None:
                            result_part = result.find_or_create_partition(
                                (1,) * lhs_array.ndim
                            )
                            lhs_launch = ()
                            for ax in xrange(rhs_array.ndim):
                                if ax == axis:
                                    lhs_launch += (launch_space[ax],)
                                else:
                                    lhs_launch += (1,)
                            launch_space = lhs_launch
                        else:
                            result_part = result.find_or_create_partition(
                                lhs_launch
                            )
                            full_launch = ()
                            next_ax = 0
                            for ax in xrange(rhs_array.ndim):
                                if ax == axis:
                                    full_launch += (launch_space[ax],)
                                else:
                                    full_launch += (lhs_launch[next_ax],)
                                    next_ax += 1
                            launch_space = full_launch
                        sharding_space = self.runtime.find_or_create_index_space(  # noqa E501
                            launch_space
                        )
                        reduction_part = reduction_field.find_or_create_partition(  # noqa E501
                            launch_space
                        )
                        radix_generation = 1
                        while reduction_shape[axis] > 1:
                            argbuf = BufferBuilder()
                            argbuf.pack_dimension(RADIX)
                            # Make the new reduction space and the new launch
                            # space
                            new_reduction_shape = ()
                            local_launch_space = ()
                            for ax in xrange(len(reduction_shape)):
                                if ax == axis:
                                    new_launch = (
                                        reduction_shape[ax] + RADIX - 1
                                    ) // RADIX
                                    new_reduction_shape += (new_launch,)
                                    local_launch_space += (new_launch,)
                                else:
                                    new_reduction_shape += (
                                        reduction_shape[ax],
                                    )
                                    local_launch_space += (launch_space[ax],)
                            # Perform index task launches to do the reductions
                            if new_reduction_shape[axis] == 1 and not argred:
                                # If this is the last reduction we can put it
                                # right in the output array assuming we don't
                                # need a conversion
                                new_reduction_field = result
                                new_reduction_part = result_part
                                new_reduction_proj = result_proj
                                argbuf.pack_dimension(
                                    -1
                                )  # No collapse out dimension
                            else:
                                new_reduction_field = (
                                    self.runtime.allocate_field(  # noqa E501
                                        new_reduction_shape,
                                        reduction_field.field.dtype,
                                    )
                                )
                                new_reduction_part = new_reduction_field.find_or_create_partition(  # noqa E501
                                    local_launch_space
                                )
                                new_reduction_proj = 0  # identity projection
                                argbuf.pack_dimension(
                                    axis
                                )  # Collapsing out domension axis
                            argbuf.pack_dimension(
                                axis
                            )  # Collapsing in dimension axis
                            self.pack_shape(
                                argbuf,
                                lhs_array.shape,
                                result_part.tile_shape,
                                result_proj,
                            )
                            argbuf.pack_accessor(
                                new_reduction_field.field.field_id,
                                new_reduction_field.transform,
                            )
                            for idx in xrange(RADIX):
                                argbuf.pack_accessor(
                                    reduction_field.field.field_id,
                                    reduction_field.transform,
                                )
                            task = IndexTask(
                                self.runtime.get_binary_task_id(
                                    redop,
                                    result_type=task_dtype,
                                    first_argument_type=task_dtype,
                                    second_argument_type=task_dtype,
                                ),
                                Rect(local_launch_space),
                                self.runtime.empty_argmap,
                                argbuf.get_string(),
                                argbuf.get_size(),
                                mapper=self.runtime.mapper_id,
                                tag=(
                                    NumPyMappingTag.RADIX_GEN_TAG
                                    & (radix_generation << RADIX_GEN_SHIFT)
                                )
                                | (
                                    NumPyMappingTag.RADIX_DIM_TAG
                                    & (axis << RADIX_DIM_SHIFT)
                                ),
                            )
                            task.set_sharding_space(sharding_space)
                            task.add_write_requirement(
                                new_reduction_part,
                                new_reduction_field.field.field_id,
                                new_reduction_proj,
                                tag=NumPyMappingTag.KEY_REGION_TAG,
                                flags=legion.LEGION_COMPLETE_PROJECTION_WRITE_FLAG,  # noqa E501
                            )
                            for idx in xrange(RADIX):
                                task.add_read_requirement(
                                    reduction_part,
                                    reduction_field.field.field_id,
                                    self.runtime.get_radix_projection_functor_id(  # noqa E501
                                        len(local_launch_space),
                                        axis,
                                        RADIX,
                                        idx,
                                    ),
                                )
                            self.runtime.dispatch(task)
                            reduction_shape = new_reduction_shape
                            reduction_field = new_reduction_field
                            reduction_part = new_reduction_part
                            radix_generation += 1
                    # If we need a conversion back from argred, do that now
                    if argred:
                        out_space = result.compute_parallel_launch_space()
                        if out_space is None:
                            # Single task conversion
                            argbuf = BufferBuilder()
                            argbuf.pack_dimension(axis)
                            self.pack_shape(argbuf, lhs_array.shape)
                            argbuf.pack_accessor(
                                result.field.field_id, result.transform
                            )
                            argbuf.pack_accessor(
                                reduction_field.field.field_id,
                                reduction_field.transform,
                            )
                            task = Task(
                                self.runtime.get_nullary_task_id(
                                    NumPyOpCode.GETARG, result_type=task_dtype
                                ),
                                argbuf.get_string(),
                                argbuf.get_size(),
                                mapper=self.runtime.mapper_id,
                            )
                            task.add_write_requirement(
                                result.region,
                                result.field.field_id,
                                tag=NumPyMappingTag.KEY_REGION_TAG,
                            )
                            task.add_read_requirement(
                                reduction_field.region,
                                reduction_field.field.field_id,
                                tag=NumPyMappingTag.NO_MEMOIZE_TAG,
                            )
                            self.runtime.dispatch(task)
                        else:
                            # Distributed task conversion
                            local_launch_space = ()
                            for ax in xrange(len(reduction_shape)):
                                if ax == axis:
                                    local_launch_space += (1,)
                                else:
                                    local_launch_space += (launch_space[ax],)
                            argbuf = BufferBuilder()
                            argbuf.pack_dimension(axis)
                            self.pack_shape(
                                argbuf,
                                lhs_array.shape,
                                result_part.tile_shape,
                                result_proj,
                            )
                            argbuf.pack_accessor(
                                result.field.field_id, result.transform
                            )
                            argbuf.pack_accessor(
                                reduction_field.field.field_id,
                                reduction_field.transform,
                            )
                            task = IndexTask(
                                self.runtime.get_nullary_task_id(
                                    NumPyOpCode.GETARG, result_type=task_dtype
                                ),
                                Rect(local_launch_space),
                                self.runtime.empty_argmap,
                                argbuf.get_string(),
                                argbuf.get_size(),
                                mapper=self.runtime.mapper_id,
                            )
                            task.add_write_requirement(
                                result_part,
                                result.field.field_id,
                                result_proj,
                                tag=NumPyMappingTag.KEY_REGION_TAG,
                            )
                            task.add_read_requirement(
                                reduction_part,
                                reduction_field.field.field_id,
                                0,
                                tag=NumPyMappingTag.NO_MEMOIZE_TAG,
                            )
                            self.runtime.dispatch(task)
            else:
                # Single task launch case
                argbuf = BufferBuilder()
                argbuf.pack_dimension(axis)
                argbuf.pack_dimension(-1)  # We're not reducing any dimensions
                if argred:
                    # Need a temporary field for the argred case
                    # Argreds have a combination dtype
                    argred_dtype = np.dtype(
                        [("f1", np.int64), ("f2", rhs_array.dtype)], align=True
                    )
                    temp = self.runtime.allocate_field(
                        lhs_array.shape, argred_dtype
                    )
                    if temp.transform:
                        # Transform from the rhs space back to our space
                        affine_transform = affine_transform.compose(
                            temp.transform
                        )
                    # Access lhs for initialization with its shape
                    self.pack_shape(argbuf, lhs_array.shape)
                    argbuf.pack_accessor(temp.field.field_id, temp.transform)
                    # Access lhs for reducing to rhs with the transformed shape
                    self.pack_shape(argbuf, rhs_array.shape)
                    argbuf.pack_accessor(temp.field.field_id, affine_transform)
                else:
                    if result.transform:
                        # Transform from the rhs space back to our space
                        affine_transform = affine_transform.compose(
                            result.transform
                        )
                    # Access lhs for initialization with its shape
                    self.pack_shape(argbuf, lhs_array.shape)
                    argbuf.pack_accessor(
                        result.field.field_id, result.transform
                    )
                    # Access lhs for reducing to rhs with the transformed shape
                    self.pack_shape(argbuf, rhs_array.shape)
                    argbuf.pack_accessor(result.field.field_id, transform)
                # Access rhs with the shape that was packed earlier for lhs
                argbuf.pack_accessor(rhs.field.field_id, rhs.transform)
                shardpt, shardfn, shardsp = rhs.find_point_sharding()
                task = Task(
                    self.runtime.get_unary_task_id(
                        op, argument_type=task_dtype, result_type=task_dtype
                    ),
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
                if shardpt is not None:
                    task.set_point(shardpt)
                if shardsp is not None:
                    task.set_sharding_space(shardsp)
                if argred:
                    task.add_write_requirement(
                        temp.region,
                        temp.field.field_id,
                        tag=NumPyMappingTag.NO_MEMOIZE_TAG,
                    )
                else:
                    task.add_write_requirement(
                        result.region,
                        result.field.field_id,
                        tag=NumPyMappingTag.NO_MEMOIZE_TAG,
                    )
                task.add_read_requirement(
                    rhs.region,
                    rhs.field.field_id,
                    tag=NumPyMappingTag.KEY_REGION_TAG,
                )
                if args is not None:
                    self.add_arguments(task, args)
                if initial is not None:
                    task.add_future(initial_future)
                self.runtime.dispatch(task)
                # if this isn't an argred then we're done, otherwise do the
                # conversion
                if argred:
                    argbuf = BufferBuilder()
                    argbuf.pack_dimension(-1)  # No collapse dimension
                    self.pack_shape(argbuf, lhs_array.shape)
                    argbuf.pack_accessor(
                        result.field.field_id, result.transform
                    )
                    argbuf.pack_accessor(temp.field.field_id, temp.transform)
                    task = Task(
                        self.runtime.get_nullary_task_id(
                            NumPyOpCode.GETARG, result_type=task_dtype
                        ),
                        argbuf.get_string(),
                        argbuf.get_size(),
                        mapper=self.runtime.mapper_id,
                        tag=shardfn,
                    )
                    if shardpt is not None:
                        task.set_point(shardpt)
                    if shardsp is not None:
                        task.set_sharding_space(shardsp)
                    task.add_write_requirement(
                        result.region,
                        result.field.field_id,
                        tag=NumPyMappingTag.KEY_REGION_TAG,
                    )
                    task.add_read_requirement(
                        temp.region,
                        temp.field.field_id,
                        tag=NumPyMappingTag.NO_MEMOIZE_TAG,
                    )
                    self.runtime.dispatch(task)
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

        lhs_arg = DeferredArrayView(lhs_array)
        rhs1_arg = DeferredArrayView(rhs1_array)
        rhs2_arg = DeferredArrayView(rhs2_array)

        # Align and broadcast region arguments if necessary
        launch_space, key_arg = lhs_arg.find_key_view(rhs1_arg, rhs2_arg)
        key_arg.update_tag(NumPyMappingTag.KEY_REGION_TAG)
        rhs1_arg, offset1 = rhs1_arg.broadcast(lhs_arg)
        rhs2_arg, offset2 = rhs2_arg.broadcast(lhs_arg)

        if launch_space is not None:
            lhs_arg = lhs_arg.align_partition(key_arg)
            rhs1_arg = rhs1_arg.align_partition(key_arg, offset1)
            rhs2_arg = rhs2_arg.align_partition(key_arg, offset2)
            if lhs_arg is not key_arg:
                lhs_arg.copy_key_partition_from(key_arg)

        # Populate the Legate launcher
        any_scalar_rhs = rhs1_arg.scalar or rhs2_arg.scalar
        all_scalar_rhs = rhs1_arg.scalar and rhs2_arg.scalar

        if all_scalar_rhs:
            task_id = NumPyOpCode.SCALAR_BINARY_OP
        elif any_scalar_rhs:
            task_id = NumPyOpCode.BROADCAST_BINARY_OP
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

        if not lhs_arg.scalar:
            lhs_arg.add_to_legate_op(op, False)
        rhs1_arg.add_to_legate_op(op, True)
        rhs2_arg.add_to_legate_op(op, True)

        if any_scalar_rhs:
            op.add_scalar_arg(rhs2_arg.scalar, bool)

        if args is not None:
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
        rhs1 = rhs1_array.base
        rhs2 = rhs2_array.base
        # Should have been handled by type checkking on the front end
        assert rhs1_array.dtype == rhs2_array.dtype
        common_dtype = rhs1_array.dtype
        if rhs1_array.size == 1 and rhs2_array.size == 1:
            # Scalar case
            assert isinstance(rhs1, Future) and isinstance(rhs2, Future)
            task = Task(
                self.runtime.get_binary_task_id(
                    op,
                    result_type=lhs_array.dtype,
                    first_argument_type=common_dtype,
                    second_argument_type=common_dtype,
                    variant_code=NumPyVariantCode.SCALAR,
                ),
                mapper=self.runtime.mapper_id,
            )
            task.add_future(rhs1)
            task.add_future(rhs2)
            if args is not None:
                self.add_arguments(task, args)
            future = self.runtime.dispatch(task)
            lhs_array.base = future
        # Reduction to a single value, see if we have a broadcast
        elif broadcast is not None:
            # Scalar should have been handled above
            assert not isinstance(rhs1, Future) or not isinstance(rhs2, Future)
            has_future = isinstance(rhs1, Future) or isinstance(rhs2, Future)
            # Compute our transforms
            (
                transform1,
                offset1,
                proj1_id,
                mapping_tag1,
            ) = self.runtime.compute_broadcast_transform(
                lhs_array.shape, rhs1_array.shape
            )
            (
                transform2,
                offset2,
                proj2_id,
                mapping_tag2,
            ) = self.runtime.compute_broadcast_transform(
                lhs_array.shape, rhs2_array.shape
            )
            # Compute our launch space
            launch_space = self.runtime.compute_parallel_launch_space_by_shape(
                broadcast
            )
            argbuf = BufferBuilder()
            if launch_space is not None:
                tile_shape = self.runtime.compute_tile_shape(
                    broadcast, launch_space
                )
                self.pack_shape(argbuf, broadcast, tile_shape, 0)
            else:
                self.pack_shape(argbuf, broadcast)
            if not isinstance(rhs1, Future):
                self.pack_transform_accessor(argbuf, rhs1, transform1)
            if not isinstance(rhs2, Future):
                self.pack_transform_accessor(argbuf, rhs2, transform2)
            if has_future:  # Pack the index of the future information
                if isinstance(rhs1, Future):
                    argbuf.pack_32bit_uint(0)
                else:
                    argbuf.pack_32bit_uint(1)
            if launch_space is not None:
                # Index task launch case
                task_variant = (
                    NumPyVariantCode.BROADCAST_REDUCTION
                    if has_future
                    else NumPyVariantCode.REDUCTION
                )
                task = IndexTask(
                    self.runtime.get_binary_task_id(
                        op,
                        result_type=lhs_array.dtype,
                        first_argument_type=common_dtype,
                        second_argument_type=common_dtype,
                        variant_code=task_variant,
                    ),
                    Rect(launch_space),
                    self.runtime.empty_argmap,
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                )
                if isinstance(rhs1, Future):
                    task.add_future(rhs1)
                elif transform1 is not None:
                    shape_transform1 = AffineTransform(
                        rhs1_array.ndim, len(broadcast), False
                    )
                    shape_transform1.trans = transform1
                    shape_transform1.offset = offset1
                    rhs1_shape = shape_transform1.apply(launch_space)
                    rhs1_part = rhs1.find_or_create_partition(rhs1_shape)
                else:
                    rhs1_part = rhs1.find_or_create_partition(launch_space)
                task.add_read_requirement(
                    rhs1_part,
                    rhs1.field.field_id,
                    proj1_id,
                    tag=mapping_tag1,
                )
                if isinstance(rhs2, Future):
                    task.add_future(rhs2)
                elif transform2 is not None:
                    shape_transform2 = Transform(
                        rhs2_array.ndim, len(broadcast), False
                    )
                    shape_transform2.trans = transform2
                    shape_transform2.offset = offset2
                    rhs2_shape = shape_transform2.apply(launch_space)
                    rhs2_part = rhs2.find_or_create_partition(rhs2_shape)
                else:
                    rhs2_part = rhs2.find_or_create_partition(launch_space)
                task.add_read_requirement(
                    rhs2_part,
                    rhs2.field.field_id,
                    proj2_id,
                    tag=mapping_tag2,
                )
                if args is not None:
                    self.add_arguments(task, args)
                redop = self.runtime.get_reduction_op_id(op, lhs_array.dtype)
                result = self.runtime.dispatch(task, redop)
            else:
                # Single task launch case
                task_variant = (
                    NumPyVariantCode.BROADCAST_REDUCTION
                    if has_future
                    else NumPyVariantCode.REDUCTION
                )
                task = Task(
                    self.runtime.get_binary_task_id(
                        op,
                        result_type=lhs_array.dtype,
                        first_argument_type=common_dtype,
                        second_argument_type=common_dtype,
                        variant_code=task_variant,
                    ),
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                )
                if isinstance(rhs1, Future):
                    task.add_future(rhs1)
                else:
                    task.add_read_requirement(rhs1.region, rhs1.field.field_id)
                if isinstance(rhs2, Future):
                    task.add_future(rhs2)
                else:
                    task.add_read_requirement(rhs2.region, rhs2.field.field_id)
                if args is not None:
                    self.add_arguments(task, args)
                result = self.runtime.dispatch(task)
            lhs_array.base = result
        else:
            # Non-broadcast reduction to a single value
            # No transforms should be necessary here
            assert rhs1_array.shape == rhs2_array.shape
            # Scalar should have been handled above
            assert not isinstance(rhs1, Future) or not isinstance(rhs2, Future)
            has_future = isinstance(rhs1, Future) or isinstance(rhs2, Future)
            launch_space = (
                rhs1.compute_parallel_launch_space()
                if not isinstance(rhs1, Future)
                else rhs2.compute_parallel_launch_space()
            )
            argbuf = BufferBuilder()
            if launch_space is not None:
                if rhs1_array.size > 1:
                    rhs1_part = rhs1.find_or_create_partition(launch_space)
                    self.pack_shape(
                        argbuf, rhs1_array.shape, rhs1_part.tile_shape, 0
                    )
                else:
                    rhs2_part = rhs2.find_or_create_partition(launch_space)
                    self.pack_shape(
                        argbuf, rhs2_array.shape, rhs2_part.tile_shape, 0
                    )
            else:
                if rhs1_array.size > 1:
                    self.pack_shape(argbuf, rhs1_array.shape)
                else:
                    self.pack_shape(argbuf, rhs2_array.shape)
            argbuf.pack_accessor(rhs1.field.field_id, rhs1.transform)
            argbuf.pack_accessor(rhs2.field.field_id, rhs2.transform)
            if launch_space is not None:
                task = IndexTask(
                    self.runtime.get_binary_task_id(
                        op,
                        result_type=lhs_array.dtype,
                        first_argument_type=common_dtype,
                        second_argument_type=common_dtype,
                        variant_code=NumPyVariantCode.REDUCTION,
                    ),
                    Rect(launch_space),
                    self.runtime.empty_argmap,
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                )
                if isinstance(rhs1, Future):
                    task.add_future(rhs1)
                else:
                    rhs1_part = rhs1.find_or_create_partition(launch_space)
                    task.add_read_requirement(
                        rhs1_part,
                        rhs1.field.field_id,
                        0,
                        tag=NumPyMappingTag.KEY_REGION_TAG,
                    )
                if isinstance(rhs2, Future):
                    task.add_future(rhs2)
                else:
                    rhs2_part = rhs2.find_or_create_partition(launch_space)
                    task.add_read_requirement(
                        rhs2_part,
                        rhs2.field.field_id,
                        0,
                        tag=NumPyMappingTag.KEY_REGION_TAG,
                    )
                if args is not None:
                    self.add_arguments(task, args)
                redop = self.runtime.get_reduction_op_id(op, lhs_array.dtype)
                result = self.runtime.dispatch(task, redop)
            else:
                task_id = self.runtime.get_binary_task_id(
                    op,
                    result_type=lhs_array.dtype,
                    first_argument_type=common_dtype,
                    second_argument_type=common_dtype,
                    variant_code=NumPyVariantCode.REDUCTION,
                )
                task = Task(
                    task_id,
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                )
                task.add_read_requirement(rhs1.region, rhs1.field.field_id)
                task.add_read_requirement(rhs2.region, rhs2.field.field_id)
                if args is not None:
                    self.add_arguments(task, args)
                result = self.runtime.dispatch(task)
            array = np.frombuffer(result.get_buffer(1), dtype=bool, count=1)
            scalar = self.runtime.create_scalar(array.data, np.dtype("bool"))
            lhs_array.base = scalar
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

    def ternary_op(
        self, op, src1, src2, src3, where, args, stacklevel, callsite=None
    ):
        if where is not True:
            # Haven't seen any cases where this is needed yet
            raise NotImplementedError(
                "need support for non-trivial where for ternary op"
            )
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
        rhs1 = rhs1_array.base
        rhs2 = rhs2_array.base
        rhs3 = rhs3_array.base
        if lhs_array.size == 1:
            task = Task(
                self.runtime.get_ternary_task_id(
                    op, lhs_array.dtype, scalar=True
                ),
                mapper=self.runtime.mapper_id,
            )
            if args is not None:
                self.add_arguments(task, args)
            task.add_future(rhs1)
            task.add_future(rhs2)
            task.add_future(rhs3)
            result = self.runtime.dispatch(task)
            lhs_array.base = result
        else:
            # Normal/broadcast version of this task
            result = lhs_array.base
            # Compute our transforms
            (
                transform1,
                offset1,
                proj1_id,
                mapping_tag1,
            ) = self.runtime.compute_broadcast_transform(
                lhs_array.shape, rhs1_array.shape
            )
            (
                transform2,
                offset2,
                proj2_id,
                mapping_tag2,
            ) = self.runtime.compute_broadcast_transform(
                lhs_array.shape, rhs2_array.shape
            )
            (
                transform3,
                offset3,
                proj3_id,
                mapping_tag3,
            ) = self.runtime.compute_broadcast_transform(
                lhs_array.shape, rhs3_array.shape
            )
            # Compute our launch space
            launch_space = result.compute_parallel_launch_space()
            # Scalar should have been handled above
            assert (
                not isinstance(rhs1, Future)
                or not isinstance(rhs2, Future)
                or not isinstance(rhs3, Future)
            )
            has_future = (
                isinstance(rhs1, Future)
                or isinstance(rhs2, Future)
                or isinstance(rhs3, Future)
            )
            argbuf = BufferBuilder()
            if launch_space is not None:
                (
                    result_part,
                    shardfn,
                    shardsp,
                ) = result.find_or_create_key_partition()
                self.pack_shape(
                    argbuf, lhs_array.shape, result_part.tile_shape, 0
                )
            else:
                self.pack_shape(argbuf, lhs_array.shape)
            argbuf.pack_accessor(result.field.field_id, result.transform)
            if not isinstance(rhs1, Future):
                if has_future:
                    argbuf.pack_bool(False)  # Not a future
                self.pack_transform_accessor(argbuf, rhs1, transform1)
            else:
                argbuf.pack_bool(True)  # Is a future
            if not isinstance(rhs2, Future):
                if has_future:
                    argbuf.pack_bool(False)  # Not a future
                self.pack_transform_accessor(argbuf, rhs2, transform2)
            else:
                argbuf.pack_bool(True)  # Is a future
            if not isinstance(rhs3, Future):
                if has_future:
                    argbuf.pack_bool(False)
                self.pack_transform_accessor(argbuf, rhs3, transform3)
            else:
                argbuf.pack_bool(True)  # Is a future
            if launch_space is not None:
                # Index task launch case
                # Helper method for adding read requirements
                def add_read_requirement(
                    task, array, region, proj, trans, offset, tag
                ):
                    if isinstance(region, Future):
                        task.add_future(region)
                    else:
                        part = region.find_or_create_congruent_partition(
                            result_part, trans, offset
                        )
                        task.add_read_requirement(
                            part, region.field.field_id, proj, tag=tag
                        )

                task = IndexTask(
                    self.runtime.get_ternary_task_id(
                        op,
                        lhs_array.dtype,
                        variant_code=NumPyVariantCode.BROADCAST
                        if has_future
                        else NumPyVariantCode.NORMAL,
                    ),
                    Rect(launch_space),
                    self.runtime.empty_argmap,
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
                if shardsp is not None:
                    task.set_sharding_space(shardsp)
                # result_part is computed above
                assert len(launch_space) == lhs_array.ndim
                task.add_write_requirement(
                    result_part,
                    result.field.field_id,
                    0,
                    tag=NumPyMappingTag.KEY_REGION_TAG,
                )
                add_read_requirement(
                    task,
                    rhs1_array,
                    rhs1,
                    proj1_id,
                    transform1,
                    offset1,
                    mapping_tag1,
                )
                add_read_requirement(
                    task,
                    rhs2_array,
                    rhs2,
                    proj2_id,
                    transform2,
                    offset2,
                    mapping_tag2,
                )
                add_read_requirement(
                    task,
                    rhs3_array,
                    rhs3,
                    proj3_id,
                    transform3,
                    offset3,
                    mapping_tag3,
                )
                if args is not None:
                    self.add_arguments(task, args)
                self.runtime.dispatch(task)
            else:
                # Single task launch case
                # Helper method for adding read requirements
                def add_read_requirement(task, region, tag):
                    if isinstance(region, Future):
                        task.add_future(region)
                    else:
                        task.add_read_requirement(
                            region.region, region.field.field_id, tag=tag
                        )

                shardpt, shardfn, shardsp = result.find_point_sharding()
                task = Task(
                    self.runtime.get_ternary_task_id(
                        op,
                        lhs_array.dtype,
                        variant_code=NumPyVariantCode.BROADCAST
                        if has_future
                        else NumPyVariantCode.NORMAL,
                    ),
                    argbuf.get_string(),
                    argbuf.get_size(),
                    mapper=self.runtime.mapper_id,
                    tag=shardfn,
                )
                if shardpt is not None:
                    task.set_point(shardpt)
                if shardsp is not None:
                    task.set_sharding_space(shardsp)
                task.add_write_requirement(
                    result.region,
                    result.field.field_id,
                    tag=NumPyMappingTag.KEY_REGION_TAG,
                )
                add_read_requirement(task, rhs1, mapping_tag1)
                add_read_requirement(task, rhs2, mapping_tag2)
                add_read_requirement(task, rhs3, mapping_tag3)
                if args is not None:
                    self.add_arguments(task, args)
                self.runtime.dispatch(task)
        self.runtime.profile_callsite(stacklevel + 1, True, callsite)
        if self.runtime.shadow_debug:
            self.shadow.ternary_op(
                op,
                src1.shadow,
                src2.shadow,
                src3.shadow,
                where if not isinstance(where, NumPyThunk) else where.shadow,
                args,
                stacklevel=(stacklevel + 1),
            )
            self.runtime.check_shadow(self, op)

    # A helper method for attaching arguments
    def add_arguments(self, op, args):
        assert args is not None
        for numpy_array in args:
            assert numpy_array.size == 1
            future = self.runtime.create_future(
                numpy_array.data, numpy_array.nbytes
            )
            op.add_future(future)

    # A helper method for support for 16 bit arithmetic
    def convert_float32_to_float16(self, result, lhs_array, collapse_dim=None):
        dst = lhs_array.base
        launch_space = dst.compute_parallel_launch_space()
        argbuf = BufferBuilder()
        if launch_space is not None:
            dst_part, shardfn, shardsp = dst.find_or_create_key_partition()
            self.pack_shape(argbuf, lhs_array.shape, dst_part.tile_shape, 0)
        else:
            self.pack_shape(argbuf, lhs_array.shape)
        argbuf.pack_accessor(dst.field.field_id, dst.transform)
        if lhs_array.ndim < len(result.shape):
            assert collapse_dim is not None
            assert lhs_array.ndim + 1 == len(result.shape)
            # Construct a transform to use for indexing the additional
            # dimension
            collapse_transform = AffineTransform(
                len(result.shape), lhs_array.ndim, eye=False
            )
            d2 = 0
            for d1 in xrange(len(result.shape)):
                if d1 != collapse_dim:
                    collapse_transform.trans[d1, d2] = 1
                    d2 += 1
            if result.transform is not None:
                argbuf.pack_accessor(
                    result.field.field_id,
                    collapse_transform.compose(result.transform),
                )
            else:
                argbuf.pack_accessor(result.field.field_id, collapse_transform)
        else:
            assert collapse_dim is None
            assert lhs_array.ndim == len(result.shape)
            # Common case when they are the same size
            argbuf.pack_accessor(result.field.field_id, result.transform)
        if launch_space is not None:
            task = IndexTask(
                self.runtime.get_unary_task_id(
                    NumPyOpCode.CONVERT,
                    result_type=lhs_array.dtype,
                    argument_type=np.dtype(np.float32),
                ),
                Rect(launch_space),
                self.runtime.empty_argmap,
                argbuf.get_string(),
                argbuf.get_size(),
                mapper=self.runtime.mapper_id,
                tag=shardfn,
            )
            if shardsp is not None:
                task.set_sharding_space(shardsp)
            task.add_write_requirement(
                dst_part,
                dst.field.field_id,
                0,
                tag=NumPyMappingTag.KEY_REGION_TAG,
            )
            if lhs_array.ndim < len(result.shape):
                # Make a partition with the right number of subregions
                part_shape = ()
                tile_shape = ()
                offsets = ()
                d2 = 0
                for d1 in xrange(len(result.shape)):
                    if d1 != collapse_dim:
                        part_shape += (launch_space[d2],)
                        tile_shape += (dst_part.tile_shape[d2],)
                        offsets += (dst_part.tile_offset[d2],)
                        d2 += 1
                    else:
                        part_shape += (result.shape[d1],)
                        tile_shape += (1,)
                        offsets += (0,)
                result_part = result.find_or_create_partition(
                    part_shape, tile_shape, offsets
                )
                if len(result.shape) == 2:
                    if collapse_dim == 0:
                        result_proj = NumPyProdCode.PROJ_1D_2D_Y
                    else:
                        assert collapse_dim == 1
                        result_proj = NumPyProjCode.PROJ_1D_2D_X
                elif len(result.shape) == 3:
                    if collapse_dim == 0:
                        result_proj = NumPyProjCode.PROJ_2D_3D_YZ
                    elif collapse_dim == 1:
                        result_proj = NumPyProjCode.PROJ_2D_3D_XZ
                    else:
                        assert collapse_dim == 2
                        result_proj = NumPyProjCode.PROJ_2D_3D_XY
                else:
                    raise NotImplementedError(
                        "Need support for 16-bit convert with more dims"
                    )
                result_proj += self.runtime.first_proj_id
                task.add_read_requirement(
                    result_part, result.field.field_id, result_proj
                )
            else:
                result_part = result.find_or_create_congruent_partition(
                    dst_part
                )
                task.add_read_requirement(
                    result_part, result.field.field_id, 0
                )
            self.runtime.dispatch(task)
        else:
            shardpt, shardfn, shardsp = dst.find_point_sharding()
            task = Task(
                self.runtime.get_unary_task_id(
                    NumPyOpCode.CONVERT,
                    result_type=lhs_array.dtype,
                    argument_type=np.dtype(np.float32),
                ),
                argbuf.get_string(),
                argbuf.get_size(),
                mapper=self.runtime.mapper_id,
                tag=shardfn,
            )
            if shardpt is not None:
                task.set_point(shardpt)
            if shardsp is not None:
                task.set_sharding_space(shardsp)
            task.add_write_requirement(dst.region, dst.field.field_id)
            if lhs_array.ndim < len(result.shape):
                # TODO get the right subregion to use
                raise NotImplementedError("Get subview for 16 bit conversion")
            else:
                task.add_read_requirement(result.region, result.field.field_id)
            self.runtime.dispatch(task)

    def pack_shape(
        self, buf, shape, chunk_shape=None, proj=None, pack_dim=True
    ):
        dim = len(shape)
        if pack_dim:
            buf.pack_dimension(dim)
        buf.pack_point(shape)
        if chunk_shape is not None:
            assert proj is not None
            if proj != 0:
                buf.pack_32bit_int(proj - self.runtime.first_proj_id)
            else:
                buf.pack_32bit_int(proj)
            assert len(chunk_shape) == dim
            buf.pack_point(chunk_shape)
        else:
            assert proj is None
            buf.pack_32bit_int(-1)

    @staticmethod
    def compute_strides(shape):
        stride = 1
        result = ()
        for dim in reversed(shape):
            result = (stride,) + result
            stride *= dim
        return result

    @staticmethod
    def pack_transform_accessor(argbuf, region, transform):
        if transform is not None:
            if region.transform:
                transform = transform.compose(region.transform)
            argbuf.pack_accessor(region.field.field_id, transform)
        else:
            argbuf.pack_accessor(region.field.field_id, region.transform)
