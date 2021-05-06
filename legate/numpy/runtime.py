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

import functools
import gc
import inspect
import math
import struct
import sys
import weakref
from collections import OrderedDict, deque

import numpy as np

from legate.core import (
    LEGATE_MAX_DIM,
    LEGATE_MAX_FIELDS,
    AffineTransform,
    Attach,
    Detach,
    FieldID,
    FieldSpace,
    Future,
    FutureMap,
    IndexPartition,
    IndexSpace,
    InlineMapping,
    PartitionByRestriction,
    Point,
    Rect,
    Region,
    Transform,
    ffi,
    get_legion_context,
    get_legion_runtime,
    legate_add_attachment,
    legate_find_attachment,
    legate_remove_attachment,
    legion,
)

from .config import *  # noqa F403
from .deferred import DeferredArray
from .eager import EagerArray
from .lazy import LazyArray
from .thunk import NumPyThunk
from .utils import calculate_volume


# Helper method for python 3 support
def _iterkeys(obj):
    return obj.keys() if sys.version_info > (3,) else obj.viewkeys()


def _iteritems(obj):
    return obj.items() if sys.version_info > (3,) else obj.viewitems()


def _itervalues(obj):
    return obj.values() if sys.version_info > (3,) else obj.viewvalues()


try:
    reduce  # Python 2
except NameError:
    reduce = functools.reduce

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

try:
    long  # Python 2
except NameError:
    long = int  # Python 3


# A Field holds a reference to a field in a region tree
# that can be used by many different RegionField objects
class Field(object):
    __slots__ = [
        "runtime",
        "region",
        "field_id",
        "dtype",
        "shape",
        "partition",
        "own",
    ]

    def __init__(self, runtime, region, field_id, dtype, shape, own=True):
        self.runtime = runtime
        self.region = region
        self.field_id = field_id
        self.dtype = dtype
        self.shape = shape
        self.partition = None
        self.own = own

    def __del__(self):
        if self.own:
            # Return our field back to the runtime
            self.runtime.free_field(
                self.region,
                self.field_id,
                self.dtype,
                self.shape,
                self.partition,
            )


_sizeof_int = ffi.sizeof("int")
_sizeof_size_t = ffi.sizeof("size_t")
assert _sizeof_size_t == 4 or _sizeof_size_t == 8


# A helper class for doing field management with control replication
class FieldMatch(object):
    __slots__ = ["manager", "fields", "input", "output", "future"]

    def __init__(self, manager, fields):
        self.manager = manager
        self.fields = fields
        # Allocate arrays of ints that are twice as long as fields since
        # our values will be 'field_id,tree_id' pairs
        if len(fields) > 0:
            alloc_string = "int[" + str(2 * len(fields)) + "]"
            self.input = ffi.new(alloc_string)
            self.output = ffi.new(alloc_string)
            # Fill in the input buffer with our data
            for idx in xrange(len(fields)):
                region, field_id = fields[idx]
                self.input[2 * idx] = region.handle.tree_id
                self.input[2 * idx + 1] = field_id
        else:
            self.input = ffi.NULL
            self.output = ffi.NULL
        self.future = None

    def launch(self, runtime, context):
        assert self.future is None
        self.future = Future(
            legion.legion_context_consensus_match(
                runtime,
                context,
                self.input,
                self.output,
                len(self.fields),
                2 * _sizeof_int,
            )
        )
        return self.future

    def update_free_fields(self):
        # If we know there are no fields then we can be done early
        if len(self.fields) == 0:
            return
        # Wait for the future to be ready
        if not self.future.is_ready():
            self.future.wait()
        # Get the size of the buffer in the returned
        if _sizeof_size_t == 4:
            num_fields = struct.unpack_from("I", self.future.get_buffer(4))[0]
        else:
            num_fields = struct.unpack_from("Q", self.future.get_buffer(8))[0]
        assert num_fields <= len(self.fields)
        if num_fields > 0:
            # Put all the returned fields onto the ordered queue in the order
            # that they are in this list since we know
            ordered_fields = [None] * num_fields
            for region, field_id in self.fields:
                found = False
                for idx in xrange(num_fields):
                    if self.output[2 * idx] != region.handle.tree_id:
                        continue
                    if self.output[2 * idx + 1] != field_id:
                        continue
                    assert ordered_fields[idx] is None
                    ordered_fields[idx] = (region, field_id)
                    found = True
                    break
                if not found:
                    # Not found so put it back int the unordered queue
                    self.manager.free_field(region, field_id, ordered=False)
            # Notice that we do this in the order of the list which is the
            # same order as they were in the array returned by the match
            for region, field_id in ordered_fields:
                self.manager.free_field(region, field_id, ordered=True)
        else:
            # No fields on all shards so put all our fields back into
            # the unorered queue
            for region, field_id in self.fields:
                self.manager.free_field(region, field_id, ordered=False)


# This class manages the allocation and reuse of fields
class FieldManager(object):
    __slots__ = [
        "runtime",
        "shape",
        "dtype",
        "free_fields",
        "freed_fields",
        "matches",
        "match_counter",
        "match_frequency",
        "top_regions",
        "initial_future",
        "fill_space",
        "tile_shape",
    ]

    def __init__(self, runtime, shape, dtype):
        self.runtime = runtime
        self.shape = shape
        self.dtype = dtype
        # This is a sanitized list of (region,field_id) pairs that is
        # guaranteed to be ordered across all the shards even with
        # control replication
        self.free_fields = deque()
        # This is an unsanitized list of (region,field_id) pairs which is not
        # guaranteed to be ordered across all shards with control replication
        self.freed_fields = list()
        # A list of match operations that have been issued and for which
        # we are waiting for values to come back
        self.matches = deque()
        self.match_counter = 0
        # Figure out how big our match frequency is based on our size
        volume = reduce(lambda x, y: x * y, self.shape)
        size = volume * self.dtype.itemsize
        if size > runtime.max_field_reuse_size:
            # Figure out the ratio our size to the max reuse size (round up)
            ratio = (
                size + runtime.max_field_reuse_size - 1
            ) // runtime.max_field_reuse_size
            assert ratio >= 1
            # Scale the frequency by the ratio, but make it at least 1
            self.match_frequency = (
                runtime.max_field_reuse_frequency + ratio - 1
            ) // ratio
        else:
            self.match_frequency = runtime.max_field_reuse_frequency
        self.top_regions = list()  # list of top-level regions with this shape
        self.initial_future = None
        self.fill_space = None
        self.tile_shape = None

    def destroy(self):
        while self.top_regions:
            region = self.top_regions.pop()
            region.destroy()
        self.free_fields = None
        self.freed_fields = None
        self.initial_future = None
        self.fill_space = None

    def allocate_field(self):
        # Increment our match counter
        self.match_counter += 1
        # If the match counter equals our match frequency then do an exchange
        if self.match_counter == self.match_frequency:
            # This is where the rubber meets the road between control
            # replication and garbage collection. We need to see if there
            # are any freed fields that are shared across all the shards.
            # We have to test this deterministically no matter what even
            # if we don't have any fields to offer ourselves because this
            # is a collective with other shards. If we have any we can use
            # the first one and put the remainder on our free fields list
            # so that we can reuse them later. If there aren't any then
            # all the shards will go allocate a new field.
            local_freed_fields = self.freed_fields
            # The match now owns our freed fields so make a new list
            # Have to do this before dispatching the match
            self.freed_fields = list()
            match = FieldMatch(self, local_freed_fields)
            # Dispatch the match
            self.runtime.dispatch(match)
            # Put it on the deque of outstanding matches
            self.matches.append(match)
            # Reset the match counter back to 0
            self.match_counter = 0
        # First, if we have a free field then we know everyone has one of those
        if len(self.free_fields) > 0:
            return self.free_fields.popleft()
        # If we don't have any free fields then see if we have a pending match
        # outstanding that we can now add to our free fields and use
        while len(self.matches) > 0:
            match = self.matches.popleft()
            match.update_free_fields()
            # Check again to see if we have any free fields
            if len(self.free_fields) > 0:
                return self.free_fields.popleft()
        # Still don't have a field
        # Scan through looking for a free field of the right type
        for reg in self.top_regions:
            # Check to see if we've maxed out the fields for this region
            # Note that this next block ensures that we go
            # through all the fields in a region before reusing
            # any of them. This is important for avoiding false
            # aliasing in the generation of fields
            if len(reg.field_space) < LEGATE_MAX_FIELDS:
                region = reg
                field_id = reg.field_space.allocate_field(self.dtype)
                return region, field_id
        # If we make it here then we need to make a new region
        index_space = self.runtime.find_or_create_index_space(self.shape)
        field_space = self.runtime.find_or_create_field_space(self.dtype)
        handle = legion.legion_logical_region_create(
            self.runtime.runtime,
            self.runtime.context,
            index_space.handle,
            field_space.handle,
            True,
        )
        region = Region(
            self.runtime.context,
            self.runtime.runtime,
            index_space,
            field_space,
            handle,
        )
        self.top_regions.append(region)
        field_id = None
        # See if this is a new fields space or not
        if len(field_space) > 0:
            # This field space has been used already, grab the first
            # field for ourselves and put any other ones on the free list
            for fid in _iterkeys(field_space.fields):
                if field_id is None:
                    field_id = fid
                else:
                    self.free_fields.append((region, fid))
        else:
            field_id = field_space.allocate_field(self.dtype)
        return region, field_id

    def free_field(self, region, field_id, ordered=False):
        if ordered:
            # Issue a fill to clear the field for re-use and enable the
            # Legion garbage collector to reclaim any physical instances
            # We'll disable this for now until we see evidence that we
            # actually need the Legion garbage collector
            # if self.initial_future is None:
            #    value = np.array(0, dtype=dtype)
            #    self.initial_future = self.runtime.create_future(value.data, value.nbytes) # noqa E501
            #    self.fill_space = self.runtime.compute_parallel_launch_space_by_shape( # noqa E501
            #                                                                    self.shape) # noqa E501
            #    if self.fill_space is not None:
            #        self.tile_shape = self.runtime.compute_tile_shape(self.shape, # noqa E501
            #                                                          self.fill_space) # noqa E501
            # if self.fill_space is not None and self.tile_shape in region.tile_partitions: # noqa E501
            #    partition = region.tile_partitions[self.tile_key]
            #    fill = IndexFill(partition, 0, region, field_id, self.initial_future, # noqa E501
            #                     mapper=self.runtime.mapper_id)
            # else:
            #    # We better be the top-level region
            #    fill = Fill(region, region, field_id, self.initial_future,
            #                mapper=self.runtime.mapper_id)
            # self.runtime.dispatch(fill)
            if self.free_fields is not None:
                self.free_fields.append((region, field_id))
        else:  # Put this on the unordered list
            if self.freed_fields is not None:
                self.freed_fields.append((region, field_id))


def _find_or_create_partition(
    runtime, region, color_shape, tile_shape, offset, transform
):
    # Compute the extent and transform for this partition operation
    lo = (0,) * len(tile_shape)
    # Legion is inclusive so map down
    hi = tuple(map(lambda x: (x - 1), tile_shape))
    if offset is not None:
        assert len(offset) == len(tile_shape)
        lo = tuple(map(lambda x, y: (x + y), lo, offset))
        hi = tuple(map(lambda x, y: (x + y), hi, offset))
    # Construct the transform to use based on the color space
    tile_transform = Transform(len(tile_shape), len(tile_shape))
    for idx, tile in enumerate(tile_shape):
        tile_transform.trans[idx, idx] = tile
    # If we have a translation back to the region space we need to apply that
    if transform is not None:
        # Transform the extent points into the region space
        lo = transform.apply(lo)
        hi = transform.apply(hi)
        # Compose the transform from the color space into our shape space with
        # the transform from our shape space to region space
        tile_transform = tile_transform.compose(transform)
    # Now that we have the points in the global coordinate space we can build
    # the domain for the extent
    extent = Rect(hi, lo, exclusive=False)
    # Check to see if we already made a partition like this
    if region.index_space.children:
        color_lo = Point((0,) * len(color_shape), dim=len(color_shape))
        color_hi = Point(dim=len(color_shape))
        for idx in range(color_hi.dim):
            color_hi[idx] = color_shape[idx] - 1
        for part in region.index_space.children:
            if not isinstance(part.functor, PartitionByRestriction):
                continue
            if part.functor.transform != tile_transform:
                continue
            if part.functor.extent != extent:
                continue
            # Lastly check that the index space domains match
            color_bounds = part.color_space.get_bounds()
            if color_bounds.lo != color_lo or color_bounds.hi != color_hi:
                continue
            # Get the corresponding logical partition
            result = region.get_child(part)
            # Annotate it with our meta-data
            if not hasattr(result, "color_shape"):
                result.color_shape = color_shape
                result.tile_shape = tile_shape
                result.tile_offset = offset
            return result
    color_space = runtime.find_or_create_index_space(color_shape)
    functor = PartitionByRestriction(tile_transform, extent)
    index_partition = IndexPartition(
        runtime.context,
        runtime.runtime,
        region.index_space,
        color_space,
        functor,
        kind=legion.LEGION_DISJOINT_COMPLETE_KIND,
        keep=True,  # export this partition functor to other libraries
    )
    partition = region.get_child(index_partition)
    partition.color_shape = color_shape
    partition.tile_shape = tile_shape
    partition.tile_offset = offset
    return partition


# A region field holds a reference to a field in a logical region
class RegionField(object):
    def __init__(
        self,
        runtime,
        region,
        field,
        shape,
        parent=None,
        transform=None,
        dim_map=None,
        key=None,
        view=None,
    ):
        self.runtime = runtime
        self.region = region
        self.field = field
        self.shape = shape
        self.parent = parent
        self.transform = transform
        self.dim_map = dim_map
        self.key = key
        self.key_partition = None  # The key partition for this region field
        self.subviews = None  # RegionField subviews of this region field
        self.view = view  # The view slice tuple used to make this region field
        self.launch_space = None  # Parallel launch space for this region_field
        self.shard_function = 0  # Default to no shard function
        self.shard_space = None  # Sharding space for launches
        self.shard_point = None  # Tile point we overlap with in root
        self.attach_array = None  # Numpy array that we attached to this field
        self.numpy_array = (
            None  # Numpy array that we returned for the application
        )
        self.interface = None  # Numpy array interface
        self.physical_region = None  # Physical region for attach
        self.physical_region_refs = 0
        self.physical_region_mapped = False

    def __del__(self):
        if self.attach_array is not None:
            self.detach_numpy_array(unordered=True, defer=True)

    def has_parallel_launch_space(self):
        return self.launch_space is not None

    def compute_parallel_launch_space(self):
        # See if we computed it already
        if self.launch_space == ():
            return None
        if self.launch_space is not None:
            return self.launch_space
        if self.parent is not None:
            key_partition, _, __ = self.find_or_create_key_partition()
            if key_partition is None:
                self.launch_space = ()
            else:
                self.launch_space = key_partition.color_shape
        else:  # top-level region so just do the natural partitioning
            self.launch_space = self.runtime.compute_parallel_launch_space_by_shape(  # noqa E501
                self.shape
            )
            if self.launch_space is None:
                self.launch_space = ()
        if self.launch_space == ():
            return None
        return self.launch_space

    def find_point_sharding(self):
        # By the time we call this we should have a launch space
        # even if it is an empty one
        assert self.launch_space is not None
        return self.shard_point, self.shard_function, self.shard_space

    def set_key_partition(self, part, shardfn, shardsp):
        self.launch_space = part.color_shape
        self.key_partition = part
        self.shard_function = shardfn
        self.shard_space = shardsp

    def find_or_create_key_partition(self):
        if self.key_partition is not None:
            return self.key_partition, self.shard_function, self.shard_space
        # We already tried to compute it and did not have one so we're done
        if self.launch_space == ():
            return None, None, None
        if self.parent is not None:
            # Figure out how many tiles we overlap with of the root
            root = self.parent
            while root.parent is not None:
                root = root.parent
            root_key, rootfn, rootsp = root.find_or_create_key_partition()
            if root_key is None:
                self.launch_space = ()
                return None, None, None
            # Project our bounds through the transform into the
            # root coordinate space to get our bounds in the root
            # coordinate space
            lo = np.zeros((len(self.shape),), dtype=np.int64)
            hi = np.array(self.shape, dtype=np.int64) - 1
            if self.transform:
                lo = self.transform.apply(lo)
                hi = self.transform.apply(hi)
            # Compute the lower bound tile and upper bound tile
            assert len(lo) == len(root_key.tile_shape)
            color_lo = tuple(map(lambda x, y: x // y, lo, root_key.tile_shape))
            color_hi = tuple(map(lambda x, y: x // y, hi, root_key.tile_shape))
            color_tile = root_key.tile_shape
            if self.transform:
                # Check to see if this transform is invertible
                # If it is then we'll reuse the key partition of the
                # root in order to guide how we do the partitioning
                # for this view to maximimize locality. If the transform
                # is not invertible then we'll fall back to doing the
                # standard mapping of the index space
                invertible = True
                for m in range(len(root.shape)):
                    nonzero = False
                    for n in range(len(self.shape)):
                        if self.transform.trans[m, n] != 0:
                            if nonzero:
                                invertible = False
                                break
                            if self.transform.trans[m, n] != 1:
                                invertible = False
                                break
                            nonzero = True
                    if not invertible:
                        break
                if not invertible:
                    # Not invertible so fall back to the standard case
                    launch_space = (
                        self.runtime.compute_parallel_launch_space_by_shape(
                            self.shape
                        )
                    )
                    if launch_space == ():
                        return None, None
                    tile_shape = self.runtime.compute_tile_shape(
                        self.shape, launch_space
                    )
                    self.key_partition = _find_or_create_partition(
                        self.runtime,
                        self.region,
                        launch_space,
                        tile_shape,
                        offset=(0,) * len(launch_space),
                        transform=self.transform,
                    )
                    self.shard_function = (
                        self.runtime.first_shard_id
                        + legate_numpy.NUMPY_SHARD_TILE_1D
                        + len(self.shape)
                        - 1
                    )
                    return (
                        self.key_partition,
                        self.shard_function,
                        self.shard_space,
                    )
                # We're invertible so do the standard inversion
                inverse = np.transpose(self.transform.trans)
                # We need to make a make a special sharding functor here that
                # projects the points in our launch space back into the space
                # of the root partitions sharding space
                # First construct the affine mapping for points in our launch
                # space back into the launch space of the root
                # This is the special case where we have a special shard
                # function and sharding space that is different than our normal
                # launch space because it's a subset of the root's launch space
                launch_transform = AffineTransform(
                    len(root.shape), len(self.shape), False
                )
                launch_transform.trans = self.transform.trans
                launch_transform.offset = color_lo
                self.shard_function = (
                    self.runtime.find_or_create_transform_sharding_functor(
                        launch_transform
                    )
                )
                tile_offset = np.zeros((len(self.shape),), dtype=np.int64)
                for n in range(len(self.shape)):
                    nonzero = False
                    for m in range(len(root.shape)):
                        if inverse[n, m] == 0:
                            continue
                        nonzero = True
                        break
                    if not nonzero:
                        tile_offset[n] = 1
                color_lo = tuple((inverse @ color_lo).flatten())
                color_hi = tuple((inverse @ color_hi).flatten())
                color_tile = tuple(
                    (inverse @ color_tile).flatten() + tile_offset
                )
                # Reset lo and hi back to our space
                lo = np.zeros((len(self.shape),), dtype=np.int64)
                hi = np.array(self.shape, dtype=np.int64) - 1
            else:
                # If there is no transform then can just use the root function
                self.shard_function = rootfn
            self.shard_space = root_key.index_partition.color_space
            # Launch space is how many tiles we have in each dimension
            color_shape = tuple(
                map(lambda x, y: (x - y) + 1, color_hi, color_lo)
            )
            # Check to see if they are all one, if so then we don't even need
            # to bother with making the partition
            volume = reduce(lambda x, y: x * y, color_shape)
            assert volume > 0
            if volume == 1:
                self.launch_space = ()
                # We overlap with exactly one point in the root
                # Therefore just record this point
                self.shard_point = Point(color_lo)
                return None, None, None
            # Now compute the offset for the partitioning
            # This will shift the tile down if necessary to align with the
            # boundaries at the root while still covering all of our elements
            offset = tuple(
                map(
                    lambda x, y: 0 if (x % y) == 0 else ((x % y) - y),
                    lo,
                    color_tile,
                )
            )
            self.key_partition = _find_or_create_partition(
                self.runtime,
                self.region,
                color_shape,
                color_tile,
                offset,
                self.transform,
            )
        else:
            launch_space = self.compute_parallel_launch_space()
            if launch_space is None:
                return None, None, None
            tile_shape = self.runtime.compute_tile_shape(
                self.shape, launch_space
            )
            self.key_partition = _find_or_create_partition(
                self.runtime,
                self.region,
                launch_space,
                tile_shape,
                offset=(0,) * len(launch_space),
                transform=self.transform,
            )
            self.shard_function = (
                self.runtime.first_shard_id
                + legate_numpy.NUMPY_SHARD_TILE_1D
                + len(self.shape)
                - 1
            )
        return self.key_partition, self.shard_function, self.shard_space

    def find_or_create_congruent_partition(
        self, part, transform=None, offset=None
    ):
        if transform is not None:
            shape_transform = AffineTransform(
                transform.shape[0], transform.shape[1], False
            )
            shape_transform.trans = transform
            shape_transform.offset = offset
            offset_transform = Transform(
                transform.shape[0], transform.shape[1], False
            )
            offset_transform.trans = transform
            return self.find_or_create_partition(
                shape_transform.apply(part.color_shape),
                shape_transform.apply(part.tile_shape),
                offset_transform.apply(part.tile_offset),
            )
        else:
            assert len(self.shape) == len(part.color_shape)
            return self.find_or_create_partition(
                part.color_shape, part.tile_shape, part.tile_offset
            )

    def find_or_create_partition(
        self, launch_space, tile_shape=None, offset=None
    ):
        # Compute a tile shape based on our shape
        if tile_shape is None:
            tile_shape = self.runtime.compute_tile_shape(
                self.shape, launch_space
            )
        if offset is None:
            offset = (0,) * len(launch_space)
        # Tile shape should have the same number of dimensions as our shape
        assert len(launch_space) == len(self.shape)
        assert len(tile_shape) == len(self.shape)
        assert len(offset) == len(self.shape)
        # Do a quick check to see if this is congruent to our key partition
        if (
            self.key_partition is not None
            and launch_space == self.key_partition.color_shape
            and tile_shape == self.key_partition.tile_shape
            and offset == self.key_partition.tile_offset
        ):
            return self.key_partition
        # Continue this process on the region object, to ensure any created
        # partitions are shared between RegionField objects referring to the
        # same region
        return _find_or_create_partition(
            self.runtime,
            self.region,
            launch_space,
            tile_shape,
            offset,
            self.transform,
        )

    def find_or_create_indirect_partition(self, launch_space):
        assert len(launch_space) != len(self.shape)
        # If there is a mismatch in the number of dimensions then we need
        # to compute a partition and projection functor that can transform
        # the points into a partition that makes sense
        raise NotImplementedError("need support for indirect partitioning")

    def attach_numpy_array(self, numpy_array, share=False):
        assert self.parent is None
        assert isinstance(numpy_array, np.ndarray)
        # If we already have a numpy array attached
        # then we have to detach it first
        if self.attach_array is not None:
            if self.attach_array is numpy_array:
                return
            else:
                self.detach_numpy_array(unordered=False)
        # Now we can attach the new one and then do the acquire
        attach = Attach(
            self.region,
            self.field.field_id,
            numpy_array,
            mapper=self.runtime.mapper_id,
        )
        # If we're not sharing then there is no need to map or restrict the
        # attachment
        if not share:
            # No need for restriction for us
            attach.set_restricted(False)
            # No need for mapping in the restricted case
            attach.set_mapped(False)
        else:
            self.physical_region_mapped = True
        self.physical_region = self.runtime.dispatch(attach)
        # Due to the working of the Python interpreter's garbage collection
        # algorithm we make the detach operation for this now and register it
        # with the runtime so that we know that it won't be collected when the
        # RegionField object is collected
        detach = Detach(self.physical_region, flush=True)
        # Dangle these fields off here to prevent premature collection
        detach.field = self.field
        detach.array = numpy_array
        self.detach_key = self.runtime.register_detachment(detach)
        # Add a reference here to prevent collection in for inline mapped cases
        assert self.physical_region_refs == 0
        # This reference will never be removed, we'll delete the
        # physical region once the object is deleted
        self.physical_region_refs = 1
        self.attach_array = numpy_array
        if share:
            # If we're sharing this then we can also make this our numpy array
            self.numpy_array = weakref.ref(numpy_array)

    def detach_numpy_array(self, unordered, defer=False):
        assert self.parent is None
        assert self.attach_array is not None
        assert self.physical_region is not None
        detach = self.runtime.remove_detachment(self.detach_key)
        detach.unordered = unordered
        self.runtime.detach_array_field(
            self.attach_array, self.field, detach, defer
        )
        self.physical_region = None
        self.physical_region_mapped = False
        self.attach_array = None

    def get_inline_mapped_region(self):
        if self.parent is None:
            if self.physical_region is None:
                # We don't have a valid numpy array so we need to do an inline
                # mapping and then use the buffer to share the storage
                mapping = InlineMapping(
                    self.region,
                    self.field.field_id,
                    mapper=self.runtime.mapper_id,
                )
                self.physical_region = self.runtime.dispatch(mapping)
                self.physical_region_mapped = True
                # Wait until it is valid before returning
                self.physical_region.wait_until_valid()
            elif not self.physical_region_mapped:
                # If we have a physical region but it is not mapped then
                # we actually need to remap it, we do this by launching it
                self.runtime.dispatch(self.physical_region)
                self.physical_region_mapped = True
                # Wait until it is valid before returning
                self.physical_region.wait_until_valid()
            # Increment our ref count so we know when it can be collected
            self.physical_region_refs += 1
            return self.physical_region
        else:
            return self.parent.get_inline_mapped_region()

    def decrement_inline_mapped_ref_count(self):
        if self.parent is None:
            assert self.physical_region_refs > 0
            self.physical_region_refs -= 1
            if self.physical_region_refs == 0:
                self.runtime.unmap_region(self.physical_region)
                self.physical_region = None
                self.physical_region_mapped = False
        else:
            self.parent.decrement_inline_mapped_ref_count()

    def get_numpy_array(self):
        # See if we still have a valid numpy array to use
        if self.numpy_array is not None:
            # Test the weak reference to see if it is still alive
            result = self.numpy_array()
            if result is not None:
                return result
        physical_region = self.get_inline_mapped_region()
        # We need a pointer to the physical allocation for this physical region
        dim = len(self.shape)
        # Build the accessor for this physical region
        if self.transform is not None:
            # We have a transform so build the accessor special with a
            # transform
            func = getattr(
                legion,
                "legion_physical_region_get_field_accessor_array_{}d_with_transform".format(  # noqa E501
                    dim
                ),
            )
            accessor = func(
                physical_region.handle,
                ffi.cast("legion_field_id_t", self.field.field_id),
                self.transform.raw(),
            )
        else:
            # No transfrom so we can do the normal thing
            func = getattr(
                legion,
                "legion_physical_region_get_field_accessor_array_{}d".format(
                    dim
                ),
            )
            accessor = func(
                physical_region.handle,
                ffi.cast("legion_field_id_t", self.field.field_id),
            )
        # Now that we've got our accessor we can get a pointer to the memory
        rect = ffi.new("legion_rect_{}d_t *".format(dim))
        for d in xrange(dim):
            rect[0].lo.x[d] = 0
            rect[0].hi.x[d] = self.shape[d] - 1  # inclusive
        subrect = ffi.new("legion_rect_{}d_t *".format(dim))
        offsets = ffi.new("legion_byte_offset_t[]", dim)
        func = getattr(
            legion, "legion_accessor_array_{}d_raw_rect_ptr".format(dim)
        )
        base_ptr = func(accessor, rect[0], subrect, offsets)
        assert base_ptr is not None
        # Check that the subrect is the same as in the in rect
        for d in xrange(dim):
            assert rect[0].lo.x[d] == subrect[0].lo.x[d]
            assert rect[0].hi.x[d] == subrect[0].hi.x[d]
        shape = tuple(rect.hi.x[i] - rect.lo.x[i] + 1 for i in xrange(dim))
        strides = tuple(offsets[i].offset for i in xrange(dim))
        # Numpy doesn't know about CFFI pointers, so we have to cast
        # this to a Python long before we can hand it off to Numpy.
        base_ptr = long(ffi.cast("size_t", base_ptr))
        initializer = _RegionNdarray(
            shape, self.field.dtype, base_ptr, strides, False
        )
        array = np.asarray(initializer)

        # This will be the unmap call that will be invoked once the weakref is
        # removed
        # We will use it to unmap the inline mapping that was performed
        def decrement(region_field, ref):
            region_field.decrement_inline_mapped_ref_count()

        # Curry bind arguments to the function
        callback = functools.partial(decrement, self)
        # Save a weak reference to the array so we don't prevent collection
        self.numpy_array = weakref.ref(array, callback)
        return array


# This is a dummy object that is only used as an initializer for the
# RegionField object above. It is thrown away as soon as the
# RegionField is constructed.
class _RegionNdarray(object):
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


class Callsite(object):
    def __init__(self, filename, lineno, funcname, context=None, index=None):
        self.filename = filename
        self.lineno = lineno
        self.funcname = funcname
        if context is not None:
            self.line = context[index]
        else:
            self.line = None

    def __eq__(self, rhs):
        if self.filename != rhs.filename:
            return False
        if self.lineno != rhs.lineno:
            return False
        if self.funcname != rhs.funcname:
            return False
        return True

    def __hash__(self):
        return hash(self.filename) ^ hash(self.lineno) ^ hash(self.funcname)

    def __repr__(self):
        return (
            "Callsite "
            + str(self.filename)
            + ":"
            + str(self.lineno)
            + " "
            + str(self.funcname)
            + ("\n" + self.line)
            if self.line is not None
            else ""
        )


class Runtime(object):
    __slots__ = [
        "context",
        "runtime",
        "index_spaces",
        "field_spaces",
        "registered_detachments",
        "next_detachment_key",
        "deferred_detachments",
        "pending_detachments",
        "current_random_epoch",
        "num_pieces",
        "radix",
        "min_shard_volume",
        "max_eager_volume",
        "max_field_reuse_size",
        "max_field_reuse_frequency",
        "test_mode",
        "launch_spaces",
        "piece_factors",
        "empty_argmap",
        "field_managers",
        "shadow_debug",
        "callsite_summaries",
        "first_task_id",
        "mapper_id",
        "first_redop_id",
        "first_proj_id",
        "first_shard_id",
        "ptr_to_thunk",
        "transform_sharding_functors",
        "transform_sharding_offset",
        "destroyed",
    ]

    def __init__(self, runtime, context):
        self.context = context
        self.runtime = runtime
        try:
            # Prune it out so the application does not see it
            sys.argv.remove("-lg:numpy:shadow")
            self.shadow_debug = True
        except ValueError:
            self.shadow_debug = False
        try:
            # Prune it out so the application does not see it
            sys.argv.remove("-lg:numpy:test")
            self.test_mode = True
        except ValueError:
            self.test_mode = False
        try:
            # Prune it out so the application does not see it
            sys.argv.remove("-lg:numpy:summarize")
            self.callsite_summaries = dict()
        except ValueError:
            self.callsite_summaries = None
        self.index_spaces = OrderedDict()  # map shapes to index spaces
        self.field_spaces = OrderedDict()  # map dtype to field spaces
        self.field_managers = (
            OrderedDict()
        )  # map from (shape,dtype) to field managers
        self.ptr_to_thunk = None  # map from external array pointer to thunks
        self.transform_sharding_functors = None
        self.transform_sharding_offset = legate_numpy.NUMPY_SHARD_EXTRA
        self.registered_detachments = None
        self.next_detachment_key = 0
        self.deferred_detachments = None
        self.pending_detachments = (
            None  # Prevent premature collection of external resources
        )
        self.current_random_epoch = 0
        self.destroyed = False
        # Get the initial task ID and mapper ID
        encoded_name = NUMPY_LIB_NAME.encode("utf-8")
        self.first_task_id = legion.legion_runtime_generate_library_task_ids(
            self.runtime, encoded_name, legate_numpy.NUMPY_MAX_TASKS
        )
        self.mapper_id = legion.legion_runtime_generate_library_mapper_ids(
            self.runtime, encoded_name, legate_numpy.NUMPY_MAX_MAPPERS
        )
        self.first_redop_id = (
            legion.legion_runtime_generate_library_reduction_ids(
                self.runtime, encoded_name, legate_numpy.NUMPY_MAX_REDOPS
            )
        )
        self.first_proj_id = (
            legion.legion_runtime_generate_library_projection_ids(
                self.runtime, encoded_name, legate_numpy.NUMPY_PROJ_LAST
            )
        )
        self.first_shard_id = (
            legion.legion_runtime_generate_library_sharding_ids(
                self.runtime, encoded_name, legate_numpy.NUMPY_SHARD_LAST
            )
        )
        # This next part we can only do if we have a context which we will if
        # we're running on one node or we are control replicated. Alternatively
        # we are running on multiple nodes without control replication and
        # we'll never be here in Python cause all the task implementations are
        # in C++/CUDA
        if self.context is not None:
            # Figure out how many pieces we want to target when making
            # partitions
            # We do this abstractly regardless of knowing the machine type
            f1 = Future(
                legion.legion_runtime_select_tunable_value(
                    self.runtime,
                    self.context,
                    legate_numpy.NUMPY_TUNABLE_NUM_PIECES,
                    self.mapper_id,
                    0,
                )
            )
            # Figure our our radix for reduction trees if necessary
            f2 = Future(
                legion.legion_runtime_select_tunable_value(
                    self.runtime,
                    self.context,
                    legate_numpy.NUMPY_TUNABLE_RADIX,
                    self.mapper_id,
                    0,
                )
            )
            # Figure out the minimum shard volume for arrays
            f3 = Future(
                legion.legion_runtime_select_tunable_value(
                    self.runtime,
                    self.context,
                    legate_numpy.NUMPY_TUNABLE_MIN_SHARD_VOLUME,
                    self.mapper_id,
                    0,
                )
            )
            # Figure out the maximum eager array size
            f4 = Future(
                legion.legion_runtime_select_tunable_value(
                    self.runtime,
                    self.context,
                    legate_numpy.NUMPY_TUNABLE_MAX_EAGER_VOLUME,
                    self.mapper_id,
                    0,
                )
            )
            # Figure out the number of allocations that need to be done before
            # reusing fields
            f5 = Future(
                legion.legion_runtime_select_tunable_value(
                    self.runtime,
                    self.context,
                    legate_numpy.NUMPY_TUNABLE_FIELD_REUSE_SIZE,
                    self.mapper_id,
                    0,
                )
            )
            f6 = Future(
                legion.legion_runtime_select_tunable_value(
                    self.runtime,
                    self.context,
                    legate_numpy.NUMPY_TUNABLE_FIELD_REUSE_FREQUENCY,
                    self.mapper_id,
                    0,
                )
            )
            self.num_pieces = struct.unpack_from("i", f1.get_buffer(4))[0]
            if self.num_pieces > 1:
                self.launch_spaces = dict()
                # Compute the prime number factors of our number of pieces
                factors = list()
                pieces = self.num_pieces
                while pieces % 2 == 0:
                    factors.append(2)
                    pieces = pieces // 2
                while pieces % 3 == 0:
                    factors.append(3)
                    pieces = pieces // 3
                while pieces % 5 == 0:
                    factors.append(5)
                    pieces = pieces // 5
                while pieces % 7 == 0:
                    factors.append(7)
                    pieces = pieces // 7
                while pieces % 11 == 0:
                    factors.append(11)
                    pieces = pieces // 11
                if pieces > 1:
                    raise ValueError(
                        "legate.numpy currently doesn't support processor "
                        + "counts with large prime factors greater than 11"
                    )
                self.piece_factors = list(reversed(factors))
                # Keep an empty argmap around for launching things
                self.empty_argmap = legion.legion_argument_map_create()
            else:
                self.launch_spaces = None
                self.piece_factors = None
                self.empty_argmap = None
            # Wait for the futures and get the results
            self.radix = struct.unpack_from("i", f2.get_buffer(4))[0]
            self.min_shard_volume = struct.unpack_from("i", f3.get_buffer(4))[
                0
            ]
            self.max_eager_volume = struct.unpack_from("i", f4.get_buffer(4))[
                0
            ]
            self.max_field_reuse_size = struct.unpack_from(
                "Q", f5.get_buffer(8)
            )[0]
            self.max_field_reuse_frequency = struct.unpack_from(
                "i", f6.get_buffer(4)
            )[0]
        # Make sure that our NumPyLib object knows about us so it can destroy
        # us
        numpy_lib.set_runtime(self)

    def destroy(self):
        assert not self.destroyed
        if self.empty_argmap is not None:
            legion.legion_argument_map_destroy(self.empty_argmap)
            self.empty_argmap = None
        # Remove references to our legion resources so they can be collected
        self.field_managers = None
        self.field_spaces = None
        self.index_spaces = None
        if self.callsite_summaries is not None:
            f = Future(
                legion.legion_runtime_select_tunable_value(
                    self.runtime,
                    self.context,
                    NumPyTunable.NUM_GPUS,
                    self.mapper_id,
                    0,
                )
            )
            num_gpus = struct.unpack_from("i", f.get_buffer(4))[0]
            print(
                "---------------- Legate.NumPy Callsite Summaries "
                "----------------"
            )
            for callsite, counts in sorted(
                _iteritems(self.callsite_summaries),
                key=lambda site: (
                    site[0].filename,
                    site[0].lineno,
                    site[0].funcname,
                ),
            ):
                print(
                    str(callsite.funcname)
                    + " @ "
                    + str(callsite.filename)
                    + ":"
                    + str(callsite.lineno)
                )
                print("  Invocations: " + str(counts[1]))
                if num_gpus > 0:
                    print(
                        "  Legion GPU Accelerated: %d (%.2f%%)"
                        % (counts[0], (100.0 * counts[0]) / counts[1])
                    )
                else:
                    print(
                        "  Legion CPU Accelerated: %d (%.2f%%)"
                        % (counts[0], (100.0 * counts[0]) / counts[1])
                    )
            print(
                "-------------------------------------------------------------"
                "----"
            )
            self.callsite_summaries = None
        gc.collect()
        while self.deferred_detachments:
            self.perform_detachments()
            # Make sure progress is made on any of these operations
            legion.legion_context_progress_unordered_operations(
                self.runtime, self.context
            )
            gc.collect()
        # Always make sure we wait for any pending detachments to be done
        # so that we don't lose the references and make the GC unhappy
        gc.collect()
        while self.pending_detachments:
            self.prune_detachments()
            gc.collect()
        self.destroyed = True

    def create_callsite(self, stacklevel):
        assert stacklevel > 0
        stack = inspect.stack()
        caller_frame = stack[stacklevel]
        callee_frame = stack[stacklevel - 1]
        return Callsite(
            caller_frame[1],
            caller_frame[2],
            callee_frame[3],
            caller_frame[4],
            caller_frame[5],
        )

    def profile_callsite(self, stacklevel, accelerated, callsite=None):
        if self.callsite_summaries is None:
            return
        if callsite is None:
            callsite = self.create_callsite(stacklevel + 1)
        assert isinstance(callsite, Callsite)
        # Record the callsite if we haven't done so already
        if callsite in self.callsite_summaries:
            counts = self.callsite_summaries[callsite]
            self.callsite_summaries[callsite] = (
                counts[0] + 1 if accelerated else 0,
                counts[1] + 1,
            )
        else:
            self.callsite_summaries[callsite] = (1 if accelerated else 0, 1)

    def create_future(self, data, size, wrap=False, dtype=None, shape=()):
        result = Future()
        result.set_value(self.runtime, data, size)
        if wrap:
            assert dtype is not None
            # If we have a shape then all the extents should be 1 for now
            for extent in shape:
                assert extent == 1
            return DeferredArray(
                self, result, shape=shape, dtype=dtype, scalar=True
            )
        else:
            return result

    def allocate_field(self, shape, dtype):
        assert not self.destroyed
        region = None
        field_id = None
        # Regions all have fields of the same field type and shape
        key = (shape, dtype)
        # if we don't have a field manager yet then make one
        if key not in self.field_managers:
            self.field_managers[key] = FieldManager(self, shape, dtype)
        region, field_id = self.field_managers[key].allocate_field()
        field = Field(self, region, field_id, dtype, shape)
        return RegionField(self, region, field, shape)

    def free_field(self, region, field_id, dtype, shape, partition):
        # Have a guard here to make sure that we don't try to
        # do this after we have been destroyed
        if self.destroyed:
            return
        # Now save it in our data structure for free fields eligible for reuse
        key = (shape, dtype)
        if self.field_managers is not None:
            self.field_managers[key].free_field(region, field_id)

    def compute_parallel_launch_space_by_shape(self, shape):
        assert self.num_pieces > 0
        # Easy case if we only have one piece: no parallel launch space
        if self.num_pieces == 1:
            return None
        # If there is only one point or no points then we never do a parallel
        # launch
        all_ones_or_zeros = True
        for ext in shape:
            if ext > 1:
                all_ones_or_zeros = False
                break
            else:  # Better be a one or zero if we get here
                assert ext == 1 or ext == 0
        # If we only have one point then we never do parallel launches
        if all_ones_or_zeros:
            return None
        # Check to see if we already did the math
        if shape in self.launch_spaces:
            return self.launch_spaces[shape]
        # Prune out any dimensions that are 1
        temp_shape = ()
        temp_dims = ()
        volume = 1
        for dim in xrange(len(shape)):
            assert shape[dim] > 0
            if shape[dim] == 1:
                continue
            temp_shape = temp_shape + (shape[dim],)
            temp_dims = temp_dims + (dim,)
            volume *= shape[dim]
        # Figure out how many shards we can make with this array
        max_pieces = (
            volume + self.min_shard_volume - 1
        ) // self.min_shard_volume
        assert max_pieces > 0
        # If we can only make one piece return that now
        if max_pieces == 1:
            self.launch_spaces[shape] = None
            return None
        else:
            # TODO: a better heuristic here For now if we can make at least two
            # pieces then we will make N pieces
            max_pieces = self.num_pieces
        # Otherwise we need to compute it ourselves
        # First compute the N-th root of the number of pieces
        dims = len(temp_shape)
        temp_result = ()
        if dims == 0:
            # Project back onto the original number of dimensions
            result = ()
            for dim in xrange(len(shape)):
                result = result + (1,)
            return result
        elif dims == 1:
            # Easy case for one dimensional things
            temp_result = (min(temp_shape[0], max_pieces),)
        elif dims == 2:
            if volume < max_pieces:
                # TBD: Once the max_pieces heuristic is fixed, this should
                # never happen
                temp_result = temp_shape
            else:
                # Two dimensional so we can use square root to try and generate
                # as square a pieces as possible since most often we will be
                # doing matrix operations with these
                nx = temp_shape[0]
                ny = temp_shape[1]
                swap = nx > ny
                if swap:
                    temp = nx
                    nx = ny
                    ny = temp
                n = math.sqrt(float(max_pieces * nx) / float(ny))
                # Need to constraint n to be an integer with numpcs % n == 0
                # try rounding n both up and down
                n1 = int(math.floor(n + 1e-12))
                n1 = max(n1, 1)
                while max_pieces % n1 != 0:
                    n1 -= 1
                n2 = int(math.ceil(n - 1e-12))
                while max_pieces % n2 != 0:
                    n2 += 1
                # pick whichever of n1 and n2 gives blocks closest to square
                # i.e. gives the shortest long side
                side1 = max(nx // n1, ny // (max_pieces // n1))
                side2 = max(nx // n2, ny // (max_pieces // n2))
                px = n1 if side1 <= side2 else n2
                py = max_pieces // px
                # we need to trim launch space if it is larger than the
                # original shape in one of the dimensions (can happen in
                # testing)
                if swap:
                    temp_result = (
                        min(py, temp_shape[0]),
                        min(px, temp_shape[1]),
                    )
                else:
                    temp_result = (
                        min(px, temp_shape[0]),
                        min(py, temp_shape[1]),
                    )
        else:
            # For higher dimensions we care less about "square"-ness
            # and more about evenly dividing things, compute the prime
            # factors for our number of pieces and then round-robin
            # them onto the shape, with the goal being to keep the
            # last dimension >= 32 for good memory performance on the GPU
            temp_result = list()
            for dim in xrange(dims):
                temp_result.append(1)
            factor_prod = 1
            for factor in self.piece_factors:
                # Avoid exceeding the maximum number of pieces
                if factor * factor_prod > max_pieces:
                    break
                factor_prod *= factor
                remaining = tuple(
                    map(lambda s, r: (s + r - 1) // r, temp_shape, temp_result)
                )
                big_dim = remaining.index(max(remaining))
                if big_dim < len(temp_dims) - 1:
                    # Not the last dimension, so do it
                    temp_result[big_dim] *= factor
                else:
                    # Last dim so see if it still bigger than 32
                    if (
                        len(remaining) == 1
                        or remaining[big_dim] // factor >= 32
                    ):
                        # go ahead and do it
                        temp_result[big_dim] *= factor
                    else:
                        # Won't be see if we can do it with one of the other
                        # dimensions
                        big_dim = remaining.index(
                            max(remaining[0 : len(remaining) - 1])
                        )
                        if remaining[big_dim] // factor > 0:
                            temp_result[big_dim] *= factor
                        else:
                            # Fine just do it on the last dimension
                            temp_result[len(temp_dims) - 1] *= factor
        # Project back onto the original number of dimensions
        assert len(temp_result) == dims
        result = ()
        for dim in xrange(len(shape)):
            if dim in temp_dims:
                result = result + (temp_result[temp_dims.index(dim)],)
            else:
                result = result + (1,)
        # Save the result for later
        self.launch_spaces[shape] = result
        return result

    def compute_tile_shape(self, shape, launch_space):
        assert len(shape) == len(launch_space)
        # Over approximate the tiles so that the ends might be small
        return tuple(map(lambda x, y: (x + y - 1) // y, shape, launch_space))

    def find_or_create_index_space(self, bounds):
        if bounds in self.index_spaces:
            return self.index_spaces[bounds]
        # Haven't seen this before so make it now
        rect = Rect(bounds)
        handle = legion.legion_index_space_create_domain(
            self.runtime, self.context, rect.raw()
        )
        result = IndexSpace(self.context, self.runtime, handle=handle)
        # Save this for the future
        self.index_spaces[bounds] = result
        return result

    def find_or_create_field_space(self, dtype):
        if dtype in self.field_spaces:
            return self.field_spaces[dtype]
        # Haven't seen this type before so make it now
        field_space = FieldSpace(self.context, self.runtime)
        self.field_spaces[dtype] = field_space
        return field_space

    def find_or_create_view(self, parent, view, dim_map, shape, key):
        assert len(shape) <= len(view)
        assert len(parent.shape) <= len(view)
        assert len(view) == len(dim_map)
        # Iterate through our parent region's subviews and see if
        # we find one that matches the view that we want
        if parent.subviews:
            for child in parent.subviews:
                if child.view == view:
                    return DeferredArray(
                        self,
                        child,
                        shape,
                        parent.field.dtype,
                        scalar=False,
                    )
        # We need to make this subview
        # If all the slices have strides of one then this is a dense
        # subview and we can make this partition with a call to create
        # partition by restriction, otherwise we'll fall back to the
        # general but slow partition by field, we'll also compute our
        # transform back to the parent address space here
        dense = True
        # Transfrom from our space back to the parent's space
        transform = AffineTransform(len(parent.shape), len(shape), False)
        parent_idx = 0  # Index of parent dimensions
        child_idx = 0  # Index of child dimensions
        for idx in xrange(len(view)):
            # If this is an added dimension then it doesn't even contribute
            # back to the parent space so we can skip it
            if dim_map[idx] > 0:
                child_idx += 1
                continue
            slc = view[idx]
            assert parent_idx < len(parent.shape)
            transform.offset[parent_idx] = slc.start
            assert (
                slc.step >= 0
            )  # Should have handled negative values before this
            # If this is a collapsed dimension then we can skip it
            if dim_map[idx] < 0:
                parent_idx += 1
                continue
            assert child_idx < len(shape)
            transform.trans[parent_idx, child_idx] = slc.step
            child_idx += 1
            parent_idx += 1
            # Our temporary density check for now
            if slc.step > 1:
                dense = False
        assert child_idx == len(shape)
        assert parent_idx == len(parent.shape)
        # Compose our transforms if necessary
        if parent.transform:
            transform = transform.compose(parent.transform)
        # If the child shape has the same number of points as the parent
        # region then we don't actually need to make a subregion, we can
        # just use the parent region with the transform
        parent_volume = 1
        for dim in xrange(len(parent.shape)):
            parent_volume *= parent.shape[dim]
        child_volume = 1
        for dim in xrange(len(shape)):
            child_volume *= shape[dim]
        if parent_volume == child_volume:
            # Same number of points, so no need to make a subregion here
            region_field = RegionField(
                self,
                parent.region,
                parent.field,
                shape,
                parent,
                transform,
                dim_map,
                key,
            )
            return DeferredArray(
                self, region_field, shape, parent.field.dtype, scalar=False
            )
        elif dense:
            # We can do a single call to create partition by restriction
            # Build the rect for the subview
            lo = ()
            hi = ()
            # As an interesting optimization, if we can evenly tile the
            # region in all dimensions of the parent region, then we'll make
            # a disjoint tiled partition with as many children as possible
            tile = True
            tile_shape = ()
            for dim in xrange(len(view)):
                slc = view[dim]
                lo += (slc.start,)
                # Legion is inclusive
                hi += (slc.stop - 1,)
                # If we're still trying to tile do the analysis
                if tile:
                    stride = slc.stop - slc.start
                    if slc.start > 0 and (slc.start % stride) != 0:
                        tile = False
                        continue
                    if (
                        slc.stop < parent.shape[dim]
                        and ((parent.shape[dim] - slc.stop) % stride) != 0
                    ):
                        tile = False
                        continue
                    tile_shape += (stride,)
            if tile:
                # Compute the color space bounds and then see how big it is
                assert len(lo) == len(tile_shape)
                color_space_bounds = ()
                for dim in xrange(len(lo)):
                    assert (parent.shape[dim] % tile_shape[dim]) == 0
                    color_space_bounds += (
                        parent.shape[dim] // tile_shape[dim],
                    )
                volume = reduce(lambda x, y: x * y, color_space_bounds)
                # If it would generate a very large number of elements then
                # we'll apply a heuristic for now and not actually tile it
                # TODO: A better heurisitc for this in the future
                if volume > 256 and volume > 16 * self.num_pieces:
                    tile = False
            # See if we're making a tiled partition or a one-off partition
            if tile:
                # Compute the color of the tile that we care about
                tile_color = ()
                for dim in xrange(len(lo)):
                    assert (lo[dim] % tile_shape[dim]) == 0
                    tile_color += (lo[dim] // tile_shape[dim],)
                assert len(view) == len(tile_shape)
                partition = _find_or_create_partition(
                    self,
                    parent.region,
                    color_space_bounds,
                    tile_shape,
                    None,
                    parent.transform,
                )
                # Then we can build the actual child region that we want and
                # save it in the subviews that we computed
                child_region = partition.get_child(Point(tile_color))
                if not parent.subviews:
                    parent.subviews = list()
                region_field = RegionField(
                    self,
                    child_region,
                    parent.field,
                    shape,
                    parent,
                    transform,
                    dim_map,
                    key,
                    view,
                )
                parent.subviews.append(region_field)
                return DeferredArray(
                    self, region_field, shape, parent.field.dtype, scalar=False
                )
            else:
                # If necessary we may need to transform these dimensions back
                # into the global address space, not we do this with the parent
                # transform since they are in the parent's coordinate space
                if parent.transform:
                    lo = parent.transform.apply(lo)
                    hi = parent.transform.apply(hi)
                # Now that we have the points in the global coordinate space
                # we can build the domain for the extent
                extent = Rect(hi, lo, exclusive=False)
                # Get the unit color space
                color_space = self.find_or_create_index_space((1,))
                # Function for doing the call to make the partition
                identity_transform = Transform(len(lo), 1)
                functor = PartitionByRestriction(identity_transform, extent)
                index_partition = IndexPartition(
                    self.context,
                    self.runtime,
                    parent.region.index_space,
                    color_space,
                    functor,
                    kind=legion.LEGION_DISJOINT_INCOMPLETE_KIND,
                    keep=True,
                )
                partition = parent.region.get_child(index_partition)
                child_region = partition.get_child(Point((0,)))
                if not parent.subviews:
                    parent.subviews = list()
                region_field = RegionField(
                    self,
                    child_region,
                    parent.field,
                    shape,
                    parent,
                    transform,
                    dim_map,
                    key,
                    view,
                )
                parent.subviews.append(region_field)
                return DeferredArray(
                    self, region_field, shape, parent.field.dtype, scalar=False
                )
        else:
            # We need fill in a phased partition operation from Legion
            raise NotImplementedError("implement partition by phase")

    def create_transform_view(self, region_field, new_shape, transform):
        assert isinstance(region_field, RegionField)
        # Compose the transform if necessary
        if region_field.transform is not None:
            transform = transform.compose(region_field.transform)
        new_region_field = RegionField(
            self,
            region_field.region,
            region_field.field,
            shape=new_shape,
            transform=transform,
            parent=region_field,
        )
        return DeferredArray(
            self,
            new_region_field,
            new_shape,
            region_field.field.dtype,
            scalar=False,
        )

    def find_or_create_transform_sharding_functor(self, transform):
        if self.transform_sharding_functors is None:
            self.transform_sharding_functors = dict()
        key = (transform.M, transform.N)
        transforms = self.transform_sharding_functors.get(key)
        if transforms is None:
            transforms = list()
            self.transform_sharding_functors[key] = transforms
        for previous, func in transforms:
            if previous == transform:
                return func
        if self.transform_sharding_offset == legate_numpy.NUMPY_SHARD_LAST:
            raise RuntimeError(
                "Exceeded allocations of sharding IDs. "
                "Increase value of NUMPY_SHARD_LAST in legate_numpy_c.h"
            )
        shard_offset = self.transform_sharding_offset
        self.transform_sharding_offset += 1
        # If we get here then we need to make it, call out to
        # NumPy C API to make and register the sharding function
        legate_numpy.legate_numpy_create_transform_sharding_functor(
            self.first_shard_id,
            shard_offset,
            transform.M,
            transform.N,
            ffi.cast("long *", transform.transform.ctypes.data),
        )
        func = self.first_shard_id + shard_offset
        transforms.append((transform, func))
        return func

    def dispatch(self, operation, redop=None):
        # See if we have any deferred or pending detachments to deal with
        if self.deferred_detachments:
            self.perform_detachments()
        if self.pending_detachments:
            self.prune_detachments()
        # Launch the operation, always user our mapper
        if redop:
            return operation.launch(self.runtime, self.context, redop)
        else:
            return operation.launch(self.runtime, self.context)

    def unmap_region(self, physical_region):
        physical_region.unmap(self.runtime, self.context)

    def perform_detachments(self):
        detachments = self.deferred_detachments
        self.deferred_detachments = None
        for array, field, detach in detachments:
            self.detach_array_field(array, field, detach, defer=False)

    def register_detachment(self, detach):
        key = self.next_detachment_key
        if self.registered_detachments is None:
            self.registered_detachments = dict()
        self.registered_detachments[key] = detach
        self.next_detachment_key += 1
        return key

    def remove_detachment(self, detach_key):
        detach = self.registered_detachments[detach_key]
        del self.registered_detachments[detach_key]
        return detach

    def prune_detachments(self):
        to_remove = None
        for future in _iterkeys(self.pending_detachments):
            if future.is_ready():
                if to_remove is None:
                    to_remove = list()
                to_remove.append(future)
        if to_remove is not None:
            for future in to_remove:
                del self.pending_detachments[future]

    def set_next_random_epoch(self, epoch):
        self.current_random_epoch = epoch

    def get_next_random_epoch(self):
        result = self.current_random_epoch
        self.current_random_epoch += 1
        return result

    @staticmethod
    def is_supported_type(dtype):
        assert isinstance(dtype, np.dtype)
        return dtype.type in numpy_field_type_offsets

    def get_numpy_thunk(self, obj, stacklevel, share=False, dtype=None):
        # Check to see if this object implements the Legate data interface
        if hasattr(obj, "__legate_data_interface__"):
            legate_data = obj.__legate_data_interface__
            if legate_data["version"] != 1:
                raise NotImplementedError(
                    "Need support for other Legate data interface versions"
                )
            data = legate_data["data"]
            if len(data) != 1:
                raise ValueError("Legate data must be array-like")
            field = next(iter(data))
            array = data[field]
            stores = array.stores()
            if len(stores) != 2:
                raise ValueError("Legate data must be array-like")
            if stores[0] is not None:
                raise NotImplementedError("Need support for masked arrays")
            store = stores[1]
            kind = store.kind
            dtype = np.dtype(store.type.to_pandas_dtype())
            primitive = store.storage
            if kind == Future:
                return DeferredArray(
                    self, primitive, shape=(), dtype=dtype, scalar=True
                )
            elif kind == FutureMap:
                raise NotImplementedError("Need support for FutureMap inputs")
            elif kind == (Region, FieldID):
                region_field = self.instantiate_region_field(
                    primitive[0], primitive[1].field_id, dtype
                )
            elif kind == (Region, int):
                region_field = self.instantiate_region_field(
                    primitive[0], primitive[1], dtype
                )
            else:
                raise TypeError("Unknown LegateStore type")
            return DeferredArray(
                self, region_field, region_field.shape, dtype, scalar=False
            )
        # See if this is a normal numpy array
        if not isinstance(obj, np.ndarray):
            # If it's not, make it into a numpy array
            if share:
                obj = np.asarray(obj, dtype=dtype)
            else:
                obj = np.array(obj, dtype=dtype)
        elif dtype is not None and dtype != obj.dtype:
            obj = obj.astype(dtype)
        elif not share:
            obj = obj.copy()
        return self.find_or_create_array_thunk(
            obj, stacklevel=(stacklevel + 1), share=share
        )

    def instantiate_region_field(self, region, fid, dtype):
        if region.parent is None:
            # This is just a top-level region so the conversion is easy
            bounds = region.index_space.domain
            if not bounds.dense:
                raise ValueError(
                    "legate.numpy currently only support "
                    + "dense legate thunks"
                )
            # figure out the shape and transform for this top-level region
            shape = ()
            need_transform = False
            for idx in xrange(bounds.rect.dim):
                if bounds.rect.lo[idx] != 0:
                    shape += ((bounds.rect.hi[idx] - bounds.rect.lo[idx]) + 1,)
                    need_transform = True
                else:
                    shape += (bounds.rect.hi[idx] + 1,)
            # Make the field
            field = Field(self.runtime, region, fid, dtype, shape, own=False)
            # If we need a transform then compute that now
            if need_transform:
                transform = AffineTransform(len(shape), len(shape), True)
                for idx in xrange(bounds.rect.dim):
                    transform.offset[idx] = bounds.rect[idx]
            else:
                transform = None
            region_field = RegionField(
                self, region, field, shape, transform=transform
            )
        else:
            raise NotImplementedError(
                "legate.numpy needs to handle " + "subregion legate thunk case"
            )
        return region_field

    @staticmethod
    def has_external_attachment(array):
        assert array.base is None or not isinstance(array.base, np.ndarray)
        ptr = long(array.ctypes.data)
        return legate_find_attachment(ptr, array.nbytes) is not None

    def attach_array_field(self, array, share):
        assert array.base is None or not isinstance(array.base, np.ndarray)
        # NumPy arrays are not hashable, so look up the pointer for the array
        # which should be unique for all root NumPy arrays
        ptr = long(array.ctypes.data)
        result = legate_find_attachment(ptr, array.nbytes)
        if result is not None:
            region_field = self.instantiate_region_field(
                result[0], result[1], array.dtype
            )
        else:
            region_field = self.allocate_field(array.shape, array.dtype)
        # Now do the attachment
        region_field.attach_numpy_array(array, share)
        # Tell Legate about the attachment so that we have it for the future
        legate_add_attachment(
            ptr, array.nbytes, region_field.region, region_field.field.field_id
        )
        return region_field

    def detach_array_field(self, array, field, detach, defer):
        if defer:
            # If we need to defer this until later do that now
            if self.deferred_detachments is None:
                self.deferred_detachments = list()
            self.deferred_detachments.append((array, field, detach))
            return
        future = self.dispatch(detach)
        # Dangle a reference to the field off the future to prevent the
        # field from being recycled until the detach is done
        future.field_reference = field
        assert array.base is None
        # Remove the region field from the ptr_to_thunk
        # NumPy arrays are not hashable, so look up the pointer for the array
        # which should be unique for all root NumPy arrays
        ptr = long(array.ctypes.data)
        if ptr in self.ptr_to_thunk:
            del self.ptr_to_thunk[ptr]
        # We also need to tell the core legate library that this array
        # is no longer attached
        legate_remove_attachment(ptr, array.nbytes)
        # If the future is already ready, then no need to track it
        if future.is_ready():
            return
        if not self.pending_detachments:
            self.pending_detachments = dict()
        self.pending_detachments[future] = array

    @staticmethod
    def compute_parent_child_mapping(array):
        # We need an algorithm for figuring out how to compute the
        # slice object that was used to generate a child array from
        # a parent array so we can build the same mapping from a
        # logical region to a subregion
        parent_ptr = long(array.base.ctypes.data)
        child_ptr = long(array.ctypes.data)
        assert child_ptr >= parent_ptr
        ptr_diff = child_ptr - parent_ptr
        parent_shape = array.base.shape
        div = (
            reduce(lambda x, y: x * y, parent_shape)
            if len(parent_shape) > 1
            else parent_shape[0]
        )
        div *= array.dtype.itemsize
        offsets = list()
        # Compute the offsets in the parent index
        for n in parent_shape:
            mod = div
            div //= n
            offsets.append((ptr_diff % mod) // div)
        assert div == array.dtype.itemsize
        # Now build the view and dimmap for the parent to create the view
        key = ()
        view = ()
        dim_map = ()
        child_idx = 0
        child_strides = tuple(array.strides)
        parent_strides = tuple(array.base.strides)
        for idx in xrange(array.base.ndim):
            # Handle the adding and removing dimension cases
            if parent_strides[idx] == 0:
                # This was an added dimension in the parent
                if child_strides[child_idx] == 0:
                    # Kept an added dimension
                    key += (slice(None, None, None),)
                    view += (slice(None, None, None),)
                    dim_map += (0,)
                else:
                    # Removed an added dimension
                    key += (slice(None, None, None),)
                    view += (slice(0, 1, 1),)
                    dim_map += (-1,)
                child_idx += 1
                continue
            elif child_idx == array.ndim:
                key += (slice(offsets[idx], offsets[idx] + 1, 1),)
                view += (slice(offsets[idx], offsets[idx] + 1, 1),)
                dim_map += (-1,)
                continue
            elif child_strides[child_idx] == 0:
                # Added dimension in the child not in the parent
                while child_strides[child_idx] == 0:
                    key += (np.newaxis,)
                    view += (slice(0, 1, 1),)
                    dim_map += (1,)
                    child_idx += 1
                # Fall through to the base case
            # Stides in the child should always be greater than or equal
            # to the strides in the parent, if they're not, then that
            # must be an added dimension
            start = offsets[idx]
            if child_strides[child_idx] < parent_strides[idx]:
                key += (slice(start, start + 1, 1),)
                view += (slice(start, start + 1, 1),)
                dim_map += (-1,)
                # Doesn't count against the child_idx
            else:
                stride = child_strides[child_idx] // parent_strides[idx]
                stop = start + stride * array.shape[child_idx]
                key += (slice(start, stop, stride),)
                view += (slice(start, stop, stride),)
                dim_map += (0,)
                child_idx += 1
        assert child_idx <= array.ndim
        if child_idx < array.ndim:
            return None, None, None
        else:
            return key, view, dim_map

    def find_or_create_array_thunk(
        self, array, stacklevel, share=False, defer=False
    ):
        assert isinstance(array, np.ndarray)
        # We have to be really careful here to handle the case of
        # aliased numpy arrays that are passed in from the application
        # In case of aliasing we need to make sure that they are
        # mapped to the same logical region. The way we handle this
        # is to always create the thunk for the root array and
        # then create sub-thunks that mirror the array views
        if array.base is not None and isinstance(array.base, np.ndarray):
            key, view, dim_map = self.compute_parent_child_mapping(array)
            if key is None:
                # This base array wasn't made with a view
                if not share:
                    return self.find_or_create_array_thunk(
                        array.copy(),
                        stacklevel=(stacklevel + 1),
                        share=False,
                        defer=defer,
                    )
                raise NotImplementedError(
                    "legate.numpy does not currently know "
                    + "how to attach to array views that are not affine "
                    + "transforms of their parent array."
                )
            parent_thunk = self.find_or_create_array_thunk(
                array.base,
                stacklevel=(stacklevel + 1),
                share=share,
                defer=defer,
            )
            # Don't store this one in the ptr_to_thunk as we only want to
            # store the root ones
            return parent_thunk.get_item(
                key=key,
                stacklevel=(stacklevel + 1),
                view=view,
                dim_map=dim_map,
            )
        elif array.size == 0:
            # We always store completely empty arrays with eager thunks
            assert not defer
            return EagerArray(self, array)
        # Get the pointer for this array so we can look it up
        ptr = long(array.ctypes.data)
        if self.ptr_to_thunk is not None and ptr in self.ptr_to_thunk:
            cached_thunk = self.ptr_to_thunk[ptr]
            if not defer or isinstance(cached_thunk, DeferredArray):
                return cached_thunk
        # Once it's a normal numpy array we can make it into one of our arrays
        # Check to see if it is a type that we support for doing deferred
        # execution and big enough to be worth off-loading onto Legion
        if self.is_supported_type(array.dtype) and (
            defer
            or not self.is_eager_shape(array.shape)
            or self.has_external_attachment(array)
        ):
            if array.size == 1 and not share:
                # This is a single value array
                result = self.create_future(
                    array.data,
                    array.nbytes,
                    wrap=True,
                    dtype=array.dtype,
                    shape=array.shape,
                )
                # We didn't attach to this so we don't need to save it
                return result
            else:
                # This is not a scalar so make a field
                region_field = self.attach_array_field(array, share)
                result = DeferredArray(
                    self,
                    region_field,
                    shape=array.shape,
                    dtype=array.dtype,
                    scalar=(array.size == 1),
                )
            # If we're doing shadow debug make an EagerArray shadow
            if self.shadow_debug:
                result.shadow = EagerArray(self, array.copy())
        else:
            assert not defer
            # Make this into an eager evaluated thunk
            result = EagerArray(self, array)
        # Store the result in a weakvalue dictionary so that we can track
        # whether the tree is still alive or not
        if self.ptr_to_thunk is None:
            self.ptr_to_thunk = weakref.WeakValueDictionary()
        self.ptr_to_thunk[ptr] = result
        return result

    def create_empty_thunk(self, shape, dtype, inputs=None):
        # Convert to a tuple type if necessary
        if type(shape) == int:
            shape = (shape,)
        if self.is_supported_type(dtype) and not (
            self.is_eager_shape(shape) and self.are_all_eager_inputs(inputs)
        ):
            if len(shape) == 0:
                # Empty tuple
                result = DeferredArray(
                    self, Future(), shape=(), dtype=dtype, scalar=True
                )
            else:
                volume = reduce(lambda x, y: x * y, shape)
                if volume == 1:
                    result = DeferredArray(
                        self, Future(), shape=shape, dtype=dtype, scalar=True
                    )
                else:
                    region_field = self.allocate_field(shape, dtype)
                    result = DeferredArray(
                        self,
                        region_field,
                        shape=shape,
                        dtype=dtype,
                        scalar=False,
                    )
            # If we're doing shadow debug make an EagerArray shadow
            if self.shadow_debug:
                result.shadow = EagerArray(self, np.empty(shape, dtype=dtype))
            return result
        else:
            return EagerArray(self, np.empty(shape, dtype=dtype))

    def is_eager_shape(self, shape):
        volume = calculate_volume(shape)
        # Empty arrays are ALWAYS eager
        if volume == 0:
            return True
        # If we're testing then the answer is always no
        if self.test_mode:
            return False
        # Note the off by 1 case here, technically arrays with size
        # up to LEGATE_MAX_DIM inclusive should be allowed, but we
        # often use an extra dimension for reductions in legate.numpy
        if len(shape) >= LEGATE_MAX_DIM:
            return True
        if len(shape) == 0:
            return self.max_eager_volume > 0
        # See if the volume is large enough
        return volume <= self.max_eager_volume

    @staticmethod
    def are_all_eager_inputs(inputs):
        if inputs is None:
            return True
        for inp in inputs:
            assert isinstance(inp, NumPyThunk)
            if not isinstance(inp, EagerArray):
                return False
        return True

    @staticmethod
    def is_eager_array(array):
        return isinstance(array, EagerArray)

    @staticmethod
    def is_deferred_array(array):
        return isinstance(array, DeferredArray)

    @staticmethod
    def is_lazy_array(array):
        return isinstance(array, LazyArray)

    def to_eager_array(self, array, stacklevel):
        if self.is_eager_array(array):
            return array
        elif self.is_deferred_array(array):
            raise NotImplementedError("convert deferred array to eager array")
        elif self.is_lazy_array(array):
            raise NotImplementedError("convert lazy array to eager array")
        else:
            raise RuntimeError("invalid array type")

    def to_deferred_array(self, array, stacklevel):
        if self.is_deferred_array(array):
            return array
        elif self.is_eager_array(array):
            return array.to_deferred_array(stacklevel=(stacklevel + 1))
        elif self.is_lazy_array(array):
            raise NotImplementedError("convert lazy array to deferred array")
        else:
            raise RuntimeError("invalid array type")

    def to_lazy_array(self, array, stacklevel):
        if self.is_lazy_array(array):
            return array
        elif self.is_deferred_array(array):
            raise NotImplementedError("convert deferred array to lazy array")
        elif self.is_eager_array(array):
            raise NotImplementedError("convert eager array to lazy array")
        else:
            raise RuntimeError("invalid array type")

    def get_nullary_task_id(
        self, op_code, result_type, variant_code=NumPyVariantCode.NORMAL
    ):
        assert isinstance(op_code, NumPyOpCode)
        assert isinstance(variant_code, NumPyVariantCode)
        return (
            self.first_task_id
            + op_code.value * NUMPY_TYPE_OFFSET
            + variant_code.value
            + numpy_field_type_offsets[result_type.type] * NUMPY_MAX_VARIANTS
        )

    def get_unary_task_id(
        self,
        op_code,
        result_type,
        argument_type,
        variant_code=NumPyVariantCode.NORMAL,
    ):
        assert isinstance(op_code, NumPyOpCode)
        assert isinstance(variant_code, NumPyVariantCode)

        # CONVERT's ID is special
        if op_code == NumPyOpCode.CONVERT:
            return (
                self.first_task_id
                + legate_numpy.NUMPY_CONVERT_OFFSET
                + numpy_field_type_offsets[result_type.type]
                * NUMPY_TYPE_OFFSET
                + variant_code.value
                + numpy_field_type_offsets[argument_type.type]
                * NUMPY_MAX_VARIANTS
            )

        # unary tasks distinguish themselves by argument_type
        return (
            self.first_task_id
            + op_code.value * NUMPY_TYPE_OFFSET
            + variant_code.value
            + numpy_field_type_offsets[argument_type.type] * NUMPY_MAX_VARIANTS
        )

    def get_binary_task_id(
        self,
        op_code,
        result_type,
        first_argument_type,
        second_argument_type,
        variant_code=NumPyVariantCode.NORMAL,
    ):
        assert isinstance(op_code, NumPyOpCode)
        assert isinstance(variant_code, NumPyVariantCode)

        # binary tasks distinguish themselves by first_argument_type
        return (
            self.first_task_id
            + op_code.value * NUMPY_TYPE_OFFSET
            + variant_code.value
            + numpy_field_type_offsets[first_argument_type.type]
            * NUMPY_MAX_VARIANTS
        )

    @staticmethod
    def get_binary_task_variant_code(inputs, output):
        broadcast = False
        inplace = False
        for x in inputs:
            if x is output:
                inplace = True
            if x.size == 1:
                broadcast = True
        if inplace and broadcast:
            return NumPyVariantCode.INPLACE_BROADCAST
        if inplace:
            return NumPyVariantCode.INPLACE
        if broadcast:
            return NumPyVariantCode.BROADCAST
        return NumPyVariantCode.NORMAL

    def get_ternary_task_id(self, op_code, result_type, variant_code):
        assert isinstance(op_code, NumPyOpCode)
        assert isinstance(variant_code, NumPyVariantCode)

        # ternary tasks distinguish themselves by result_type
        # XXX check this once we have more ternary ops in addition to WHERE
        return (
            self.first_task_id
            + op_code.value * NUMPY_TYPE_OFFSET
            + variant_code.value
            + numpy_field_type_offsets[result_type.type] * NUMPY_MAX_VARIANTS
        )

    def get_weighted_bincount_task_id(self, dt1, dt2):
        result = self.first_task_id + legate_numpy.NUMPY_BINCOUNT_OFFSET
        result += numpy_field_type_offsets[dt1.type] * NUMPY_TYPE_OFFSET
        result += numpy_field_type_offsets[dt2.type] * NUMPY_MAX_VARIANTS
        return result

    def get_reduction_op_id(self, op, field_dtype):
        redop_id = numpy_reduction_op_offsets[op]
        if redop_id < legion.LEGION_REDOP_KIND_TOTAL:
            # This is a built-in legion op-code
            result = (
                legion.LEGION_REDOP_BASE + redop_id * legion.LEGION_TYPE_TOTAL
            )
            result += numpy_field_type_offsets[field_dtype.type]
        else:
            # this is a custom numpy reduction
            result = self.first_redop_id + redop_id * NUMPY_MAX_TYPES
            result += numpy_field_type_offsets[field_dtype.type]
        return result

    def get_radix_projection_functor_id(
        self, total_dims, collapse_dim, radix, offset
    ):
        assert offset < radix
        assert collapse_dim < total_dims
        if total_dims == 2:
            assert collapse_dim == 0 or collapse_dim == 1
            if radix == 4:
                if offset == 0:
                    return self.first_proj_id + (
                        NumPyProjCode.PROJ_RADIX_2D_X_4_0
                        if collapse_dim == 0
                        else NumPyProjCode.PROJ_RADIX_2D_Y_4_0
                    )
                elif offset == 1:
                    return self.first_proj_id + (
                        NumPyProjCode.PROJ_RADIX_2D_X_4_1
                        if collapse_dim == 0
                        else NumPyProjCode.PROJ_RADIX_2D_Y_4_1
                    )
                elif offset == 2:
                    return self.first_proj_id + (
                        NumPyProjCode.PROJ_RADIX_2D_X_4_2
                        if collapse_dim == 0
                        else NumPyProjCode.PROJ_RADIX_2D_Y_4_2
                    )
                else:
                    return self.first_proj_id + (
                        NumPyProjCode.PROJ_RADIX_2D_X_4_3
                        if collapse_dim == 0
                        else NumPyProjCode.PROJ_RADIX_2D_Y_4_3
                    )
            else:
                raise NotImplementedError(
                    "Need radix projection functor for radix "
                    + str(radix)
                    + "in two dimensions"
                )
        elif total_dims == 3:
            assert collapse_dim >= 0 and collapse_dim <= 2
            if radix == 4:
                if offset == 0:
                    return self.first_proj_id + (
                        NumPyProjCode.PROJ_RADIX_3D_X_4_0
                        if collapse_dim == 0
                        else NumPyProjCode.PROJ_RADIX_3D_Y_4_0
                        if collapse_dim == 1
                        else NumPyProjCode.PROJ_RADIX_3D_Z_4_0
                    )
                elif offset == 1:
                    return self.first_proj_id + (
                        NumPyProjCode.PROJ_RADIX_3D_X_4_1
                        if collapse_dim == 0
                        else NumPyProjCode.PROJ_RADIX_3D_Y_4_1
                        if collapse_dim == 1
                        else NumPyProjCode.PROJ_RADIX_3D_Z_4_1
                    )
                elif offset == 2:
                    return self.first_proj_id + (
                        NumPyProjCode.PROJ_RADIX_3D_X_4_2
                        if collapse_dim == 0
                        else NumPyProjCode.PROJ_RADIX_3D_Y_4_2
                        if collapse_dim == 1
                        else NumPyProjCode.PROJ_RADIX_3D_Z_4_2
                    )
                else:
                    return self.first_proj_id + (
                        NumPyProjCode.PROJ_RADIX_3D_X_4_3
                        if collapse_dim == 0
                        else NumPyProjCode.PROJ_RADIX_3D_Y_4_3
                        if collapse_dim == 1
                        else NumPyProjCode.PROJ_RADIX_3D_Z_4_3
                    )
            else:
                raise NotImplementedError(
                    "Need radix projection functor for radix "
                    + str(radix)
                    + "in three dimensions"
                )
        else:
            raise NotImplementedError(
                "Need radix projection functor for dim " + str(total_dims)
            )

    def compute_broadcast_transform(self, output_shape, input_shape):
        output_ndim = len(output_shape)
        input_ndim = len(input_shape)
        assert output_shape != input_shape
        assert output_ndim >= input_ndim
        transform = np.zeros((input_ndim, output_ndim), dtype=np.int64)
        offset = np.zeros((input_ndim,), dtype=np.int64)
        input_dim = 0
        broadcast_dims = ()
        start_dim = output_ndim - input_ndim
        for dim in range(start_dim, output_ndim):
            if input_shape[input_dim] == output_shape[dim]:
                transform[input_dim, dim] = 1
            else:
                assert input_shape[input_dim] == 1
                broadcast_dims = broadcast_dims + (input_dim,)
                offset[input_dim] = 1
            input_dim += 1
        if input_ndim == 1:
            if output_ndim == 2:
                return (
                    transform,
                    offset,
                    self.first_proj_id + NumPyProjCode.PROJ_2D_1D_Y,
                )
            elif output_ndim == 3:
                return (
                    transform,
                    offset,
                    self.first_proj_id + NumPyProjCode.PROJ_3D_1D_Z,
                )
        elif input_ndim == 2:
            if output_ndim == 2:
                assert len(broadcast_dims) == 1
                if broadcast_dims[0] == 0:
                    return (
                        transform,
                        offset,
                        self.first_proj_id + NumPyProjCode.PROJ_2D_2D_Y,
                    )
                else:
                    assert broadcast_dims[0] == 1
                    return (
                        transform,
                        offset,
                        self.first_proj_id + NumPyProjCode.PROJ_2D_2D_X,
                    )
            else:
                assert output_ndim == 3
                return (
                    transform,
                    offset,
                    self.first_proj_id + NumPyProjCode.PROJ_3D_2D_YZ,
                )
        elif input_ndim == 3:
            if len(broadcast_dims) == 1:
                if broadcast_dims[0] == 0:
                    return (
                        transform,
                        offset,
                        self.first_proj_id + NumPyProjCode.PROJ_3D_3D_YZ,
                    )
                elif broadcast_dims[0] == 1:
                    return (
                        transform,
                        offset,
                        self.first_proj_id + NumPyProjCode.PROJ_3D_3D_XZ,
                    )
                else:
                    assert broadcast_dims[0] == 2
                    return (
                        transform,
                        offset,
                        self.first_proj_id + NumPyProjCode.PROJ_3D_3D_XY,
                    )
            else:
                assert len(broadcast_dims) == 2
                if broadcast_dims == (0, 1):
                    return (
                        transform,
                        offset,
                        self.first_proj_id + NumPyProjCode.PROJ_3D_3D_Z,
                    )
                elif broadcast_dims == (1, 2):
                    return (
                        transform,
                        offset,
                        self.first_proj_id + NumPyProjCode.PROJ_3D_3D_X,
                    )
                else:
                    assert broadcast_dims == (0, 2)
                    return (
                        transform,
                        offset,
                        self.first_proj_id + NumPyProjCode.PROJ_3D_3D_Y,
                    )
        else:
            raise NotImplementedError(
                "Legate needs support for more than 3 dimensions"
            )

    def get_reduction_transform(self, input_shape, output_shape, axes):
        input_ndim = len(input_shape)
        output_ndim = len(output_shape)
        assert len(axes) > 0
        # In the case where we keep dimensions the arrays can be the same size
        if output_ndim == input_ndim:
            # The transform in this case is just identity transform
            transform = np.zeros((output_ndim, input_ndim), dtype=np.int64)
            for dim in xrange(input_ndim):
                if dim in axes:
                    continue
                transform[dim, dim] = 1
            assert input_ndim > 1  # Should never have the 1-D case here
            if input_ndim == 2:
                assert len(axes) == 1
                if axes[0] == 0:
                    return (
                        transform,
                        self.first_proj_id + NumPyProjCode.PROJ_2D_2D_Y,
                    )
                else:
                    assert axes[0] == 1
                    return (
                        transform,
                        self.first_proj_id + NumPyProjCode.PROJ_2D_2D_X,
                    )
            elif input_ndim == 3:
                if len(axes) == 1:
                    if axes[0] == 0:
                        return (
                            transform,
                            self.first_proj_id + NumPyProjCode.PROJ_3D_3D_YZ,
                        )
                    elif axes[0] == 1:
                        return (
                            transform,
                            self.first_proj_id + NumPyProjCode.PROJ_3D_3D_XZ,
                        )
                    else:
                        assert axes[0] == 2
                        return (
                            transform,
                            self.first_proj_id + NumPyProjCode.PROJ_3D_3D_XY,
                        )
                else:
                    assert len(axes) == 2
                    if axes == (0, 1):
                        return (
                            transform,
                            self.first_proj_id + NumPyProjCode.PROJ_3D_3D_Z,
                        )
                    elif axes == (0, 2):
                        return (
                            transform,
                            self.first_proj_id + NumPyProjCode.PROJ_3D_3D_Y,
                        )
                    else:
                        assert axes == (1, 2)
                        return (
                            transform,
                            self.first_proj_id + NumPyProjCode.PROJ_3D_3D_X,
                        )
            else:
                raise NotImplementedError(
                    "Legate needs support for more than 3 dimensions"
                )
        else:
            # This is where we don't keep the dimensions
            assert output_ndim + len(axes) == input_ndim
            transform = np.zeros((output_ndim, input_ndim), dtype=np.int64)
            output_dim = 0
            for dim in xrange(input_ndim):
                if dim in axes:
                    continue
                transform[output_dim, dim] = 1
                output_dim += 1
            if input_ndim == 2:
                assert len(axes) == 1
                if axes[0] == 0:
                    return (
                        transform,
                        self.first_proj_id + NumPyProjCode.PROJ_2D_1D_Y,
                    )
                else:
                    assert axes[0] == 1
                    return (
                        transform,
                        self.first_proj_id + NumPyProjCode.PROJ_2D_1D_X,
                    )
            elif input_ndim == 3:
                if len(axes) == 1:
                    if axes[0] == 0:
                        return (
                            transform,
                            self.first_proj_id + NumPyProjCode.PROJ_3D_2D_YZ,
                        )
                    elif axes[0] == 1:
                        return (
                            transform,
                            self.first_proj_id + NumPyProjCode.PROJ_3D_2D_XZ,
                        )
                    else:
                        assert axes[0] == 2
                        return (
                            transform,
                            self.first_proj_id + NumPyProjCode.PROJ_3D_2D_XY,
                        )
                else:
                    assert len(axes) == 2
                    if axes == (0, 1):
                        return (
                            transform,
                            self.first_proj_id + NumPyProjCode.PROJ_3D_1D_Z,
                        )
                    elif axes == (0, 2):
                        return (
                            transform,
                            self.first_proj_id + NumPyProjCode.PROJ_3D_1D_Y,
                        )
                    else:
                        assert axes == (1, 2)
                        return (
                            transform,
                            self.first_proj_id + NumPyProjCode.PROJ_3D_1D_X,
                        )
            else:
                raise NotImplementedError(
                    "Legate needs support for more than 3 dimensions"
                )

    def check_shadow(self, thunk, op):
        assert thunk.shadow is not None
        # Check the kind of this array and see if we should use allclose or
        # array_equal
        if thunk.dtype.kind == "f":
            if not np.allclose(
                thunk.__numpy_array__, thunk.shadow.__numpy_array__
            ):
                raise RuntimeError("Shadow array check failed for " + op)
        else:
            if not np.array_equal(
                thunk.__numpy_array__, thunk.shadow.__numpy_array__
            ):
                raise RuntimeError("Shadow array check failed for " + op)


runtime = Runtime(get_legion_runtime(), get_legion_context())
