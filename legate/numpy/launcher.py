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

from collections import OrderedDict
from enum import IntEnum

import numpy as np

from legate.core import ArgumentMap, BufferBuilder, IndexTask, Task


class Permission(IntEnum):
    NO_ACCESS = 0
    READ = 1
    WRITE = 2
    READ_WRITE = 3


class ScalarArg(object):
    _serializers = {
        bool: BufferBuilder.pack_bool,
        np.int8: BufferBuilder.pack_8bit_int,
        np.int16: BufferBuilder.pack_16bit_int,
        np.int32: BufferBuilder.pack_32bit_int,
        np.int64: BufferBuilder.pack_64bit_int,
        np.uint8: BufferBuilder.pack_8bit_uint,
        np.uint16: BufferBuilder.pack_16bit_uint,
        np.uint32: BufferBuilder.pack_32bit_uint,
        np.uint64: BufferBuilder.pack_64bit_uint,
        np.float32: BufferBuilder.pack_32bit_float,
        np.float64: BufferBuilder.pack_64bit_float,
    }

    def __init__(self, value, dtype):
        self._value = value
        self._dtype = dtype

    def pack(self, buf):
        if self._dtype in self._serializers:
            self._serializers[self._dtype](buf, self._value)
        else:
            raise ValueError("Unsupported data type: %s" % str(self._dtype))


class DtypeArg(object):
    def __init__(self, dtype):
        self._dtype = dtype

    def pack(self, buf):
        buf.pack_dtype(self._dtype)


class PointArg(object):
    def __init__(self, point):
        self._point = point

    def pack(self, buf):
        buf.pack_point(self._point)


class RegionFieldArg(object):
    def __init__(self, dim, region_idx, field_id, dtype, transform):
        self._dim = dim
        self._region_idx = region_idx
        self._field_id = field_id
        self._dtype = dtype
        self._transform = transform

    def pack(self, buf):
        buf.pack_32bit_int(self._dim)
        buf.pack_dtype(self._dtype)
        buf.pack_32bit_uint(self._region_idx)
        buf.pack_32bit_uint(self._field_id)
        if self._transform is not None:
            buf.pack_32bit_int(self._transform.M)
            buf.pack_32bit_int(self._transform.N)
            for x in range(0, self._transform.M):
                for y in range(0, self._transform.N):
                    buf.pack_64bit_int(self._transform.trans[x, y])
            for x in range(0, self._transform.M):
                buf.pack_64bit_int(self._transform.offset[x])
        else:
            buf.pack_32bit_int(-1)


_single_task_calls = {
    Permission.NO_ACCESS: Task.add_no_access_requirement,
    Permission.READ: Task.add_read_requirement,
    Permission.WRITE: Task.add_write_requirement,
    Permission.READ_WRITE: Task.add_read_write_requirement,
}

_index_task_calls = {
    Permission.NO_ACCESS: IndexTask.add_no_access_requirement,
    Permission.READ: IndexTask.add_read_requirement,
    Permission.WRITE: IndexTask.add_write_requirement,
    Permission.READ_WRITE: IndexTask.add_read_write_requirement,
}


class _Broadcast(object):
    def add(self, runtime, task, arg, fields):
        f = _index_task_calls[arg.permission]
        f(task, arg.region, fields, 0, parent=arg.region, tag=arg.tag)

    def add_single(self, task, arg, fields):
        f = _single_task_calls[arg.permission]
        f(task, arg.region, fields, tag=arg.tag, flags=arg.flags)

    def __hash__(self):
        return hash("Broadcast")


Broadcast = _Broadcast()


class Projection(object):
    def __init__(self, part, proj=0):
        self.part = part
        self.proj = proj

    def add(self, runtime, task, arg, fields):
        f = _index_task_calls[arg.permission]
        f(task, self.part, fields, self.proj, tag=arg.tag, flags=arg.flags)

    def add_single(self, task, arg, fields):
        f = _single_task_calls[arg.permission]
        f(task, arg.region, fields, tag=arg.tag)

    def __hash__(self):
        return hash((self.part, self.proj))


class RegionReq(object):
    def __init__(self, region, proj, permission, tag, flags):
        self.region = region
        self.proj = proj
        self.permission = permission
        self.tag = tag
        self.flags = flags

    def __repr__(self):
        return (
            str(self.region)
            + ","
            + str(self.proj)
            + ","
            + str(self.permission)
            + ","
            + str(self.tag)
            + ","
            + str(self.flags)
        )

    def __hash__(self):
        return hash(
            (self.region, self.proj, self.permission, self.tag, self.flags)
        )

    def __eq__(self, other):
        return (
            self.region == other.region
            and self.proj == other.proj
            and self.permission == other.permission
            and self.tag == other.tag
            and self.flags == other.flags
        )


class Map(object):
    def __init__(self, runtime, task_id, tag=0):
        assert type(tag) != bool
        self._runtime = runtime
        self._task_id = runtime.get_task_id(task_id)
        self._args = list()
        self._region_args = OrderedDict()
        self._next_region_idx = 0
        self._projections = list()
        self._future_args = list()
        self._future_map_args = list()
        self._tag = tag
        self._sharding_space = None

    def __del__(self):
        self._region_args.clear()
        self._projections.clear()
        self._future_args.clear()
        self._future_map_args.clear()

    def _add_region_arg(self, region_arg, field_id):
        if region_arg not in self._region_args:
            idx = self._next_region_idx
            self._next_region_idx += 1
            self._region_args[region_arg] = ([field_id], idx)
            return idx
        else:
            (fields, idx) = self._region_args[region_arg]
            if field_id not in fields:
                fields.append(field_id)
            return idx

    def add_scalar_arg(self, value, dtype):
        self._args.append(ScalarArg(value, dtype))

    def add_dtype_arg(self, dtype):
        self._args.append(DtypeArg(dtype))

    def add_region_arg(self, store, transform, proj, perm, tag, flags):
        (region, field_id) = store.storage
        region_idx = self._add_region_arg(
            RegionReq(region, proj, perm, tag, flags), field_id
        )

        self._args.append(
            RegionFieldArg(
                region.index_space.get_dim(),
                region_idx,
                field_id,
                store.type.to_pandas_dtype(),
                transform,
            )
        )

    def add_no_access(self, store, transform, proj, tag=0, flags=0):
        self.add_region_arg(
            store, transform, proj, Permission.NO_ACCESS, tag, flags
        )

    def add_input(self, store, transform, proj, tag=0, flags=0):
        self.add_region_arg(
            store, transform, proj, Permission.READ, tag, flags
        )

    def add_output(self, store, transform, proj, tag=0, flags=0):
        self.add_region_arg(
            store, transform, proj, Permission.WRITE, tag, flags
        )

    def add_inout(self, store, transform, proj, tag=0, flags=0):
        self.add_region_arg(
            store, transform, proj, Permission.READ_WRITE, tag, flags
        )

    def add_future(self, future):
        self._future_args.append(future)

    def add_future_map(self, future_map):
        self._future_map_args.append(future_map)

    def add_point(self, point):
        self._args.append(PointArg(point))

    def add_shape(self, shape, chunk_shape=None, proj=None):
        assert chunk_shape is None or len(shape) == len(chunk_shape)
        self.add_scalar_arg(len(shape), np.int32)
        self.add_point(shape)
        if chunk_shape is not None:
            assert proj is not None
            self.add_scalar_arg(proj, np.int32)
            self.add_point(chunk_shape)
        else:
            assert proj is None
            self.add_scalar_arg(-1, np.int32)

    def set_sharding_space(self, space):
        self._sharding_space = space

    def build_task(self, launch_domain, argbuf):
        for arg in self._args:
            arg.pack(argbuf)
        task = IndexTask(
            self._task_id,
            launch_domain,
            self._runtime.empty_argmap,
            argbuf.get_string(),
            argbuf.get_size(),
            mapper=self._runtime.mapper_id,
            tag=self._tag,
        )
        if self._sharding_space is not None:
            task.set_sharding_space(self._sharding_space)

        for region_arg, (fields, _) in self._region_args.items():
            region_arg.proj.add(self._runtime, task, region_arg, fields)
        for future in self._future_args:
            task.add_future(future)
        for future_map in self._future_map_args:
            task.add_point_future(ArgumentMap(future_map=future_map))
        return task

    def build_single_task(self, argbuf):
        for arg in self._args:
            arg.pack(argbuf)
        task = Task(
            self._task_id,
            argbuf.get_string(),
            argbuf.get_size(),
            mapper=self._runtime.mapper_id,
            tag=self._tag,
        )
        for region_arg, (fields, _) in self._region_args.items():
            region_arg.proj.add_single(task, region_arg, fields)
        for future in self._future_args:
            task.add_future(future)
        if len(self._region_args) == 0:
            task.set_local_function(True)
        return task

    def execute(self, launch_domain):
        # Note that we should hold a reference to this buffer
        # until we launch a task, otherwise the Python GC will
        # collect the Python object holding the buffer, which
        # in turn will deallocate the C side buffer.
        argbuf = BufferBuilder()
        task = self.build_task(launch_domain, argbuf)
        return self._runtime.dispatch(task)

    def execute_single(self):
        argbuf = BufferBuilder()
        return self._runtime.dispatch(self.build_single_task(argbuf))
