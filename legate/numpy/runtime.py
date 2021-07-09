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

import inspect
import struct
import sys
from functools import reduce

import numpy as np

import legate.core.types as ty
from legate.core import (
    LEGATE_MAX_DIM,
    AffineTransform,
    FieldID,
    Future,
    FutureMap,
    Region,
    legion,
)
from legate.core.runtime import RegionField

from .config import *  # noqa F403
from .deferred import DeferredArray
from .eager import EagerArray
from .lazy import LazyArray
from .thunk import NumPyThunk
from .utils import calculate_volume, get_arg_dtype, get_arg_value_dtype


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


_supported_dtypes = {
    np.bool_: ty.bool_,
    np.int8: ty.int8,
    np.int16: ty.int16,
    np.int32: ty.int32,
    np.int: ty.int64,
    np.int64: ty.int64,
    np.uint8: ty.uint8,
    np.uint16: ty.uint16,
    np.uint32: ty.uint32,
    np.uint: ty.uint64,
    np.uint64: ty.uint64,
    np.float16: ty.float16,
    np.float32: ty.float32,
    np.float: ty.float64,
    np.float64: ty.float64,
    np.complex64: ty.complex64,
    np.complex128: ty.complex128,
}


class Runtime(object):
    __slots__ = [
        "legate_context",
        "legate_runtime",
        "current_random_epoch",
        "max_eager_volume",
        "test_mode",
        "shadow_debug",
        "callsite_summaries",
        "first_task_id",
        "mapper_id",
        "first_redop_id",
        "first_proj_id",
        "destroyed",
    ]

    def __init__(self, legate_context):
        self.legate_context = legate_context
        self.legate_runtime = get_legate_runtime()
        self.current_random_epoch = 0
        self.destroyed = False

        # Get the initial task ID and mapper ID
        self.mapper_id = legate_context.first_mapper_id
        self.first_redop_id = legate_context.first_redop_id

        self.max_eager_volume = self.legate_context.get_tunable(
            NumPyTunable.MAX_EAGER_VOLUME,
            ty.int32,
        )

        # Make sure that our NumPyLib object knows about us so it can destroy
        # us
        numpy_lib.set_runtime(self)
        self._register_dtypes()
        self._parse_command_args()

    def _register_dtypes(self):
        type_system = self.legate_context.type_system
        for numpy_type, core_type in _supported_dtypes.items():
            type_system.make_alias(np.dtype(numpy_type), core_type)

    def _parse_command_args(self):
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

    def get_arg_dtype(self, value_dtype):
        arg_dtype = get_arg_dtype(value_dtype)
        type_system = self.legate_context.type_system
        if arg_dtype not in type_system:
            # We assign T's type code to Argval<T>
            code = type_system[value_dtype].code
            type_system.add_type(arg_dtype, arg_dtype.itemsize, code)
        return arg_dtype

    def destroy(self):
        assert not self.destroyed
        if self.callsite_summaries is not None:
            num_gpus = self.legate_context.get_tunable(
                NumPyTunable.NUM_GPUS,
                ty.int32,
            )
            print(
                "---------------- Legate.NumPy Callsite Summaries "
                "----------------"
            )
            for callsite, counts in sorted(
                self.callsite_summaries.items(),
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

    def create_scalar(self, array, dtype, shape=None, wrap=False):
        if dtype.kind == "V":
            is_arg = True
            code = numpy_field_type_offsets[get_arg_value_dtype(dtype)]
        else:
            is_arg = False
            code = numpy_field_type_offsets[dtype.type]
        data = array.tobytes()
        buf = struct.pack(f"ii{len(data)}s", int(is_arg), code, data)
        future = self.legate_runtime.create_future(buf, len(buf))
        if wrap:
            assert all(extent == 1 for extent in shape)
            assert shape is not None
            store = self.legate_context.create_store(
                dtype,
                shape=shape,
                storage=future,
                optimize_scalar=True,
            )
            result = DeferredArray(self, store, dtype=dtype, scalar=True)
            if self.shadow_debug:
                result.shadow = EagerArray(self, np.array(array))
        else:
            result = future
        return result

    def set_next_random_epoch(self, epoch):
        self.current_random_epoch = epoch

    def get_next_random_epoch(self):
        result = self.current_random_epoch
        self.current_random_epoch += 1
        return result

    def is_supported_type(self, dtype):
        return np.dtype(dtype) in self.legate_context.type_system

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
            for idx in range(bounds.rect.dim):
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
                for idx in range(bounds.rect.dim):
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

    def has_external_attachment(self, array):
        assert array.base is None or not isinstance(array.base, np.ndarray)
        return self.legate_runtime.has_attachment(array)

    @staticmethod
    def compute_parent_child_mapping(array):
        # We need an algorithm for figuring out how to compute the
        # slice object that was used to generate a child array from
        # a parent array so we can build the same mapping from a
        # logical region to a subregion
        parent_ptr = int(array.base.ctypes.data)
        child_ptr = int(array.ctypes.data)
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
        for idx in range(array.base.ndim):
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
                result = self.create_scalar(
                    array.data,
                    array.dtype,
                    array.shape,
                    wrap=True,
                )
                # We didn't attach to this so we don't need to save it
                return result
            else:
                # This is not a scalar so make a field
                region_field = self.legate_context.attach_array(
                    array, array.dtype, share
                )
                result = DeferredArray(
                    self,
                    region_field,
                    dtype=array.dtype,
                    scalar=(array.size == 1),
                )
            # If we're doing shadow debug make an EagerArray shadow
            if self.shadow_debug:
                result.shadow = EagerArray(self, array.copy(), shadow=True)
        else:
            assert not defer
            # Make this into an eager evaluated thunk
            result = EagerArray(self, array)
        return result

    def create_empty_thunk(self, shape, dtype, inputs=None):
        # Convert to a tuple type if necessary
        if type(shape) == int:
            shape = (shape,)
        if self.is_supported_type(dtype) and not (
            self.is_eager_shape(shape) and self.are_all_eager_inputs(inputs)
        ):
            store = self.legate_context.create_store(
                dtype, shape=shape, optimize_scalar=True
            )
            scalar = store.kind == Future
            result = DeferredArray(self, store, dtype=dtype, scalar=scalar)
            # If we're doing shadow debug make an EagerArray shadow
            if self.shadow_debug:
                result.shadow = EagerArray(
                    self,
                    np.empty(shape, dtype=dtype),
                    shadow=True,
                )
            return result
        else:
            return EagerArray(self, np.empty(shape, dtype=dtype))

    def create_unbound_thunk(self, dtype):
        store = self.legate_context.create_store(dtype, unbound=True)
        return DeferredArray(self, store, dtype=dtype, scalar=False)

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
            return EagerArray(self, array.__numpy_array__())
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

    def get_task_id(self, op_code):
        return self.first_task_id + op_code.value

    def _convert_reduction_op_id(self, redop_id, field_dtype):
        base = (
            legion.LEGION_REDOP_BASE
            if redop_id < legion.LEGION_REDOP_KIND_TOTAL
            else self.first_redop_id
        )
        result = base + redop_id * legion.LEGION_TYPE_TOTAL
        return result + numpy_field_type_offsets[field_dtype.type]

    def get_reduction_op_id(self, op, field_dtype):
        redop_id = numpy_reduction_op_offsets[op]
        return self._convert_reduction_op_id(redop_id, field_dtype)

    def get_unary_reduction_op_id(self, op, field_dtype):
        redop_id = numpy_unary_reduction_op_offsets[op]
        return self._convert_reduction_op_id(redop_id, field_dtype)

    def get_scalar_reduction_op_id(self, op):
        return self.first_redop_id + numpy_scalar_reduction_op_offsets[op]

    def get_reduction_identity(self, op, dtype):
        return numpy_unary_reduction_identities[op](dtype)

    def check_shadow(self, thunk, op):
        assert thunk.shadow is not None
        # Check the kind of this array and see if we should use allclose or
        # array_equal
        legate_result = thunk.__numpy_array__()
        numpy_result = thunk.shadow.__numpy_array__()

        if thunk.dtype.kind == "f":
            passed = np.allclose(legate_result, numpy_result)
        else:
            passed = np.array_equal(legate_result, numpy_result)
        if not passed:
            print("===== Legate =====")
            print(legate_result)
            print("===== NumPy =====")
            print(numpy_result)
            raise RuntimeError(f"Shadow array check failed for {op}")


runtime = Runtime(numpy_context)
