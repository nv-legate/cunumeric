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

import struct
import sys
import warnings
from functools import reduce

import numpy as np

import legate.core.types as ty
from legate.core import LEGATE_MAX_DIM, Rect, get_legate_runtime, legion

from .config import (
    CuNumericOpCode,
    CuNumericRedopCode,
    CuNumericTunable,
    cunumeric_context,
    cunumeric_lib,
)
from .deferred import DeferredArray
from .eager import EagerArray
from .thunk import NumPyThunk
from .utils import calculate_volume, find_last_user_stacklevel, get_arg_dtype

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
        "api_calls",
        "current_random_epoch",
        "destroyed",
        "legate_context",
        "legate_runtime",
        "max_eager_volume",
        "num_gpus",
        "num_procs",
        "preload_cudalibs",
        "report_coverage",
        "report_dump_callstack",
        "report_dump_csv",
        "test_mode",
        "warning",
    ]

    def __init__(self, legate_context):
        self.legate_context = legate_context
        self.legate_runtime = get_legate_runtime()
        self.current_random_epoch = 0
        self.destroyed = False
        self.api_calls = []

        self.max_eager_volume = int(
            self.legate_context.get_tunable(
                CuNumericTunable.MAX_EAGER_VOLUME,
                ty.int32,
            )
        )
        self.num_procs = int(
            self.legate_context.get_tunable(
                CuNumericTunable.NUM_PROCS,
                ty.int32,
            )
        )
        self.num_gpus = int(
            self.legate_context.get_tunable(
                CuNumericTunable.NUM_GPUS,
                ty.int32,
            )
        )

        # Make sure that our CuNumericLib object knows about us so it can
        # destroy us
        cunumeric_lib.set_runtime(self)
        self._register_dtypes()
        self._parse_command_args()
        if self.num_gpus > 0 and self.preload_cudalibs:
            self._load_cudalibs()

    def _register_dtypes(self):
        type_system = self.legate_context.type_system
        for numpy_type, core_type in _supported_dtypes.items():
            type_system.make_alias(np.dtype(numpy_type), core_type)

    def _parse_command_args(self):
        try:
            # Prune it out so the application does not see it
            sys.argv.remove("-cunumeric:test")
            self.test_mode = True
        except ValueError:
            self.test_mode = False
        try:
            # Prune it out so the application does not see it
            sys.argv.remove("-cunumeric:preload-cudalibs")
            self.preload_cudalibs = True
        except ValueError:
            self.preload_cudalibs = False
        try:
            # Prune it out so the application does not see it
            sys.argv.remove("-cunumeric:warn")
            self.warning = True
        except ValueError:
            self.warning = self.test_mode
        try:
            # Prune it out so the application does not see it
            sys.argv.remove("-cunumeric:report:coverage")
            self.report_coverage = True
        except ValueError:
            self.report_coverage = False
        try:
            # Prune it out so the application does not see it
            sys.argv.remove("-cunumeric:report:dump-callstack")
            self.report_dump_callstack = True
        except ValueError:
            self.report_dump_callstack = False
        try:
            # Prune it out so the application does not see it
            idx = sys.argv.index("-cunumeric:report:dump-csv")
            if idx + 1 >= len(sys.argv):
                raise RuntimeError(
                    "Please provide a filename for the reporting"
                )
            self.report_dump_csv = sys.argv[idx + 1]
            sys.argv = sys.argv[:idx] + sys.argv[idx + 2 :]
        except ValueError:
            self.report_dump_csv = None

    def record_api_call(self, name, location, implemented):
        assert self.report_coverage
        self.api_calls.append((name, location, implemented))

    def _load_cudalibs(self):
        task = self.legate_context.create_task(
            CuNumericOpCode.LOAD_CUDALIBS,
            manual=True,
            launch_domain=Rect(lo=(0,), hi=(self.num_gpus,)),
        )
        task.execute()
        self.legate_runtime.issue_execution_fence(block=True)

    def _unload_cudalibs(self):
        task = self.legate_context.create_task(
            CuNumericOpCode.UNLOAD_CUDALIBS,
            manual=True,
            launch_domain=Rect(lo=(0,), hi=(self.num_gpus,)),
        )
        task.execute()

    def get_arg_dtype(self, value_dtype):
        arg_dtype = get_arg_dtype(value_dtype)
        type_system = self.legate_context.type_system
        if arg_dtype not in type_system:
            # We assign T's type code to Argval<T>
            code = type_system[value_dtype].code
            dtype = type_system.add_type(arg_dtype, arg_dtype.itemsize, code)

            for redop in CuNumericRedopCode:
                redop_id = self.legate_context.get_reduction_op_id(
                    redop.value * legion.MAX_TYPE_NUMBER + code
                )
                dtype.register_reduction_op(redop, redop_id)
        return arg_dtype

    def _report_coverage(self):
        total = len(self.api_calls)
        implemented = sum(int(impl) for (_, _, impl) in self.api_calls)

        if total == 0:
            print("cuNumeric API coverage: 0/0")
        else:
            print(
                f"cuNumeric API coverage: {implemented}/{total} "
                f"({implemented / total * 100}%)"
            )
        if self.report_dump_csv is not None:
            with open(self.report_dump_csv, "w") as f:
                print("function_name,location,implemented", file=f)
                for (func_name, loc, impl) in self.api_calls:
                    print(f"{func_name},{loc},{impl}", file=f)

    def destroy(self):
        assert not self.destroyed
        if self.num_gpus > 0:
            self._unload_cudalibs()
        if self.report_coverage:
            self._report_coverage()
        self.destroyed = True

    def create_scalar(self, array: memoryview, dtype, shape=None, wrap=False):
        data = array.tobytes()
        buf = struct.pack(f"{len(data)}s", data)
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
            result = DeferredArray(self, store, dtype=dtype)
        else:
            result = future
        return result

    def set_next_random_epoch(self, epoch):
        self.current_random_epoch = epoch

    def get_next_random_epoch(self):
        result = self.current_random_epoch
        # self.current_random_epoch += 1
        return result

    def is_supported_type(self, dtype):
        return np.dtype(dtype) in self.legate_context.type_system

    def get_numpy_thunk(self, obj, share=False, dtype=None):
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
            if dtype is None:
                dtype = np.dtype(array.type.to_pandas_dtype())
            return DeferredArray(self, store, dtype=dtype)
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
        return self.find_or_create_array_thunk(obj, share=share)

    def has_external_attachment(self, array):
        assert array.base is None or not isinstance(array.base, np.ndarray)
        return self.legate_runtime.has_attachment(array.data)

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
                else:
                    # Removed an added dimension
                    key += (slice(None, None, None),)
                child_idx += 1
                continue
            elif child_idx == array.ndim:
                key += (slice(offsets[idx], offsets[idx] + 1, 1),)
                continue
            elif child_strides[child_idx] == 0:
                # Added dimension in the child not in the parent
                while child_strides[child_idx] == 0:
                    key += (np.newaxis,)
                    child_idx += 1
                # Fall through to the base case
            # Stides in the child should always be greater than or equal
            # to the strides in the parent, if they're not, then that
            # must be an added dimension
            start = offsets[idx]
            if child_strides[child_idx] < parent_strides[idx]:
                key += (slice(start, start + 1, 1),)
                # Doesn't count against the child_idx
            else:
                stride = child_strides[child_idx] // parent_strides[idx]
                stop = start + stride * array.shape[child_idx]
                key += (slice(start, stop, stride),)
                child_idx += 1
        assert child_idx <= array.ndim
        if child_idx < array.ndim:
            return None
        else:
            return key

    def find_or_create_array_thunk(self, array, share=False, defer=False):
        assert isinstance(array, np.ndarray)
        # We have to be really careful here to handle the case of
        # aliased numpy arrays that are passed in from the application
        # In case of aliasing we need to make sure that they are
        # mapped to the same logical region. The way we handle this
        # is to always create the thunk for the root array and
        # then create sub-thunks that mirror the array views
        if array.base is not None and isinstance(array.base, np.ndarray):
            key = self.compute_parent_child_mapping(array)
            if key is None:
                # This base array wasn't made with a view
                if not share:
                    return self.find_or_create_array_thunk(
                        array.copy(),
                        share=False,
                        defer=defer,
                    )
                raise NotImplementedError(
                    "cuNumeric does not currently know "
                    + "how to attach to array views that are not affine "
                    + "transforms of their parent array."
                )
            parent_thunk = self.find_or_create_array_thunk(
                array.base,
                share=share,
                defer=defer,
            )
            # Don't store this one in the ptr_to_thunk as we only want to
            # store the root ones
            return parent_thunk.get_item(key)
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
                store = self.legate_context.create_store(
                    array.dtype,
                    shape=array.shape,
                    optimize_scalar=False,
                )
                store.attach_external_allocation(
                    self.legate_context,
                    array.data,
                    share,
                )
                result = DeferredArray(
                    self,
                    store,
                    dtype=array.dtype,
                    numpy_array=array if share else None,
                )
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
            return DeferredArray(self, store, dtype=dtype)
        else:
            return EagerArray(self, np.empty(shape, dtype=dtype))

    def create_unbound_thunk(self, dtype):
        store = self.legate_context.create_store(dtype)
        return DeferredArray(self, store, dtype=dtype)

    def is_eager_shape(self, shape):
        volume = calculate_volume(shape)
        # Empty arrays are ALWAYS eager
        if volume == 0:
            return True
        # If we're testing then the answer is always no
        if self.test_mode:
            return False
        if len(shape) > LEGATE_MAX_DIM:
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

    def to_eager_array(self, array):
        if self.is_eager_array(array):
            return array
        elif self.is_deferred_array(array):
            return EagerArray(self, array.__numpy_array__())
        else:
            raise RuntimeError("invalid array type")

    def to_deferred_array(self, array):
        if self.is_deferred_array(array):
            return array
        elif self.is_eager_array(array):
            return array.to_deferred_array()
        else:
            raise RuntimeError("invalid array type")

    def warn(self, msg, category=UserWarning):
        if not self.warning:
            return
        stacklevel = find_last_user_stacklevel()
        warnings.warn(msg, stacklevel=stacklevel, category=category)


runtime = Runtime(cunumeric_context)
