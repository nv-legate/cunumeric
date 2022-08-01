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
from __future__ import annotations

import struct
import warnings
from functools import reduce
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import numpy as np
from legate.rc import ArgSpec, Argument, parse_command_args
from typing_extensions import TypeGuard

import legate.core.types as ty
from legate.core import LEGATE_MAX_DIM, Rect, get_legate_runtime, legion
from legate.core.context import Context as LegateContext

from .config import (
    BitGeneratorOperation,
    CuNumericOpCode,
    CuNumericRedopCode,
    CuNumericTunable,
    CuNumericTypeCodes,
    cunumeric_context,
    cunumeric_lib,
)
from .deferred import DeferredArray
from .eager import EagerArray
from .thunk import NumPyThunk
from .types import NdShape
from .utils import calculate_volume, find_last_user_stacklevel, get_arg_dtype

if TYPE_CHECKING:
    import numpy.typing as npt

    from legate.core._legion.future import Future
    from legate.core.operation import AutoTask, ManualTask

    from .types import NdShapeLike

_supported_dtypes = {
    np.bool_: ty.bool_,
    np.int8: ty.int8,
    np.int16: ty.int16,
    np.int32: ty.int32,
    int: ty.int64,
    np.int64: ty.int64,
    np.uint8: ty.uint8,
    np.uint16: ty.uint16,
    np.uint32: ty.uint32,
    np.uint: ty.uint64,
    np.uint64: ty.uint64,
    np.float16: ty.float16,
    np.float32: ty.float32,
    float: ty.float64,
    np.float64: ty.float64,
    np.complex64: ty.complex64,
    np.complex128: ty.complex128,
}

ARGS = [
    Argument(
        "test",
        ArgSpec(
            action="store_true",
            default=False,
            dest="test_mode",
            help="Enable test mode. In test mode, all cuNumeric ndarrays are managed by the distributed runtime and the NumPy fallback for small arrays is turned off.",  # noqa E501
        ),
    ),
    Argument(
        "preload-cudalibs",
        ArgSpec(
            action="store_true",
            default=False,
            dest="preload_cudalibs",
            help="Preload and initialize handles of all CUDA libraries (cuBLAS, cuSOLVER, etc.) used in cuNumericLoad CUDA libs early",  # noqa E501
        ),
    ),
    Argument(
        "warn",
        ArgSpec(
            action="store_true",
            default=False,
            dest="warning",
            help="Turn on warnings",
        ),
    ),
    Argument(
        "report:coverage",
        ArgSpec(
            action="store_true",
            default=False,
            dest="report_coverage",
            help="Print an overall percentage of cunumeric coverage",
        ),
    ),
    Argument(
        "report:dump-callstack",
        ArgSpec(
            action="store_true",
            default=False,
            dest="report_dump_callstack",
            help="Print an overall percentage of cunumeric coverage with call stack details",  # noqa E501
        ),
    ),
    Argument(
        "report:dump-csv",
        ArgSpec(
            action="store",
            type=str,
            nargs="?",
            default=None,
            dest="report_dump_csv",
            help="Save a coverage report to a specified CSV file",
        ),
    ),
]


class Runtime(object):
    def __init__(self, legate_context: LegateContext) -> None:
        self.legate_context = legate_context
        self.legate_runtime = get_legate_runtime()
        self.current_random_epoch = 0
        self.current_random_bitgenid = 0
        self.current_random_bitgen_zombies: tuple[Any, ...] = ()
        self.destroyed = False
        self.api_calls: list[tuple[str, str, bool]] = []

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
        self.has_curand = cunumeric_lib.shared_object.cunumeric_has_curand()
        self._register_dtypes()

        self.args = parse_command_args("cunumeric", ARGS)
        self.args.warning = self.args.warning or self.args.test_mode

        if self.num_gpus > 0 and self.args.preload_cudalibs:
            self._load_cudalibs()

    def _register_dtypes(self) -> None:
        type_system = self.legate_context.type_system
        for numpy_type, core_type in _supported_dtypes.items():
            type_system.make_alias(
                np.dtype(numpy_type), core_type  # type: ignore
            )

        for n in range(1, LEGATE_MAX_DIM + 1):
            self._register_point_type(n)

    def _register_point_type(self, n: int) -> None:
        type_system = self.legate_context.type_system
        point_type = "Point" + str(n)
        if point_type not in type_system:
            code = CuNumericTypeCodes.CUNUMERIC_TYPE_POINT1 + n - 1
            size_in_bytes = 8 * n
            type_system.add_type(point_type, size_in_bytes, code)

    def get_point_type(self, n: int) -> str:
        type_system = self.legate_context.type_system
        point_type = "Point" + str(n)
        if point_type not in type_system:
            raise ValueError(f"there is no point type registered for {n}")
        return point_type

    def record_api_call(
        self, name: str, location: str, implemented: bool
    ) -> None:
        assert self.args.report_coverage
        self.api_calls.append((name, location, implemented))

    def _load_cudalibs(self) -> None:
        task = self.legate_context.create_task(
            CuNumericOpCode.LOAD_CUDALIBS,
            manual=True,
            launch_domain=Rect(lo=(0,), hi=(self.num_gpus,)),
        )
        task.execute()
        self.legate_runtime.issue_execution_fence(block=True)

    def _unload_cudalibs(self) -> None:
        task = self.legate_context.create_task(
            CuNumericOpCode.UNLOAD_CUDALIBS,
            manual=True,
            launch_domain=Rect(lo=(0,), hi=(self.num_gpus,)),
        )
        task.execute()

    def get_arg_dtype(self, value_dtype: np.dtype[Any]) -> np.dtype[Any]:
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

    def _report_coverage(self) -> None:
        total = len(self.api_calls)
        implemented = sum(int(impl) for (_, _, impl) in self.api_calls)

        if total == 0:
            print("cuNumeric API coverage: 0/0")
        else:
            print(
                f"cuNumeric API coverage: {implemented}/{total} "
                f"({implemented / total * 100}%)"
            )
        if self.args.report_dump_csv is not None:
            with open(self.args.report_dump_csv, "w") as f:
                print("function_name,location,implemented", file=f)
                for (func_name, loc, impl) in self.api_calls:
                    print(f"{func_name},{loc},{impl}", file=f)

    def destroy(self) -> None:
        assert not self.destroyed
        if self.num_gpus > 0:
            self._unload_cudalibs()
        if hasattr(self, "args") and self.args.report_coverage:
            self._report_coverage()
        self.destroyed = True

    def create_scalar(
        self,
        array: Union[memoryview, npt.NDArray[Any]],
        dtype: np.dtype[Any],
        shape: Optional[NdShape] = None,
    ) -> Future:
        data = array.tobytes()
        buf = struct.pack(f"{len(data)}s", data)
        return self.legate_runtime.create_future(buf, len(buf))

    def create_wrapped_scalar(
        self,
        array: Union[memoryview, npt.NDArray[Any]],
        dtype: np.dtype[Any],
        shape: NdShape,
    ) -> DeferredArray:
        future = self.create_scalar(array, dtype, shape)
        assert all(extent == 1 for extent in shape)
        store = self.legate_context.create_store(
            dtype,
            shape=shape,
            storage=future,
            optimize_scalar=True,
        )
        return DeferredArray(self, store, dtype=dtype)

    def bitgenerator_populate_task(
        self,
        task: Union[AutoTask, ManualTask],
        taskop: int,
        generatorID: int,
        generatorType: int = 0,
        seed: Union[int, None] = 0,
        flags: int = 0,
    ) -> None:
        task.add_scalar_arg(taskop, ty.int32)
        task.add_scalar_arg(generatorID, ty.int32)
        task.add_scalar_arg(generatorType, ty.uint32)
        task.add_scalar_arg(seed, ty.uint64)
        task.add_scalar_arg(flags, ty.uint32)

    def bitgenerator_create(
        self,
        generatorType: int,
        seed: Union[int, None],
        flags: int,
        forceCreate: bool = False,
    ) -> int:
        self.current_random_bitgenid = self.current_random_bitgenid + 1
        if forceCreate:
            task = self.legate_context.create_task(
                CuNumericOpCode.BITGENERATOR,
                manual=True,
                launch_domain=Rect(lo=(0,), hi=(self.num_procs,)),
            )
            self.bitgenerator_populate_task(
                task,
                BitGeneratorOperation.CREATE,
                self.current_random_bitgenid,
                generatorType,
                seed,
                flags,
            )
            task.add_scalar_arg(
                self.current_random_bitgen_zombies, (ty.uint32,)
            )
            self.current_random_bitgen_zombies = ()
            task.execute()
            self.legate_runtime.issue_execution_fence()
        return self.current_random_bitgenid

    def bitgenerator_destroy(
        self, handle: Any, disposing: bool = True
    ) -> None:
        if disposing:
            # when called from within destructor, do not schedule a task
            self.current_random_bitgen_zombies += (handle,)
        else:
            # with explicit destruction, do schedule a task
            self.legate_runtime.issue_execution_fence()
            task = self.legate_context.create_task(
                CuNumericOpCode.BITGENERATOR,
                manual=True,
                launch_domain=Rect(lo=(0,), hi=(self.num_procs,)),
            )
            self.bitgenerator_populate_task(
                task, BitGeneratorOperation.DESTROY, handle
            )
            task.add_scalar_arg(
                self.current_random_bitgen_zombies, (ty.uint32,)
            )
            self.current_random_bitgen_zombies = ()
            task.execute()

    def set_next_random_epoch(self, epoch: int) -> None:
        self.current_random_epoch = epoch

    def get_next_random_epoch(self) -> int:
        result = self.current_random_epoch
        # self.current_random_epoch += 1
        return result

    def is_point_type(self, dtype: Union[str, np.dtype[Any]]) -> bool:
        if (
            isinstance(dtype, str)
            and len(dtype) == 6
            and dtype[0:5] == "Point"
        ):
            return True
        else:
            return False

    def is_supported_type(self, dtype: Union[str, np.dtype[Any]]) -> bool:
        if self.is_point_type(dtype):
            return dtype in self.legate_context.type_system
        else:
            return np.dtype(dtype) in self.legate_context.type_system

    def get_numpy_thunk(
        self,
        obj: Any,
        share: bool = False,
        dtype: Optional[np.dtype[Any]] = None,
    ) -> NumPyThunk:
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
        # Make sure to convert numpy matrices to numpy arrays here
        # as the former doesn't behave quite like the latter
        if type(obj) is not np.ndarray:
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

    def has_external_attachment(self, array: Any) -> bool:
        assert array.base is None or not isinstance(array.base, np.ndarray)
        return self.legate_runtime.has_attachment(array.data)

    @staticmethod
    def compute_parent_child_mapping(
        array: npt.NDArray[Any],
    ) -> Union[tuple[Union[slice, None], ...], None]:
        # We need an algorithm for figuring out how to compute the
        # slice object that was used to generate a child array from
        # a parent array so we can build the same mapping from a
        # logical region to a subregion
        parent_ptr = int(array.base.ctypes.data)  # type: ignore
        child_ptr = int(array.ctypes.data)
        assert child_ptr >= parent_ptr
        ptr_diff = child_ptr - parent_ptr
        parent_shape = array.base.shape  # type: ignore
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
        key: tuple[Union[slice, None], ...] = ()
        child_idx = 0
        child_strides = tuple(array.strides)
        parent_strides = tuple(array.base.strides)  # type: ignore
        for idx in range(array.base.ndim):  # type: ignore
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

    def find_or_create_array_thunk(
        self, array: npt.NDArray[Any], share: bool = False, defer: bool = False
    ) -> NumPyThunk:
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
                # We didn't attach to this so we don't need to save it
                return self.create_wrapped_scalar(
                    array.data,
                    array.dtype,
                    array.shape,
                )

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
            return DeferredArray(
                self,
                store,
                dtype=array.dtype,
                numpy_array=array if share else None,
            )

        assert not defer
        # Make this into an eager evaluated thunk
        return EagerArray(self, array)

    def create_empty_thunk(
        self,
        shape: NdShapeLike,
        dtype: np.dtype[Any],
        inputs: Optional[Sequence[NumPyThunk]] = None,
    ) -> NumPyThunk:
        computed_shape = (shape,) if isinstance(shape, int) else shape
        if self.is_supported_type(dtype) and not (
            self.is_eager_shape(computed_shape)
            and self.are_all_eager_inputs(inputs)
        ):
            store = self.legate_context.create_store(
                dtype, shape=computed_shape, optimize_scalar=True
            )
            return DeferredArray(self, store, dtype=dtype)
        else:
            return EagerArray(self, np.empty(shape, dtype=dtype))

    def create_unbound_thunk(
        self, dtype: np.dtype[Any], ndim: int = 1
    ) -> DeferredArray:
        store = self.legate_context.create_store(dtype, ndim=ndim)
        return DeferredArray(self, store, dtype=dtype)

    def is_eager_shape(self, shape: NdShape) -> bool:
        volume = calculate_volume(shape)
        # Newly created empty arrays are ALWAYS eager
        if volume == 0:
            return True
        # If we're testing then the answer is always no
        if self.args.test_mode:
            return False
        if len(shape) > LEGATE_MAX_DIM:
            return True
        if len(shape) == 0:
            return self.max_eager_volume > 0
        # See if the volume is large enough
        return volume <= self.max_eager_volume

    @staticmethod
    def are_all_eager_inputs(inputs: Optional[Sequence[NumPyThunk]]) -> bool:
        if inputs is None:
            return True
        for inp in inputs:
            assert isinstance(inp, NumPyThunk)
            if not isinstance(inp, EagerArray):
                return False
        return True

    @staticmethod
    def is_eager_array(array: NumPyThunk) -> TypeGuard[EagerArray]:
        return isinstance(array, EagerArray)

    @staticmethod
    def is_deferred_array(
        array: Optional[NumPyThunk],
    ) -> TypeGuard[DeferredArray]:
        return isinstance(array, DeferredArray)

    def to_eager_array(self, array: NumPyThunk) -> EagerArray:
        if self.is_eager_array(array):
            return array
        elif self.is_deferred_array(array):
            return EagerArray(self, array.__numpy_array__())
        else:
            raise RuntimeError("invalid array type")

    def to_deferred_array(self, array: NumPyThunk) -> DeferredArray:
        if self.is_deferred_array(array):
            return array
        elif self.is_eager_array(array):
            return array.to_deferred_array()
        else:
            raise RuntimeError("invalid array type")

    def warn(self, msg: str, category: type = UserWarning) -> None:
        if not self.args.warning:
            return
        stacklevel = find_last_user_stacklevel()
        warnings.warn(msg, stacklevel=stacklevel, category=category)


runtime = Runtime(cunumeric_context)
