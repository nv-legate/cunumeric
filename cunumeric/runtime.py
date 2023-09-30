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

import legate.core.types as ty
import numpy as np
from legate.core import LEGATE_MAX_DIM, ProcessorKind, Rect, get_legate_runtime
from legate.core.context import Context as LegateContext
from legate.settings import settings as legate_settings
from typing_extensions import TypeGuard

from .config import (
    BitGeneratorOperation,
    CuNumericOpCode,
    CuNumericTunable,
    cunumeric_context,
    cunumeric_lib,
)
from .deferred import DeferredArray
from .eager import EagerArray
from .settings import settings
from .thunk import NumPyThunk
from .types import NdShape
from .utils import calculate_volume, find_last_user_stacklevel, to_core_dtype

if TYPE_CHECKING:
    import numpy.typing as npt
    from legate.core._legion.future import Future
    from legate.core.operation import AutoTask, ManualTask

    from .array import ndarray


DIMENSION = int


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

        # Make sure that our CuNumericLib object knows about us so it can
        # destroy us
        cunumeric_lib.set_runtime(self)
        assert cunumeric_lib.shared_object is not None
        self.cunumeric_lib = cunumeric_lib.shared_object
        self.has_curand = cunumeric_lib.shared_object.cunumeric_has_curand()

        settings.warn = settings.warn() or legate_settings.test()

        if self.num_gpus > 0 and settings.preload_cudalibs():
            self._load_cudalibs()

        # Maps dimensions to point types
        self._cached_point_types: dict[DIMENSION, ty.Dtype] = dict()
        # Maps value types to struct types used in argmin/argmax
        self._cached_argred_types: dict[ty.Dtype, ty.Dtype] = dict()

    @property
    def num_procs(self) -> int:
        return len(self.legate_runtime.machine)

    @property
    def num_gpus(self) -> int:
        return self.legate_runtime.machine.count(ProcessorKind.GPU)

    def get_point_type(self, dim: DIMENSION) -> ty.Dtype:
        cached = self._cached_point_types.get(dim)
        if cached is not None:
            return cached
        point_dtype = ty.array_type(ty.int64, dim) if dim > 1 else ty.int64
        self._cached_point_types[dim] = point_dtype
        return point_dtype

    def record_api_call(
        self, name: str, location: str, implemented: bool
    ) -> None:
        assert settings.report_coverage()
        self.api_calls.append((name, location, implemented))

    def _load_cudalibs(self) -> None:
        task = self.legate_context.create_manual_task(
            CuNumericOpCode.LOAD_CUDALIBS,
            launch_domain=Rect(lo=(0,), hi=(self.num_gpus,)),
        )
        task.execute()
        self.legate_runtime.issue_execution_fence(block=True)

    def _unload_cudalibs(self) -> None:
        task = self.legate_context.create_manual_task(
            CuNumericOpCode.UNLOAD_CUDALIBS,
            launch_domain=Rect(lo=(0,), hi=(self.num_gpus,)),
        )
        task.execute()

    def get_argred_type(self, value_dtype: ty.Dtype) -> ty.Dtype:
        cached = self._cached_argred_types.get(value_dtype)
        if cached is not None:
            return cached
        argred_dtype = ty.struct_type([ty.int64, value_dtype], True)
        self._cached_argred_types[value_dtype] = argred_dtype
        self.cunumeric_lib.cunumeric_register_reduction_op(
            argred_dtype.uid, value_dtype.code
        )
        return argred_dtype

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

        if (dump_csv := settings.report_dump_csv()) is not None:
            with open(dump_csv, "w") as f:
                print("function_name,location,implemented", file=f)
                for func_name, loc, impl in self.api_calls:
                    print(f"{func_name},{loc},{impl}", file=f)

    def destroy(self) -> None:
        assert not self.destroyed
        if self.num_gpus > 0:
            self._unload_cudalibs()
        if settings.report_coverage():
            self._report_coverage()
        self.destroyed = True

    def create_scalar(
        self,
        array: Union[memoryview, npt.NDArray[Any]],
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
        future = self.create_scalar(array, shape)
        assert all(extent == 1 for extent in shape)
        core_dtype = to_core_dtype(dtype)
        store = self.legate_context.create_store(
            core_dtype,
            shape=shape,
            storage=future,
            optimize_scalar=True,
        )
        return DeferredArray(self, store)

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
            task = self.legate_context.create_manual_task(
                CuNumericOpCode.BITGENERATOR,
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
            task = self.legate_context.create_manual_task(
                CuNumericOpCode.BITGENERATOR,
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
        self.current_random_epoch += 1
        return result

    def get_numpy_thunk(
        self,
        obj: Union[ndarray, npt.NDArray[Any]],
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
            return DeferredArray(self, store)
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
        assert array.base is not None
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
        key: tuple[Union[slice, None], ...] = ()
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
        dtype = to_core_dtype(array.dtype)
        if (
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
                dtype,
                shape=array.shape,
                optimize_scalar=False,
            )
            store.attach_external_allocation(
                array.data,
                share,
            )
            return DeferredArray(
                self,
                store,
                numpy_array=array if share else None,
            )

        # Make this into an eager evaluated thunk
        return EagerArray(self, array)

    def create_empty_thunk(
        self,
        shape: NdShape,
        dtype: ty.Dtype,
        inputs: Optional[Sequence[NumPyThunk]] = None,
    ) -> NumPyThunk:
        if self.is_eager_shape(shape) and self.are_all_eager_inputs(inputs):
            return self.create_eager_thunk(shape, dtype.to_numpy_dtype())

        store = self.legate_context.create_store(
            dtype, shape=shape, optimize_scalar=True
        )
        return DeferredArray(self, store)

    def create_eager_thunk(
        self,
        shape: NdShape,
        dtype: np.dtype[Any],
    ) -> NumPyThunk:
        return EagerArray(self, np.empty(shape, dtype=dtype))

    def create_unbound_thunk(
        self, dtype: ty.Dtype, ndim: int = 1
    ) -> DeferredArray:
        store = self.legate_context.create_store(dtype, ndim=ndim)
        return DeferredArray(self, store)

    def is_eager_shape(self, shape: NdShape) -> bool:
        volume = calculate_volume(shape)

        # Special cases that must always be eager:

        # Newly created empty arrays
        if volume == 0:
            return True

        # Arrays with more dimensions than what Legion was compiled for
        if len(shape) > LEGATE_MAX_DIM:
            return True

        # CUNUMERIC_FORCE_THUNK == "eager"
        if settings.force_thunk() == "eager":
            return True

        if settings.force_thunk() == "deferred":
            return False

        # no forcing; auto mode
        if len(shape) == 0:
            return self.max_eager_volume > 0

        # Otherwise, see if the volume is large enough
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
        if not settings.warn():
            return
        stacklevel = find_last_user_stacklevel()
        warnings.warn(msg, stacklevel=stacklevel, category=category)


runtime = Runtime(cunumeric_context)
