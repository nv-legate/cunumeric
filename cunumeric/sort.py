# Copyright 2022 NVIDIA Corporation
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


from cunumeric.config import CuNumericOpCode

from legate.core import types as ty


def sort_flattened(output, input, argsort, stable):
    flattened = input.reshape((input.size,), order="C")

    # run sort flattened -- return 1D solution
    sort_result = output.runtime.create_empty_thunk(
        flattened.shape, dtype=output.dtype, inputs=(flattened,)
    )
    sort(sort_result, flattened, argsort, stable=stable)
    output.base = sort_result.base
    output.numpy_array = None


def sort_swapped(output, input, argsort, sort_axis, stable):
    assert sort_axis < input.ndim - 1 and sort_axis >= 0

    # swap axes
    swapped = input.swapaxes(sort_axis, input.ndim - 1)

    swapped_copy = output.runtime.create_empty_thunk(
        swapped.shape, dtype=input.dtype, inputs=(input, swapped)
    )
    swapped_copy.copy(swapped, deep=True)

    # run sort on last axis
    if argsort is True:
        sort_result = output.runtime.create_empty_thunk(
            swapped_copy.shape, dtype=output.dtype, inputs=(swapped_copy,)
        )
        sort(sort_result, swapped_copy, argsort, stable=stable)
        output.base = sort_result.swapaxes(input.ndim - 1, sort_axis).base
        output.numpy_array = None
    else:
        sort(swapped_copy, swapped_copy, argsort, stable=stable)
        output.base = swapped_copy.swapaxes(input.ndim - 1, sort_axis).base
        output.numpy_array = None


def sort_task(output, input, argsort, stable):
    task = output.context.create_task(CuNumericOpCode.SORT)

    needs_unbound_output = output.runtime.num_gpus > 1 and input.ndim == 1

    if needs_unbound_output:
        unbound = output.runtime.create_unbound_thunk(dtype=output.dtype)
        task.add_output(unbound.base)
    else:
        task.add_output(output.base)
        task.add_alignment(output.base, input.base)

    task.add_input(input.base)

    if output.ndim > 1:
        task.add_broadcast(input.base, input.ndim - 1)
    elif output.runtime.num_gpus > 1:
        task.add_nccl_communicator()
    elif output.runtime.num_gpus == 0 and output.runtime.num_procs > 1:
        # Distributed 1D sort on CPU not supported yet
        task.add_broadcast(input.base)

    task.add_scalar_arg(argsort, bool)  # return indices flag
    task.add_scalar_arg(input.base.shape, (ty.int32,))
    task.add_scalar_arg(stable, bool)
    task.execute()

    if needs_unbound_output:
        output.base = unbound.base
        output.numpy_array = None


def sort(output, input, argsort, axis=-1, stable=False):
    if axis is None and input.ndim > 1:
        sort_flattened(output, input, argsort, stable)
    else:
        if axis is None:
            axis = 0
        elif axis < 0:
            axis = input.ndim + axis

        if axis is not input.ndim - 1:
            sort_swapped(output, input, argsort, axis, stable)

        else:
            # run actual sort task
            sort_task(output, input, argsort, stable)
