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


from cunumeric.config import CuNumericOpCode


def get_tile(array, tile_shape, j, i):
    lo = tile_shape * (j, i)
    hi = tile_shape * (j + 1, i + 1)
    slices = tuple(
        slice(lo, min(hi, m)) for lo, hi, m in zip(lo, hi, array.shape)
    )
    return array.get_item(slices)


def potrf(root, tile_shape, i):
    array = get_tile(root, tile_shape, i, i)

    task = root.context.create_task(CuNumericOpCode.POTRF)
    task.add_output(array.base)
    task.add_input(array.base)
    task.add_broadcast(array.base)
    task.execute()


def trsm(root, tile_shape, i, lo, hi):
    rhs = get_tile(root, tile_shape, i, i)

    for j in range(lo, hi):
        lhs = get_tile(root, tile_shape, j, i)

        task = root.context.create_task(CuNumericOpCode.TRSM)
        task.add_output(lhs.base)
        task.add_input(rhs.base)
        task.add_input(lhs.base)
        task.add_broadcast(lhs.base)
        task.add_broadcast(rhs.base)
        task.execute()


def syrk(root, tile_shape, k, i):
    rhs = get_tile(root, tile_shape, k, i)
    lhs = get_tile(root, tile_shape, k, k)

    task = root.context.create_task(CuNumericOpCode.SYRK)
    task.add_output(lhs.base)
    task.add_input(rhs.base)
    task.add_input(lhs.base)
    task.add_broadcast(lhs.base)
    task.add_broadcast(rhs.base)
    task.execute()


def gemm(root, tile_shape, k, i, lo, hi):
    if lo >= hi:
        return
    rhs2 = get_tile(root, tile_shape, k, i)

    for j in range(lo, hi):
        lhs = get_tile(root, tile_shape, j, k)
        rhs1 = get_tile(root, tile_shape, j, i)

        task = root.context.create_task(CuNumericOpCode.GEMM)
        task.add_output(lhs.base)
        task.add_input(rhs1.base)
        task.add_input(rhs2.base)
        task.add_input(lhs.base)
        task.add_broadcast(lhs.base)
        task.add_broadcast(rhs1.base)
        task.add_broadcast(rhs2.base)
        task.execute()


def cholesky(output, input, stacklevel=0, callsite=None):
    output.copy(input, deep=True, stacklevel=stacklevel + 1, callsite=callsite)

    num_procs = output.runtime.num_procs
    shape = output.base.shape
    color_shape = (num_procs, num_procs)
    tile_shape = (shape + color_shape - 1) // color_shape
    if tile_shape * (num_procs - 1) == shape:
        num_procs -= 1

    for i in range(num_procs):
        potrf(output, tile_shape, i)
        trsm(output, tile_shape, i, i + 1, num_procs)
        for k in range(i + 1, num_procs):
            syrk(output, tile_shape, k, i)
            gemm(output, tile_shape, k, i, k + 1, num_procs)

    output.trilu(output, 0, True, stacklevel=stacklevel + 1, callsite=callsite)
