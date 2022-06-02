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

from typing import TYPE_CHECKING, cast

from cunumeric.config import CuNumericOpCode

from legate.core import Rect, types as ty
from legate.core.operation import ManualTask

if TYPE_CHECKING:
    from legate.core.context import Context
    from legate.core.shape import Shape
    from legate.core.store import StorePartition

    from ..deferred import DeferredArray
    from ..runtime import Runtime


def transpose_copy(
    context: Context,
    launch_domain: Rect,
    p_input: StorePartition,
    p_output: StorePartition,
) -> None:
    task = cast(
        ManualTask,
        context.create_task(
            CuNumericOpCode.TRANSPOSE_COPY_2D,
            manual=True,
            launch_domain=launch_domain,
        ),
    )
    task.add_output(p_output)
    task.add_input(p_input)
    # Output has the same shape as input, but is mapped
    # to a column major instance
    task.add_scalar_arg(False, ty.int32)

    task.execute()


def potrf(context: Context, p_output: StorePartition, i: int) -> None:
    launch_domain = Rect(lo=(i, i), hi=(i + 1, i + 1))
    task = cast(
        ManualTask,
        context.create_task(
            CuNumericOpCode.POTRF, manual=True, launch_domain=launch_domain
        ),
    )
    task.add_output(p_output)
    task.add_input(p_output)
    task.execute()


def trsm(
    context: Context, p_output: StorePartition, i: int, lo: int, hi: int
) -> None:
    if lo >= hi:
        return

    rhs = p_output.get_child_store(i, i)
    lhs = p_output

    launch_domain = Rect(lo=(lo, i), hi=(hi, i + 1))
    task = cast(
        ManualTask,
        context.create_task(
            CuNumericOpCode.TRSM, manual=True, launch_domain=launch_domain
        ),
    )
    task.add_output(lhs)
    task.add_input(rhs)
    task.add_input(lhs)
    task.execute()


def syrk(context: Context, p_output: StorePartition, k: int, i: int) -> None:
    rhs = p_output.get_child_store(k, i)
    lhs = p_output

    launch_domain = Rect(lo=(k, k), hi=(k + 1, k + 1))
    task = cast(
        ManualTask,
        context.create_task(
            CuNumericOpCode.SYRK, manual=True, launch_domain=launch_domain
        ),
    )
    task.add_output(lhs)
    task.add_input(rhs)
    task.add_input(lhs)
    task.execute()


def gemm(
    context: Context,
    p_output: StorePartition,
    k: int,
    i: int,
    lo: int,
    hi: int,
) -> None:
    if lo >= hi:
        return

    rhs2 = p_output.get_child_store(k, i)
    lhs = p_output
    rhs1 = p_output

    launch_domain = Rect(lo=(lo, k), hi=(hi, k + 1))
    task = cast(
        ManualTask,
        context.create_task(
            CuNumericOpCode.GEMM, manual=True, launch_domain=launch_domain
        ),
    )
    task.add_output(lhs)
    task.add_input(rhs1, proj=lambda p: (p[0], i))
    task.add_input(rhs2)
    task.add_input(lhs)
    task.execute()


MIN_CHOLESKY_TILE_SIZE = 2048
MIN_CHOLESKY_MATRIX_SIZE = 8192


# TODO: We need a better cost model
def choose_color_shape(runtime: Runtime, shape: Shape) -> tuple[int, int]:
    if runtime.args.test_mode:
        num_tiles = runtime.num_procs * 2
        return (num_tiles, num_tiles)
    else:
        extent = shape[0]
        # If there's only one processor or the matrix is too small,
        # don't even bother to partition it at all
        if runtime.num_procs == 1 or extent <= MIN_CHOLESKY_MATRIX_SIZE:
            return (1, 1)

        # If the matrix is big enough to warrant partitioning,
        # pick the granularity that the tile size is greater than a threshold
        num_tiles = runtime.num_procs
        max_num_tiles = runtime.num_procs * 4
        while (
            (extent + num_tiles - 1) // num_tiles > MIN_CHOLESKY_TILE_SIZE
            and num_tiles * 2 <= max_num_tiles
        ):
            num_tiles *= 2

        return (num_tiles, num_tiles)


def tril(context: Context, p_output: StorePartition, n: int) -> None:
    launch_domain = Rect((n, n))
    task = cast(
        ManualTask,
        context.create_task(
            CuNumericOpCode.TRILU, manual=True, launch_domain=launch_domain
        ),
    )

    task.add_output(p_output)
    task.add_input(p_output)
    task.add_scalar_arg(True, bool)
    task.add_scalar_arg(0, ty.int32)
    # Add a fake task argument to indicate that this is for Cholesky
    task.add_scalar_arg(True, bool)

    task.execute()


def cholesky(
    output: DeferredArray, input: DeferredArray, no_tril: bool
) -> None:
    shape = output.base.shape
    initial_color_shape = choose_color_shape(output.runtime, shape)
    tile_shape = (shape + initial_color_shape - 1) // initial_color_shape
    color_shape = (shape + tile_shape - 1) // tile_shape
    n = color_shape[0]

    context = output.context
    p_input = input.base.partition_by_tiling(tile_shape)
    p_output = output.base.partition_by_tiling(tile_shape)
    transpose_copy(context, Rect(hi=color_shape), p_input, p_output)

    for i in range(n):
        potrf(context, p_output, i)
        trsm(context, p_output, i, i + 1, n)
        for k in range(i + 1, n):
            syrk(context, p_output, k, i)
            gemm(context, p_output, k, i, k + 1, n)

    if no_tril:
        return

    tril(context, p_output, n)
