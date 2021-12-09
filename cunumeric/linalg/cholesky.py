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

from legate.core import Rect


def potrf(context, part, i):
    launch_domain = Rect(lo=(i, i), hi=(i + 1, i + 1))
    task = context.create_task(
        CuNumericOpCode.POTRF, manual=True, launch_domain=launch_domain
    )
    task.add_output(part)
    task.add_input(part)
    task.execute()


def trsm(context, part, i, lo, hi):
    if lo >= hi:
        return

    rhs = part.get_child_store(i, i)
    lhs = part

    launch_domain = Rect(lo=(lo, i), hi=(hi, i + 1))
    task = context.create_task(
        CuNumericOpCode.TRSM, manual=True, launch_domain=launch_domain
    )
    task.add_output(lhs)
    task.add_input(rhs)
    task.add_input(lhs)
    task.execute()


def syrk(context, part, k, i):
    rhs = part.get_child_store(k, i)
    lhs = part

    launch_domain = Rect(lo=(k, k), hi=(k + 1, k + 1))
    task = context.create_task(
        CuNumericOpCode.SYRK, manual=True, launch_domain=launch_domain
    )
    task.add_output(lhs)
    task.add_input(rhs)
    task.add_input(lhs)
    task.execute()


def gemm(context, part, k, i, lo, hi):
    if lo >= hi:
        return

    rhs2 = part.get_child_store(k, i)
    lhs = part
    rhs1 = part

    launch_domain = Rect(lo=(lo, k), hi=(hi, k + 1))
    task = context.create_task(
        CuNumericOpCode.GEMM, manual=True, launch_domain=launch_domain
    )
    task.add_output(lhs)
    task.add_input(rhs1, proj=lambda p: (p[0], i))
    task.add_input(rhs2)
    task.add_input(lhs)
    task.execute()


def cholesky(output, input, stacklevel=0, callsite=None):
    output.copy(input, deep=True, stacklevel=stacklevel + 1, callsite=callsite)

    num_procs = output.runtime.num_procs
    shape = output.base.shape
    color_shape = (num_procs, num_procs)
    tile_shape = (shape + color_shape - 1) // color_shape
    if tile_shape * (num_procs - 1) == shape:
        num_procs -= 1

    p_output = output.base.partition_by_tiling(tile_shape)
    context = output.context

    for i in range(num_procs):
        potrf(context, p_output, i)
        trsm(context, p_output, i, i + 1, num_procs)
        for k in range(i + 1, num_procs):
            syrk(context, p_output, k, i)
            gemm(context, p_output, k, i, k + 1, num_procs)

    output.trilu(output, 0, True, stacklevel=stacklevel + 1, callsite=callsite)
