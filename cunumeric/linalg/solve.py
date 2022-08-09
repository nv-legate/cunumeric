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
from __future__ import annotations

from typing import TYPE_CHECKING

from cunumeric.config import CuNumericOpCode

from legate.core import types as ty

from .exception import LinAlgError

if TYPE_CHECKING:
    from legate.core.context import Context
    from legate.core.store import Store

    from ..deferred import DeferredArray


def transpose_copy_single(
    context: Context, input: Store, output: Store
) -> None:
    task = context.create_auto_task(CuNumericOpCode.TRANSPOSE_COPY_2D)
    task.add_output(output)
    task.add_input(input)
    # Output has the same shape as input, but is mapped
    # to a column major instance
    task.add_scalar_arg(False, ty.int32)

    task.add_broadcast(input)
    task.add_broadcast(output)

    task.execute()


def solve_single(context: Context, a : Store, b : Store) -> None:
    task = context.create_auto_task(CuNumericOpCode.SOLVE)
    task.throws_exception(LinAlgError)
    task.add_input(a)
    task.add_input(b)
    task.add_output(a)
    task.add_output(b)

    task.add_broadcast(a)
    task.add_broadcast(b)

    task.execute()


def solve(
    output: DeferredArray, a: DeferredArray, b: DeferredArray
) -> None:

    context = output.context

    a_output = a

    transpose_copy_single(context, a.base, a_output.base)
    transpose_copy_single(context, b.base, output.base)

    solve_single(context, a_output.base, b.base)
