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

from itertools import permutations, product
from typing import List, Set, Tuple

import numpy as np

import cunumeric as num

MAX_MODES = 3
MAX_OPERANDS = 3
MAX_OPERAND_DIM = 2
MAX_RESULT_DIM = 2
BASE_DIM_LEN = 10
TEST_ALL_PERMS = True


def gen_result(used_modes: int):
    for count in range(min(used_modes, MAX_RESULT_DIM) + 1):
        yield from permutations(range(used_modes), count)


def gen_operand(
    used_modes: int,
    dim_lim: int,
    mode_lim: int,
    op: List[int],
):
    # Yield the operand as constructed thus far
    yield op
    # Grow the operand, if we haven't hit the dimension limit yet
    if len(op) >= dim_lim:
        return
    # If we've hit the limit on distinct modes, only use modes
    # appearing on the same operand
    if len(op) == dim_lim - 1 and len(set(op)) >= mode_lim:
        for m in sorted(set(op)):
            op.append(m)
            yield from gen_operand(used_modes, dim_lim, mode_lim, op)
            op.pop()
        return
    # Reuse a mode previously encountered in the overall expression
    for m in range(used_modes):
        op.append(m)
        yield from gen_operand(used_modes, dim_lim, mode_lim, op)
        op.pop()
    # Add a new mode (number sequentially)
    if used_modes >= MAX_MODES:
        return
    op.append(used_modes)
    yield from gen_operand(used_modes + 1, dim_lim, mode_lim, op)
    op.pop()


def gen_expr(
    opers: List[List[int]],
    cache: Set[Tuple[Tuple[int]]],
):
    # The goal here is to avoid producing duplicate expressions, up to
    # reordering of operands and alpha-renaming, e.g. the following
    # are considered equivalent (for the purposes of testing):
    #   a,b and a,c
    #   a,b and b,a
    #   ab,bb and aa,ab
    # Don't consider reorderings of arrays
    key = tuple(sorted(tuple(op) for op in opers))
    if key in cache:
        return
    cache.add(key)
    used_modes = max((m + 1 for op in opers for m in op), default=0)
    # Build an expression using the current list of operands
    if len(opers) > 0:
        lhs = ",".join("".join(chr(ord("a") + m) for m in op) for op in opers)
        for result in gen_result(used_modes):
            rhs = "".join(chr(ord("a") + m) for m in result)
            yield lhs + "->" + rhs
    # Add a new operand and recurse
    if len(opers) >= MAX_OPERANDS:
        return
    # Always put the longest operands first
    dim_lim = len(opers[-1]) if len(opers) > 0 else MAX_OPERAND_DIM
    # Between operands of the same length, put those with the most distinct
    # modes first.
    mode_lim = len(set(opers[-1])) if len(opers) > 0 else MAX_OPERAND_DIM
    for op in gen_operand(used_modes, dim_lim, mode_lim, []):
        opers.append(op)
        yield from gen_expr(opers, cache)
        opers.pop()


def gen_array(lib, operand: str):
    # Test
    out_shape = tuple(BASE_DIM_LEN + ord(m) - ord("a") for m in operand)
    size = np.prod(out_shape)
    for axes in permutations(range(len(operand))):
        in_shape = [-1] * len(out_shape)
        for (i, j) in enumerate(axes):
            in_shape[j] = out_shape[i]
        in_shape = tuple(in_shape)
        in_arr = lib.full(in_shape, 2.0)
        if len(in_shape) > 0:
            in_arr[:] = lib.arange(size).reshape(in_shape) / size
        yield in_arr.transpose(axes)
        if not TEST_ALL_PERMS:
            return


def test():
    total_exprs = 0
    total_perms = 0
    for expr in gen_expr([], set()):
        lhs, rhs = expr.split("->")
        opers = lhs.split(",")
        # TODO: Remove these restrictions as we fix the issues
        if (
            len(opers) != 2
            or len(opers[0]) < 1
            or len(opers[1]) < 1
            or len(rhs) < 1
            or len(set(opers[0]) - set(opers[1]) - set(rhs)) > 0
            or len(set(opers[1]) - set(opers[0]) - set(rhs)) > 0
            or len(set(opers[0])) != len(opers[0])
            or len(set(opers[1])) != len(opers[1])
        ):
            continue
        perms = 0
        for (np_arrays, num_arrays) in zip(
            product(*(gen_array(np, op) for op in opers)),
            product(*(gen_array(num, op) for op in opers)),
        ):
            assert np.allclose(
                np.einsum(expr, *np_arrays),
                num.contract(expr, *num_arrays),  # TODO: switch to num.einsum
            )
            perms += 1
        total_exprs += 1
        total_perms += perms
        print(f"tested {expr} - {perms} permutations")
    print(f"tested {total_exprs} expressions, {total_perms} permutations")


if __name__ == "__main__":
    test()
