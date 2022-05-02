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

from functools import lru_cache
from itertools import permutations, product
from typing import List, Optional, Set, Tuple

import numpy as np
import pytest
from test_tools.generators import mk_0to1_array, permutes_to

import cunumeric as cn

# Limits for exhaustive expression generation routines
MAX_MODES = 3
MAX_OPERANDS = 2
MAX_OPERAND_DIM = 2
MAX_RESULT_DIM = 2
BASE_DIM_LEN = 10


def gen_result(used_modes: int):
    for count in range(min(used_modes, MAX_RESULT_DIM) + 1):
        yield from permutations(range(used_modes), count)


def gen_operand(
    used_modes: int,
    dim_lim: int,
    mode_lim: int,
    op: Optional[List[int]] = None,
):
    if op is None:
        op = []
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


# Exhaustively generate all (normalized) expressions within some limits. These
# limits are set low by default, to keep the unit test running time low.
def gen_expr(
    opers: Optional[List[List[int]]] = None,
    cache: Optional[Set[Tuple[Tuple[int]]]] = None,
):
    if opers is None:
        opers = []
    if cache is None:
        cache = set()
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
    for op in gen_operand(used_modes, dim_lim, mode_lim):
        opers.append(op)
        yield from gen_expr(opers, cache)
        opers.pop()


# Selection of expressions beyond the limits of the exhaustive generation above
LARGE_EXPRS = [
    "ca,da,bc->db",
    "ad,ac,bd->bd",
    "ca,dc,da->ad",
    "ca,dc,ba->bd",
    "ba,dc,ad->ca",
    "ab,da,db->bd",
    "db,dc,ad->ab",
    "bc,ba,db->ba",
    "bc,cd,ab->da",
    "bd,cd,ca->db",
    "cd,cb,ca->ad",
    "adb,bdc,bac->cb",
    "cad,abd,bca->db",
    "cdb,dac,abd->ab",
    "cba,cad,dbc->bc",
    "abc,bda,bcd->bd",
    "dcb,acd,bac->bc",
    "dca,bca,cbd->ad",
    "dba,cbd,cab->cb",
    "cba,adb,dca->cb",
    "cdb,acd,cba->da",
    "bac,cad,cbd->bc",
    "dac,cbd,abc,abd->cb",
    "cab,dcb,cda,bca->ca",
    "dba,bca,cda,adc->dc",
    "acb,bda,dac,acd->db",
    "bcd,cba,adb,bca->cd",
    "cad,dab,cab,acb->ba",
    "abc,abd,acd,cba->ba",
    "cba,cda,bad,acb->db",
    "dca,cba,bdc,bad->cd",
    "cbd,abd,cad,adc->ca",
    "adc,bad,bcd,acb->cb",
]


# Selection of small expressions
SMALL_EXPRS = [
    "->",
    "a->",
    "a->a",
    "a,->",
    "a,->a",
    "a,a->",
    "a,a->a",
    "a,b->ab",
    "ab,ca->a",
    "ab,ca->b",
]


@lru_cache(maxsize=None)
def mk_input_default(lib, shape):
    return [mk_0to1_array(lib, shape)]


@lru_cache(maxsize=None)
def mk_input_that_permutes_to(lib, tgt_shape):
    return [
        mk_0to1_array(lib, src_shape).transpose(axes)
        for (axes, src_shape) in permutes_to(tgt_shape)
    ]


@lru_cache(maxsize=None)
def mk_input_that_broadcasts_to(lib, tgt_shape):
    # If an operand contains the same mode multiple times, then we can't set
    # just one of them to 1. Consider the operation 'aab->ab': (10,10,11),
    # (10,10,1), (1,1,11), (1,1,1) are all acceptable input shapes, but
    # (1,10,11) is not.
    tgt_sizes = list(sorted(set(tgt_shape)))
    res = []
    for mask in product([True, False], repeat=len(tgt_sizes)):
        tgt2src_size = {
            d: (d if keep else 1) for (keep, d) in zip(mask, tgt_sizes)
        }
        src_shape = tuple(tgt2src_size[d] for d in tgt_shape)
        res.append(mk_0to1_array(lib, src_shape))
    return res


@lru_cache(maxsize=None)
def mk_typed_input(lib, shape):
    return [
        mk_0to1_array(lib, shape, np.float16),
        mk_0to1_array(lib, shape, np.float32),
        mk_0to1_array(lib, shape, np.complex64),
    ]


# Can't cache these, because they get overwritten by the operation
def mk_typed_output(lib, shape):
    return [
        lib.zeros(shape, np.float16),
        lib.zeros(shape, np.float32),
        lib.zeros(shape, np.complex64),
    ]


def check_np_vs_cn(expr, mk_input, mk_output=None):
    lhs, rhs = expr.split("->")
    opers = lhs.split(",")
    in_shapes = [
        tuple(BASE_DIM_LEN + ord(m) - ord("a") for m in op) for op in opers
    ]
    out_shape = tuple(BASE_DIM_LEN + ord(m) - ord("a") for m in rhs)
    for (np_inputs, cn_inputs) in zip(
        product(*(mk_input(np, sh) for sh in in_shapes)),
        product(*(mk_input(cn, sh) for sh in in_shapes)),
    ):
        np_res = np.einsum(expr, *np_inputs)
        cn_res = cn.einsum(expr, *cn_inputs)
        rtol = (
            1e-02 if any(x.dtype == np.float16 for x in np_inputs) else 1e-05
        )
        assert np.allclose(np_res, cn_res, rtol=rtol)
        if mk_output is not None:
            for cn_out in mk_output(cn, out_shape):
                cn.einsum(expr, *cn_inputs, out=cn_out)
                rtol = (
                    1e-02
                    if any(x.dtype == np.float16 for x in np_inputs)
                    or cn_out.dtype == np.float16
                    else 1e-05
                )
                assert np.allclose(cn_out, cn_res, rtol=rtol)


@pytest.mark.parametrize("expr", gen_expr())
def test_small(expr):
    print(f"Test small expressions (permutations and broadcasting): {expr}")
    check_np_vs_cn(expr, mk_input_that_permutes_to)
    check_np_vs_cn(expr, mk_input_that_broadcasts_to)


@pytest.mark.parametrize("expr", LARGE_EXPRS)
def test_large(expr):
    print(f"Test large expressions (default execution only): {expr}")
    check_np_vs_cn(expr, mk_input_default)


@pytest.mark.parametrize("expr", SMALL_EXPRS)
def test_cast(expr):
    print(f"Test casting: {expr}")
    check_np_vs_cn(expr, mk_typed_input, mk_typed_output)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
