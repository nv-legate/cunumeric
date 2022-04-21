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

import random
from itertools import permutations

import numpy as np
import pytest
from cunumeric.eager import diagonal_reference
from test_tools.generators import mk_seq_array

import cunumeric as num
from legate.core import LEGATE_MAX_DIM


def test_choose_1d():
    choices1 = [
        [0, 1, 2, 3],
        [10, 11, 12, 13],
        [20, 21, 22, 23],
        [30, 31, 32, 33],
    ]
    a1 = [2, 3, 1, 0]
    num_a1 = num.array(a1)
    num_choices1 = num.array(choices1)

    aout = np.array([2.3, 3.0, 1.2, 0.3])
    num_aout = num.array(aout)

    assert np.array_equal(
        np.choose(a1, choices1, out=aout),
        num.choose(num_a1, num_choices1, out=num_aout),
    )
    assert np.array_equal(aout, num_aout)

    b = [2, 4, 1, 0]
    num_b = num.array(b)
    assert np.array_equal(
        np.choose(b, choices1, mode="clip"),
        num.choose(num_b, num_choices1, mode="clip"),
    )
    assert np.array_equal(
        np.choose(b, choices1, mode="wrap"),
        num.choose(num_b, num_choices1, mode="wrap"),
    )


def test_choose_2d():
    a2 = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    choices2 = [-10, 10]
    num_a2 = num.array(a2)
    num_choices2 = num.array(choices2)
    assert np.array_equal(
        num.choose(num_a2, num_choices2), np.choose(a2, choices2)
    )

    a3 = np.array([0, 1]).reshape((2, 1, 1))
    c1 = np.array([1, 2, 3]).reshape((1, 3, 1))
    c2 = np.array([-1, -2, -3, -4, -5]).reshape((1, 1, 5))
    num_a3 = num.array(a3)
    num_c1 = num.array(c1)
    num_c2 = num.array(c2)
    assert np.array_equal(
        np.choose(a3, (c1, c2)), num.choose(num_a3, (num_c1, num_c2))
    )


@pytest.mark.parametrize("ndim", range(1, LEGATE_MAX_DIM + 1))
def test_choose_target_ndim(ndim):
    tgt_shape = (5,) * ndim
    # try various shapes that broadcast to the target shape
    shapes = [tgt_shape]
    for d in range(len(tgt_shape)):
        sh = list(tgt_shape)
        sh[d] = 1
        shapes.append(tuple(sh))
    for choices_shape in shapes:
        # make sure the choices are between 0 and 1
        np_choices = mk_seq_array(np, choices_shape) % 2
        num_choices = mk_seq_array(num, choices_shape) % 2
        for rhs1_shape in shapes:
            np_rhs1 = np.full(rhs1_shape, 42)
            num_rhs1 = num.full(rhs1_shape, 42)
            for rhs2_shape in shapes:
                # make sure rhs1 and rhs2 have different values
                np_rhs2 = np.full(rhs2_shape, 17)
                num_rhs2 = num.full(rhs2_shape, 17)
                np_res = np.choose(np_choices, (np_rhs1, np_rhs2))
                num_res = num.choose(num_choices, (num_rhs1, num_rhs2))
                assert np.array_equal(np_res, num_res)


def test_diagonal():
    ad = np.arange(24).reshape(4, 3, 2)
    num_ad = num.array(ad)
    assert np.array_equal(ad.diagonal(), num_ad.diagonal())
    assert np.array_equal(ad.diagonal(0, 1, 2), num_ad.diagonal(0, 1, 2))
    assert np.array_equal(ad.diagonal(1, 0, 2), num_ad.diagonal(1, 0, 2))
    assert np.array_equal(ad.diagonal(-1, 0, 2), num_ad.diagonal(-1, 0, 2))

    # test diagonal
    for ndim in range(2, LEGATE_MAX_DIM + 1):
        a_shape = tuple(random.randint(1, 9) for i in range(ndim))
        np_array = mk_seq_array(np, a_shape)
        num_array = mk_seq_array(num, a_shape)

        # test diagonal
        for axes in permutations(range(ndim), 2):
            diag_size = min(a_shape[axes[0]], a_shape[axes[1]]) - 1
            for offset in range(-diag_size + 1, diag_size):
                assert np.array_equal(
                    np_array.diagonal(offset, axes[0], axes[1]),
                    num_array.diagonal(offset, axes[0], axes[1]),
                )

    # test for diagonal_helper
    for ndim in range(3, LEGATE_MAX_DIM + 1):
        a_shape = tuple(random.randint(1, 9) for i in range(ndim))
        np_array = mk_seq_array(np, a_shape)
        np_array = mk_seq_array(np, a_shape)
        num_array = mk_seq_array(num, a_shape)
        for num_axes in range(3, ndim + 1):
            for axes in permutations(range(ndim), num_axes):
                res_num = num.diagonal(
                    num_array, offset=0, extract=True, axes=axes
                )
                res_ref = diagonal_reference(np_array, axes)
                assert np.array_equal(res_num, res_ref)


KS = [0, -1, 1, -2, 2]


@pytest.mark.parametrize("k", KS, ids=lambda k: f"(k={k})")
def test_diag(k):
    print(f"diag(k={k})")
    a = num.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
        ]
    )
    an = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
        ]
    )

    b = num.diag(a, k=k)
    bn = np.diag(an, k=k)
    assert np.array_equal(b, bn)

    c = num.diag(b, k=k)
    cn = np.diag(bn, k=k)
    assert np.array_equal(c, cn)

    d = num.array(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
        ]
    )
    dn = np.array(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
        ]
    )

    e = num.diag(d, k=k)
    en = np.diag(dn, k=k)
    assert np.array_equal(e, en)

    f = num.diag(e, k=k)
    fn = np.diag(en, k=k)
    assert np.array_equal(f, fn)

    return


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
