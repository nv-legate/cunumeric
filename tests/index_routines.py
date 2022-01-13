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

import random

import numpy as np
from test_tools.generators import mk_seq_array

import cunumeric as num
from legate.core import LEGATE_MAX_DIM


def test():
    # --------------------------------------------------------------
    # choose operator
    # --------------------------------------------------------------
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

    for ndim in range(1, LEGATE_MAX_DIM + 1):
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

    # --------------------------------------------------------------
    # diagonal operator
    # --------------------------------------------------------------
    ad = np.arange(24).reshape(4, 3, 2)
    num_ad = num.array(ad)
    assert np.array_equal(ad.diagonal(), num_ad.diagonal())
    assert np.array_equal(ad.diagonal(0, 1, 2), num_ad.diagonal(0, 1, 2))
    assert np.array_equal(ad.diagonal(1, 0, 2), num_ad.diagonal(1, 0, 2))

    for ndim in range(2, LEGATE_MAX_DIM + 1):
        a_shape = tuple(random.randint(1, 9) for i in range(ndim))
        np_array = mk_seq_array(np, a_shape)
        num_array = mk_seq_array(num, a_shape)
        if ndim == 2:
            axis1 = 0
            axis2 = 1
        else:
            axis1 = random.randint(0, ndim - 2)
            axis2 = random.randint(axis1 + 1, ndim - 1)
        diag_size = min(a_shape[axis1], a_shape[axis2]) - 1
        offset = random.randint(-diag_size, diag_size)
        assert np.array_equal(
            np_array.diagonal(offset, axis1, axis2),
            num_array.diagonal(offset, axis1, axis2),
        )
    return


if __name__ == "__main__":
    test()
