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

import numpy as np
from intra_array_copy import array_gen

import cunumeric as num
from legate.core import LEGATE_MAX_DIM


def test():
    choices1 = [
        [0, 1, 2, 3],
        [10, 11, 12, 13],
        [20, 21, 22, 23],
        [30, 31, 32, 33],
    ]
    a1 = [2, 3, 1, 0]
    num_a1 = num.array(a1)
    num_choices1 = num.array(choices1)

    assert np.array_equal(
        np.choose(a1, choices1), num.choose(num_a1, num_choices1)
    )

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

    for ndim in range(2, LEGATE_MAX_DIM):  # off-by-one is by design
        for np_choices, num_choices in zip(
            array_gen(np, ndim), array_gen(num, ndim)
        ):
            assert np.array_equal(np_choices, num_choices)
            for np_arr, num_arr in zip(
                array_gen(np, ndim - 1), array_gen(num, ndim - 1)
            ):
                n = np_choices.shape[0]
                np_arr_int = (np_arr * 100) % n
                num_arr_int = (num_arr * 100) % n
                if not np.issubdtype(np_arr_int.dtype, np.integer):
                    np_arr_int = np_arr_int.astype(int)
                    num_arr_int = num_arr_int.astype(int)
                assert np.array_equal(np_arr_int, num_arr_int)

                assert np.array_equal(
                    num.choose(num_arr_int, num_choices),
                    np.choose(np_arr_int, num_choices),
                )

    return


if __name__ == "__main__":
    test()
