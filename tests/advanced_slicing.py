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

import legate.numpy as lg


def sequence_2d():
    return lg.array(
        [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24],
        ]
    )


def test():

    # 1d __getitem__

    # index arrays can be boolean, of the same shape as the base array
    a = lg.arange(10)
    assert np.array_equal(a[lg.array([False, True] * 5)], [1, 3, 5, 7, 9])
    assert np.array_equal(a[a > 5], [6, 7, 8, 9])

    # index arrays can be integer, one per base array dimension
    a = lg.arange(10) * 2
    assert np.array_equal(a[lg.array([1, 3, 5, 7, 9])], [2, 6, 10, 14, 18])

    # output shape follows index array shape
    a = lg.arange(10) * 2
    assert np.array_equal(
        a[lg.array([[1, 2, 3], [4, 5, 6]])], [[2, 4, 6], [8, 10, 12]]
    )

    # index arrays can be any sequence object
    a = lg.arange(10) * 2
    assert np.array_equal(a[[4, 3, 2, 1]], [8, 6, 4, 2])

    # index arrays are automatically cast to int64
    a = lg.arange(10)
    assert np.array_equal(a[lg.arange(5, 10, dtype=np.int16)], [5, 6, 7, 8, 9])

    # advanced slicing creates copies
    a = lg.arange(10)
    b = a[range(5)]
    b[:] = 0
    assert np.array_equal(a, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # can read through views
    a = np.arange(10)
    c = np.arange(9, -1, -1)
    assert np.array_equal(a[2:7][c[5:10]], [6, 5, 4, 3, 2])

    # 1d __setitem__

    # can write through a single advanced slice
    a = lg.arange(10)
    b = lg.arange(5)
    a[range(5, 10)] = b
    assert np.array_equal(a, [0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

    # can write through views
    a = np.arange(10)
    b = np.arange(5)  # TODO: RHS cannot be a view currently
    c = np.arange(9, -1, -1)
    a[2:7][c[5:10]] = b
    assert np.array_equal(a, [0, 1, 4, 3, 2, 1, 0, 7, 8, 9])

    a = lg.arange(20)
    b = np.arange(10)
    a[10:][range(5)] = b[2:7]
    assert np.array_equal(
        a, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19]
    )

    # can copy within the same array
    # TODO: Fix #40
    # a = lg.arange(20)
    # a[10:][range(5)] = a[2:7]
    # assert np.array_equal(
    #     a, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19]
    # )

    # source & destination regions can (partially) overlap
    # TODO: Fix #40
    # a = lg.arange(20)
    # a[10:][range(5)] = a[12:17]
    # assert np.array_equal(
    #     a,
    #     [0,1,2,3,4,5,6,7,8,9,12,13,14,15,16,15,16,17,18,19],
    # )

    # 2d __getitem__

    # index arrays can be boolean, of the same shape as the base array
    a = sequence_2d()
    # TODO: Have to sort the output array, until #50 is fixed
    assert np.array_equal(np.sort(a[a > 17]), [18, 19, 20, 21, 22, 23, 24])

    # index arrays can be integer, one per base array dimension
    a = sequence_2d()
    assert np.array_equal(
        a[lg.array([0, 1, 2, 3]), lg.array([0, 1, 2, 3])], [0, 6, 12, 18]
    )

    # output shape follows index array shape
    a = sequence_2d()
    assert np.array_equal(
        a[lg.array([[1, 1, 1], [2, 2, 2]]), lg.array([[0, 1, 2], [0, 1, 2]])],
        [[5, 6, 7], [10, 11, 12]],
    )

    # index arrays can be any sequence object
    a = sequence_2d()
    assert np.array_equal(a[range(4), range(4)], [0, 6, 12, 18])

    # index arrays are automatically cast to int64
    a = sequence_2d()
    assert np.array_equal(a[range(4), range(4)], [0, 6, 12, 18])
    assert np.array_equal(
        a[
            lg.array([0, 1, 2, 3], dtype=np.int16),
            lg.array([0, 1, 2, 3], dtype=np.uint32),
        ],
        [0, 6, 12, 18],
    )

    # advanced slicing creates copies
    a = sequence_2d()
    b = a[a > 17]
    b[:] = -1
    assert a.min() == 0 and a.max() == 24

    # can read through views
    a = sequence_2d()
    cx = lg.array([0, 0, 1, 1, 2, 2])
    cy = lg.array([1, 2, 1, 2, 1, 2])
    assert np.array_equal(a[2:, 2:][cx[2:], cy[2:]], [18, 19, 23, 24])

    # 2d __setitem__

    # can write through a single advanced slice
    a = sequence_2d()
    b = lg.zeros(7, dtype=np.int64)
    a[a > 17] = b
    assert np.array_equal(
        a,
        [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    )

    # can write through views
    a = sequence_2d()
    b = lg.array([100, 101, 102, 103])  # TODO: RHS cannot be a view currently
    cx = lg.array([0, 0, 1, 1, 2, 2])
    cy = lg.array([1, 2, 1, 2, 1, 2])
    a[2:, 2:][cx[2:], cy[2:]] = b
    assert np.array_equal(
        a,
        [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 100, 101],
            [20, 21, 22, 102, 103],
        ],
    )

    # can copy within the same array
    # TODO: Fix #40
    # a = sequence_2d()
    # a[2:, :][[[1, 1], [2, 2]], [[1, 2], [1, 2]]] = a[1:3, 3:5]
    # assert np.array_equal(
    #     a,
    #     [
    #         [0, 1, 2, 3, 4],
    #         [5, 6, 7, 8, 9],
    #         [10, 11, 12, 13, 14],
    #         [15, 8, 9, 18, 19],
    #         [20, 13, 14, 23, 24],
    #     ],
    # )

    # source & destination regions can (partially) overlap
    # TODO: Fix #40
    # a = sequence_2d()
    # a[2:, :][[[1, 1], [2, 2]], [[1, 2], [1, 2]]] = a[3:5, 2:4]
    # assert np.array_equal(
    #     a,
    #     [
    #         [0, 1, 2, 3, 4],
    #         [5, 6, 7, 8, 9],
    #         [10, 11, 12, 13, 14],
    #         [15, 17, 18, 18, 19],
    #         [20, 22, 23, 23, 24],
    #     ],
    # )

    # TODO: broadcasting in index arrays
    # TODO: mixed advanced indexing
    # TODO: singleton arrays
    # TODO: views as base, index or value array (incl. reshape)

    return


if __name__ == "__main__":
    test()
